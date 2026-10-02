'''Episodic meta-trainer for FSAR baselines.

Single entry point ``main(argv)``. Reads a YAML config (see
``experiments/configs/protonet_rgb_5w5s.yaml`` for the schema), assembles a
:class:`src.data.loader.ThetisDataset`, samples N-way K-shot episodes via
:class:`src.data.episode_sampler.EpisodeSampler`, and meta-trains the head
named by ``method`` (see :data:`src.models.factory.METHODS`) with periodic
eval on ``meta_val``. Checkpoints land in
``outputs/checkpoints/<run_id>/`` and a JSON training log in
``experiments/logs/<run_id>/training.json``.

Smoke mode (``--smoke``) shrinks epochs/episodes to ~seconds for CI; it also
forces ``pretrained=False`` on the encoder so no network download is required.

``last.pt`` and the JSON log are rewritten every epoch, so a run killed by a
hosted-runtime time cap (Kaggle/Colab) can be continued with ``--resume``.
'''
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
import yaml

if __package__ in (None, ''):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from src.data.augment import (  # noqa: E402
    CenterSpatialCrop,
    ColorJitter,
    Compose,
    HorizontalFlip,
    RandomSegmentSample,
    RandomSpatialCrop,
    RandomTemporalCrop,
    ResizeVideo,
)
from src.data.episode_sampler import EpisodeSampler, split_classes  # noqa: E402
from src.data.loader import (  # noqa: E402
    ACTION_LABELS,
    MODALITY_KEY,
    PATH_COLUMNS,
    ThetisDataset,
)
from src.data.loader import load_exclusions  # noqa: E402
from src.models.base import EpisodicModel  # noqa: E402
from src.models.factory import METHODS, build_encoder, build_model  # noqa: E402
from src.utils.metrics import accuracy_with_ci  # noqa: E402


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    '''Load a YAML config file into a plain dict, with minimal validation.'''
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f'config root must be a mapping: {path}')
    for required in ('method', 'modalities', 'episode', 'optim', 'data', 'seed'):
        if required not in cfg:
            raise KeyError(f'missing required config key: {required!r}')
    if cfg['method'] not in METHODS:
        raise NotImplementedError(f'method {cfg["method"]!r} not implemented; available: {list(METHODS)}')
    modalities = cfg['modalities']
    if not modalities or not isinstance(modalities, list):
        raise ValueError('config.modalities must be a non-empty list')
    if len(modalities) != 1:
        raise NotImplementedError('Phase 2 baseline only supports a single modality per config')
    if modalities[0] not in PATH_COLUMNS:
        raise ValueError(f'unknown modality {modalities[0]!r}; valid: {list(PATH_COLUMNS)}')
    return cfg


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str | None) -> torch.device:
    if requested in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def configure_cuda_memory(device: torch.device, fraction: float = 0.92) -> float | None:
    '''Pre-flight CUDA tuning: empty cache, cap memory fraction, request expandable segments.

    On Windows, CUDA happily spills into system RAM (shared GPU memory) when
    dedicated VRAM is full — slow and unreliable. Capping the per-process
    memory fraction forces a fail-fast at the real VRAM limit so OOM errors
    are honest. Returns the cap in GiB (or ``None`` on CPU).
    '''
    if device.type != 'cuda':
        return None
    import os  # noqa: PLC0415

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        # Cap the per-process fraction so OOM fails fast at the real VRAM limit
        # (keeps headroom for cuDNN workspaces / avoids Windows shared-memory spill).
        torch.cuda.set_per_process_memory_fraction(float(fraction), device.index or 0)
        return total_gb
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Episode-aware sample loading
# ---------------------------------------------------------------------------


class EpisodeLoader:
    '''Resolve ``sample_id`` lists into stacked ``(B, T, H, W, 3)`` tensors.

    Wraps a :class:`ThetisDataset` plus an ``id → row_idx`` map so the
    trainer can ask for support/query batches by ID. Applies the configured
    transform (train or eval) per sample before stacking.
    '''

    def __init__(
        self,
        dataset: ThetisDataset,
        modality: str,
        transform: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.modality = modality
        self.tensor_key = MODALITY_KEY[modality]
        self.transform = transform
        # ThetisDataset re-indexes; build the lookup against ITS df, not the manifest.
        self.id_to_idx: dict[str, int] = {
            str(sid): i for i, sid in enumerate(self.dataset.df['sample_id'])
        }

    def load(self, sample_ids: Sequence[str]) -> torch.Tensor:
        videos: list[torch.Tensor] = []
        for sid in sample_ids:
            idx = self.id_to_idx.get(str(sid))
            if idx is None:
                raise KeyError(f'sample_id {sid!r} not in dataset for modality {self.modality!r}')
            sample = self.dataset[idx]
            if self.transform is not None:
                sample = self.transform(sample)
            tensor = sample[self.tensor_key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(np.ascontiguousarray(tensor))
            videos.append(tensor)
        return torch.stack(videos, dim=0)


def assemble_episode_tensors(
    loader: EpisodeLoader,
    episode: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    '''Materialise an episode dict into support/query tensors with labels.

    The class index used in labels is the local index of the class within the
    episode's sorted class list (so labels are in ``[0, n_way)``).
    '''
    classes: list[str] = list(episode['classes'])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    support_ids: list[str] = []
    support_labels: list[int] = []
    for cls in classes:
        for sid in episode['support'][cls]:
            support_ids.append(sid)
            support_labels.append(class_to_idx[cls])

    query_ids: list[str] = []
    query_labels: list[int] = []
    for cls in classes:
        for sid in episode['query'][cls]:
            query_ids.append(sid)
            query_labels.append(class_to_idx[cls])

    support = loader.load(support_ids).to(device, non_blocking=True)
    query = loader.load(query_ids).to(device, non_blocking=True)
    return {
        'support': support,
        'query': query,
        'support_labels': torch.tensor(support_labels, dtype=torch.long, device=device),
        'query_labels': torch.tensor(query_labels, dtype=torch.long, device=device),
        'n_way': torch.tensor(len(classes), dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


TEMPORAL_SAMPLERS = {'crop': RandomTemporalCrop, 'segment': RandomSegmentSample}


def temporal_oversample(cfg_data: dict[str, Any]) -> int:
    '''Factor by which the train dataset over-decodes ``frame_count`` before the temporal step.'''
    factor = int(cfg_data.get('temporal_oversample', 2))
    if factor < 1:
        raise ValueError(f'data.temporal_oversample must be >= 1, got {factor}')
    return factor


def build_train_transform(cfg_data: dict[str, Any], seed: int) -> Compose:
    '''Train-time augmentation. ``data.temporal_sampling`` picks the temporal step:

    - ``crop`` (default): contiguous window of ``frame_count`` out of the
      over-decoded clip — covers ``1/temporal_oversample`` of the stroke.
    - ``segment``: one random frame per segment — always spans the whole clip.

    Evaluation is unaffected either way (``build_eval_transform`` has no
    temporal step; the eval dataset decodes ``frame_count`` uniformly).
    '''
    frames = int(cfg_data.get('frame_count', 16))
    resize = int(cfg_data.get('resize_size', 128))
    crop = int(cfg_data.get('spatial_size', 112))
    sampling = cfg_data.get('temporal_sampling', 'crop')
    if sampling not in TEMPORAL_SAMPLERS:
        raise ValueError(f'unknown data.temporal_sampling {sampling!r}; valid: {list(TEMPORAL_SAMPLERS)}')
    return Compose([
        TEMPORAL_SAMPLERS[sampling](num_frames=frames, seed=seed),
        ResizeVideo(size=resize),
        RandomSpatialCrop(size=crop, seed=seed + 1),
        HorizontalFlip(p=0.5, seed=seed + 2),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, seed=seed + 3),
    ])


def build_eval_transform(cfg_data: dict[str, Any]) -> Compose:
    resize = int(cfg_data.get('resize_size', 128))
    crop = int(cfg_data.get('spatial_size', 112))
    return Compose([
        ResizeVideo(size=resize),
        CenterSpatialCrop(size=crop),
    ])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _make_sampler(
    manifest_path: str | Path,
    splits: dict[str, list[str]],
    seed: int,
    n_way: int,
    k_shot: int,
    q_query: int,
    modality: str,
    exclude: frozenset[str] | None = None,
) -> EpisodeSampler:
    return EpisodeSampler(
        manifest_path=manifest_path,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        splits=splits,
        seed=seed,
        speed_split='none',
        modality=modality,
        exclude=exclude,
        strict=False,
    )


def _run_eval(
    model: EpisodicModel,
    sampler: EpisodeSampler,
    loader: EpisodeLoader,
    split: str,
    n_episodes: int,
    n_way: int,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    '''Run ``n_episodes`` from ``split``; return ``(mean_acc, ci_half_width, mean_loss)``.

    The model forward already computes the query cross-entropy, so ``mean_loss``
    is essentially free here — it just averages ``out['loss']`` across episodes.
    Useful to catch overfitting when val_acc plateaus but val_loss creeps back up.
    '''
    model.eval()
    accs: list[float] = []
    losses: list[float] = []
    autocast_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        if use_amp and device.type == 'cuda'
        else _nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        for i in range(n_episodes):
            ep = sampler.sample_episode(split, i)
            tensors = assemble_episode_tensors(loader, ep, device)
            out = model(
                tensors['support'],
                tensors['query'],
                tensors['support_labels'],
                tensors['query_labels'],
                n_way=n_way,
            )
            accs.append(out['accuracy'])
            losses.append(float(out['loss'].detach()))
    mean, ci = accuracy_with_ci(accs)
    mean_loss = float(np.mean(losses)) if losses else float('nan')
    return mean, ci, mean_loss


def _mem_profile_enabled() -> bool:
    '''Opt-in CUDA memory profiling, toggled by ``THETIS_MEM_PROFILE=1``.'''
    import os  # noqa: PLC0415

    return os.environ.get('THETIS_MEM_PROFILE', '') not in ('', '0', 'false', 'False')


def _log_cuda_mem(tag: str, device: torch.device) -> None:
    '''Print alloc/reserved/peak MiB for ``device`` (no-op off CUDA).'''
    if device.type != 'cuda':
        return
    mib = 1024 * 1024
    alloc = torch.cuda.memory_allocated() / mib
    reserved = torch.cuda.memory_reserved() / mib
    peak = torch.cuda.max_memory_allocated() / mib
    print(f'[mem] {tag}: alloc={alloc:.0f}MiB reserved={reserved:.0f}MiB peak={peak:.0f}MiB', flush=True)


def _train_step_streaming(
    model: EpisodicModel,
    tensors: dict[str, torch.Tensor],
    n_way: int,
    optimizer: optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    device: torch.device,
    profile: bool = False,
) -> tuple[float, float]:
    '''One training step with the query set streamed in micro-batches.

    Encodes the (small) support set once to build the head's support state
    (ProtoNet: prototypes), then walks the query set ``encoder_batch_size`` clips
    at a time, calling ``backward`` per micro-batch with ``retain_graph`` so the
    shared support graph survives until the last one. Peak activation memory is
    ~``support + one micro-batch`` instead of the whole episode. For a BatchNorm-free
    encoder the accumulated gradient is mathematically identical to a single
    full-batch ``backward`` (each micro-batch contributes ``sum(ce) / n_query``, so
    the accumulated ``.grad`` equals the mean-CE gradient), given a head whose
    ``query_logits`` scores each query independently (the EpisodicModel contract).
    With BatchNorm in ``train()`` mode it is *not* identical: the support set and
    each query micro-batch are normalised as separate batches, whereas the classic
    path normalises mixed support+query chunks — hence ``stream_query`` is part of
    the experimental protocol. Returns
    ``(episode_loss, episode_accuracy)`` matching ``EpisodicModel.forward`` semantics.
    '''
    optimizer.zero_grad(set_to_none=True)
    support = tensors['support']
    query = tensors['query']
    support_labels = tensors['support_labels']
    query_labels = tensors['query_labels']
    amp_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        if use_amp
        else _nullcontext()
    )

    with amp_ctx:
        support_emb = model.encoder(support)
        state = model.support_state(support_emb, support_labels, n_way)
    if profile:
        _log_cuda_mem('after-support', device)

    n_query = int(query.shape[0])
    micro = model.encoder_batch_size or n_query
    total_loss = 0.0
    correct = 0
    for start in range(0, n_query, micro):
        end = min(start + micro, n_query)
        retain = end < n_query
        labels_mb = query_labels[start:end]
        with amp_ctx:
            query_emb = model.encoder(query[start:end])
            logits = model.query_logits(query_emb, state)
            # reduction='sum' / n_query reproduces the full-episode mean CE exactly.
            loss_mb = F.cross_entropy(logits, labels_mb, reduction='sum') / n_query
        if scaler is not None:
            scaler.scale(loss_mb).backward(retain_graph=retain)
        else:
            loss_mb.backward(retain_graph=retain)
        total_loss += float(loss_mb.detach())
        correct += int((logits.detach().argmax(dim=-1) == labels_mb).sum().item())

    if profile:
        _log_cuda_mem('after-query-loop', device)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return total_loss, correct / n_query


def run_training(
    cfg: dict[str, Any],
    smoke: bool,
    device_arg: str | None,
    resume: bool | str | Path | None = None,
) -> dict[str, Any]:
    '''Execute the meta-training loop described by ``cfg``.

    ``resume`` continues an interrupted run: ``'auto'`` (or ``True``) picks up
    ``<ckpt_dir>/last.pt``, starting from scratch when there is none; anything
    else is treated as an explicit checkpoint path. Episodes are addressed by
    index (``(epoch - 1) * episodes_per_epoch + i``) and the sampler is
    deterministic per index, so a resumed run replays the same episode stream
    as an uninterrupted one — only the augmentation RNG restarts.

    Returns the in-memory training log so tests can assert on it without
    re-reading from disk.
    '''
    seed = int(cfg['seed'])
    set_global_seed(seed)
    device = select_device(device_arg)
    configure_cuda_memory(device, float(cfg.get('optim', {}).get('cuda_memory_fraction', 0.92)))

    modality = cfg['modalities'][0]
    data_cfg = cfg['data']
    manifest_path = Path(data_cfg['manifest_path']).resolve()
    dataset_root = Path(data_cfg['dataset_root']).resolve()
    frame_count = int(data_cfg.get('frame_count', 16))

    # Class universe: prefer what's actually in the manifest (allows synthetic
    # mini-manifests in tests), fall back to the canonical 12 labels.
    manifest_classes = sorted(
        pd.read_csv(manifest_path, usecols=['action_label'])['action_label'].unique()
    )
    universe = manifest_classes if manifest_classes else list(ACTION_LABELS)
    splits = split_classes(
        universe,
        train_n=int(data_cfg.get('train_classes', 6)),
        val_n=int(data_cfg.get('val_classes', 3)),
        test_n=int(data_cfg.get('test_classes', 3)),
        seed=seed,
    )

    ep_cfg = cfg['episode']
    n_way_train = int(ep_cfg['n_way'])
    n_way_val = int(ep_cfg.get('n_way_val', n_way_train))
    k_shot = int(ep_cfg['k_shot'])
    q_query = int(ep_cfg['q_query'])

    if smoke:
        # Smoke mode: every dimension shrunk so the pipeline finishes in seconds
        # on CPU. The goal is to surface integration bugs, not to learn anything.
        episodes_per_epoch = 1
        episodes_meta_val = 1
        epochs = 1
        k_shot = min(k_shot, 2)
        q_query = min(q_query, 2)
        n_way_train = min(n_way_train, 2)
        n_way_val = min(n_way_val, 2)
        smoke_data_overrides = {
            'frame_count': 4,
            'resize_size': 72,
            'spatial_size': 64,
        }
        for k, v in smoke_data_overrides.items():
            data_cfg[k] = v
        frame_count = smoke_data_overrides['frame_count']
    else:
        episodes_per_epoch = int(ep_cfg.get('episodes_per_epoch', 200))
        episodes_meta_val = int(ep_cfg.get('episodes_meta_val', 100))
        epochs = int(cfg['optim'].get('epochs', 100))

    # Blank clips and byte-identical duplicates shipped by THETIS — the latter can
    # otherwise put the same footage in an episode's support and query set. Built
    # by `make preprocess --full-integrity`; absent file = no filter.
    exclude_defective = bool(data_cfg.get('exclude_defective', True))
    excluded = load_exclusions(manifest_path, modality, enabled=exclude_defective)
    if excluded:
        print(f'[setup] excluding {len(excluded)} defective {modality} clips', flush=True)
    elif exclude_defective:
        print(f'[setup] no exclusions found for {modality} (run preprocess --full-integrity)', flush=True)

    train_sampler = _make_sampler(
        manifest_path, splits, seed, n_way_train, k_shot, q_query, modality, excluded
    )
    val_eligible = len(splits['meta_val']) >= n_way_val
    val_sampler = (
        _make_sampler(manifest_path, splits, seed + 1, n_way_val, k_shot, q_query, modality, excluded)
        if val_eligible
        else None
    )

    train_transform = build_train_transform(data_cfg, seed)
    eval_transform = build_eval_transform(data_cfg)

    # In-memory decode cache: episodic sampling re-draws the same small class
    # pool (~990 RGB clips over 6 meta_train classes) thousands of times, so the
    # serial per-episode video decode — not the GPU — is the real bottleneck.
    # Decoding each clip once and resizing it up front to resize_size (matching
    # the ResizeVideo step in the transforms, so results are byte-identical)
    # serves every later access from RAM. Cached clips are ~1.5 MB at 128^2 vs
    # ~29 MB at native 480x640, so the full train+val pool fits in ~1.9 GB.
    # Disable via data.cache_decoded=false if RAM-constrained.
    resize_size = int(data_cfg.get('resize_size', 128))
    cache_decoded = bool(data_cfg.get('cache_decoded', True))
    cache_resize = resize_size if cache_decoded else None

    train_dataset = ThetisDataset(
        manifest_path=manifest_path,
        modalities=[modality],
        dataset_root=dataset_root,
        transform=None,  # transform applied inside EpisodeLoader to keep eval-vs-train clean
        # over-sample, then the temporal step (crop or segment) brings it back to frame_count
        frame_count=frame_count * temporal_oversample(data_cfg) if not smoke else frame_count,
        return_tensors=True,
        cache=cache_decoded,
        cache_resize=cache_resize,
    )
    eval_dataset = ThetisDataset(
        manifest_path=manifest_path,
        modalities=[modality],
        dataset_root=dataset_root,
        transform=None,
        frame_count=frame_count,
        return_tensors=True,
        cache=cache_decoded,
        cache_resize=cache_resize,
    )

    train_loader = EpisodeLoader(train_dataset, modality=modality, transform=train_transform)
    eval_loader = EpisodeLoader(eval_dataset, modality=modality, transform=eval_transform)

    encoder_cfg = cfg.get('encoder', {})
    pretrained = bool(encoder_cfg.get('pretrained', True)) and not smoke
    # Auto-enable gradient checkpointing on GPUs ≤ 6 GB unless the config
    # explicitly overrides. Saves ~70% activation memory at ~30% extra time.
    auto_checkpoint = False
    if device.type == 'cuda':
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            auto_checkpoint = total_gb < 7.0
        except Exception:  # noqa: BLE001
            pass
    use_checkpointing = bool(encoder_cfg.get('gradient_checkpointing', auto_checkpoint))
    encoder = build_encoder(encoder_cfg, pretrained=pretrained, use_checkpointing=use_checkpointing)
    # Chunked encoding: cap activations memory by running the encoder on at most
    # this many videos per forward. Critical on CPU where R(2+1)D-18 with batch=100
    # silently kills the process and on small GPUs (≤6 GB) where batch=32 OOMs.
    encoder_batch_default = _default_encoder_batch_size(device, smoke=smoke)
    encoder_batch_size = int(encoder_cfg.get('batch_size', encoder_batch_default))
    model = build_model(cfg, encoder, encoder_batch_size=encoder_batch_size).to(device)

    # Mixed precision: fp16 autocast on CUDA cuts activations roughly in half
    # and is the easiest knob to keep R(2+1)D-18 fitting in 4 GB GPUs.
    use_amp = bool(cfg.get('optim', {}).get('fp16', device.type == 'cuda')) and device.type == 'cuda'
    # Query streaming (grad accumulation) bounds training activation memory to
    # ~support + one micro-batch, the real fix for R(2+1)D-18 OOMs. Default on for
    # CUDA; classic full-batch backward otherwise (e.g. CPU smoke). Override via
    # optim.stream_query — but it changes the BatchNorm batches, so keep it fixed
    # across compared runs.
    stream_query = bool(cfg.get('optim', {}).get('stream_query', device.type == 'cuda'))

    optim_cfg = cfg['optim']
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get('learning_rate', 1e-4)),
        weight_decay=float(optim_cfg.get('weight_decay', 0.0)),
    )
    eval_every = int(optim_cfg.get('eval_every', 5))
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    vram_str = ''
    if device.type == 'cuda':
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            vram_str = f' (VRAM~{total_gb:.1f} GB)'
        except Exception:  # noqa: BLE001
            pass

    print(
        f'[setup] device={device}{vram_str} method={cfg["method"]} encoder={encoder_cfg.get("name", "r2plus1d_18")} '
        f'pretrained={pretrained} encoder_batch_size={encoder_batch_size} fp16={use_amp} '
        f'grad_checkpoint={use_checkpointing} stream_query={stream_query} '
        f'n_way_train={n_way_train} n_way_val={n_way_val} k_shot={k_shot} q_query={q_query} '
        f'episodes_per_epoch={episodes_per_epoch} epochs={epochs} smoke={smoke}',
        flush=True,
    )
    print(
        f'[setup] decode_cache={cache_decoded} cache_resize={cache_resize} '
        f'(each clip decoded from disk once at epoch 1, then served from RAM)',
        flush=True,
    )
    print(f'[setup] splits = {splits}', flush=True)

    run_id = cfg.get('run_id') or f'{cfg["method"]}_{modality}_{k_shot}s_{_timestamp()}'
    if smoke:
        run_id = f'smoke_{run_id}'
    out_root = Path(cfg.get('output_root', 'outputs')).resolve()
    log_root = Path(cfg.get('log_root', 'experiments/logs')).resolve()
    ckpt_dir = out_root / 'checkpoints' / run_id
    log_dir = log_root / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log: dict[str, Any] = {
        'run_id': run_id,
        'config': cfg,
        'splits': splits,
        'n_way_per_split': {'meta_train': n_way_train, 'meta_val': n_way_val},
        'started_at': _timestamp(),
        'smoke': bool(smoke),
        'device': str(device),
        'epochs': [],
        'best_val_acc': None,
        'best_epoch': None,
    }

    # Verbose prints in smoke mode (every episode); compact otherwise (every ~10%).
    progress_every = 1 if smoke else max(1, episodes_per_epoch // 10)

    mem_profile = _mem_profile_enabled()
    best_val = float('-inf')

    # Resume support: hosted runtimes (Kaggle/Colab) cap a session well below a
    # full 100-epoch run, so pick the run back up from the epoch after last.pt.
    start_epoch = 1
    if resume:
        auto = resume is True or str(resume) == 'auto'
        resume_path = ckpt_dir / 'last.pt' if auto else Path(resume)
        if not resume_path.is_file():
            if not auto:
                raise FileNotFoundError(f'resume checkpoint not found: {resume_path}')
            print(f'[resume] no checkpoint at {resume_path} — starting from epoch 1', flush=True)
        else:
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            if scaler is not None and state.get('scaler_state') is not None:
                scaler.load_state_dict(state['scaler_state'])
            start_epoch = int(state['epoch']) + 1
            prior = state.get('log') or {}
            log['epochs'] = list(prior.get('epochs', []))
            log['best_val_acc'] = prior.get('best_val_acc')
            log['best_epoch'] = prior.get('best_epoch')
            log['started_at'] = prior.get('started_at', log['started_at'])
            log['resumed_from'] = {'checkpoint': str(resume_path), 'at_epoch': start_epoch}
            if log['best_val_acc'] is not None:
                best_val = float(log['best_val_acc'])
            if start_epoch > epochs:
                print(
                    f'[resume] {resume_path} already covers all {epochs} epochs — nothing to do',
                    flush=True,
                )
            else:
                print(
                    f'[resume] {resume_path} -> continuing at epoch {start_epoch}/{epochs} '
                    f'(best_val={log["best_val_acc"]} @ep{log["best_epoch"]})',
                    flush=True,
                )

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        for i in range(episodes_per_epoch):
            ep_idx = (epoch - 1) * episodes_per_epoch + i
            ep = train_sampler.sample_episode('meta_train', ep_idx)
            tensors = assemble_episode_tensors(train_loader, ep, device)
            profile = mem_profile and epoch == 1 and i < 3
            if profile and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                _log_cuda_mem('episode-start', device)
            if stream_query:
                step_loss, step_acc = _train_step_streaming(
                    model, tensors, n_way_train, optimizer, scaler, use_amp, device, profile=profile
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        out = model(
                            tensors['support'],
                            tensors['query'],
                            tensors['support_labels'],
                            tensors['query_labels'],
                            n_way=n_way_train,
                        )
                    scaler.scale(out['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(
                        tensors['support'],
                        tensors['query'],
                        tensors['support_labels'],
                        tensors['query_labels'],
                        n_way=n_way_train,
                    )
                    out['loss'].backward()
                    optimizer.step()
                step_loss = float(out['loss'].detach().cpu().item())
                step_acc = out['accuracy']
                del out
            if profile:
                _log_cuda_mem('episode-end', device)
            epoch_losses.append(step_loss)
            epoch_accs.append(step_acc)
            del tensors
            if (i + 1) % progress_every == 0 or (i + 1) == episodes_per_epoch:
                running_loss = float(np.mean(epoch_losses))
                running_acc = float(np.mean(epoch_accs))
                print(
                    f'[epoch {epoch}/{epochs}] {i + 1:>4}/{episodes_per_epoch}  '
                    f'loss={running_loss:.4f} acc={running_acc:.4f}',
                    flush=True,
                )

        epoch_record: dict[str, Any] = {
            'epoch': epoch,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else float('nan'),
            'train_acc': float(np.mean(epoch_accs)) if epoch_accs else float('nan'),
        }

        run_eval = val_sampler is not None and (epoch == epochs or epoch % eval_every == 0 or smoke)
        if run_eval and val_sampler is not None:
            print(f'[epoch {epoch}] running val eval ({episodes_meta_val} episodes)...', flush=True)
            val_mean, val_ci, val_loss = _run_eval(
                model, val_sampler, eval_loader, 'meta_val', episodes_meta_val, n_way_val, device, use_amp=use_amp
            )
            epoch_record['val_acc'] = val_mean
            epoch_record['val_ci95'] = val_ci
            epoch_record['val_loss'] = val_loss
            if val_mean > best_val:
                best_val = val_mean
                log['best_val_acc'] = val_mean
                log['best_epoch'] = epoch
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_acc': val_mean,
                        'val_ci95': val_ci,
                        'val_loss': val_loss,
                        'config': cfg,
                    },
                    ckpt_dir / 'best.pt',
                )

        summary = (
            f'epoch {epoch:>3}/{epochs} | train_loss={epoch_record["train_loss"]:.4f} '
            f'train_acc={epoch_record["train_acc"]:.4f}'
        )
        if 'val_acc' in epoch_record:
            summary += (
                f' | val_loss={epoch_record["val_loss"]:.4f} '
                f'val_acc={epoch_record["val_acc"]:.4f}'
            )
        if best_val > float('-inf'):
            summary += f' | best_val={best_val:.4f} @ep{log["best_epoch"]}'
        # Wall time per epoch: the number you need to budget epochs against a
        # hosted-runtime session cap (Kaggle/Colab).
        epoch_record['seconds'] = round(time.perf_counter() - epoch_t0, 1)
        summary += f' | {epoch_record["seconds"]:.0f}s'
        print(summary)
        log['epochs'].append(epoch_record)

        # Rewrite last.pt + the JSON log every epoch (via a temp file, so a kill
        # mid-write can't corrupt them) — that is what makes --resume possible
        # after a session time cap.
        _atomic_torch_save(
            {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict() if scaler is not None else None,
                'config': cfg,
                'log': log,
            },
            ckpt_dir / 'last.pt',
        )
        _atomic_write_text(
            log_dir / 'training.json', json.dumps(log, indent=2, default=_json_default)
        )

    log['finished_at'] = _timestamp()

    _atomic_write_text(log_dir / 'training.json', json.dumps(log, indent=2, default=_json_default))
    print(f'wrote training log to {log_dir / "training.json"}')
    print(f'checkpoints in {ckpt_dir}')
    return log


def _timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    '''``torch.save`` to ``path`` via a sibling temp file, then rename.'''
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _default_encoder_batch_size(device: torch.device, smoke: bool) -> int:
    '''Pick a chunked-encoding batch size that won't OOM on common GPU sizes.

    Falls back to 4 on CPU (R(2+1)D-18 fp32 activations get huge fast) and
    scales up with available VRAM on CUDA.
    '''
    if smoke:
        return 4
    if device.type == 'cpu':
        return 4
    try:
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 8
    if total_gb < 6:
        return 4
    if total_gb < 10:
        return 8
    if total_gb < 16:
        return 16
    return 32


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict()
    raise TypeError(f'unserialisable type {type(value).__name__}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Meta-train an FSAR model (method set by the config).')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', type=str, default=None, help='cuda | cpu | auto (default: auto)')
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Smoke mode: 1 epoch, 2 episodes/epoch, random-init encoder. Verifies the pipeline end-to-end.',
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        default=None,
        metavar='CHECKPOINT',
        help='Continue an interrupted run. Bare --resume reads '
        '<output_root>/checkpoints/<run_id>/last.pt (no-op if absent); '
        'pass a path to resume from a specific checkpoint.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_training(cfg, smoke=args.smoke, device_arg=args.device, resume=args.resume)
    return 0


__all__ = [
    'EpisodeLoader',
    'assemble_episode_tensors',
    'build_eval_transform',
    'build_train_transform',
    'load_config',
    'main',
    'run_training',
    'set_global_seed',
]


if __name__ == '__main__':
    raise SystemExit(main())
