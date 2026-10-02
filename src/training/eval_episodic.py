'''Episodic evaluation of a trained FSAR checkpoint on meta_test.

Loads ``best.pt`` (or any checkpoint produced by :mod:`meta_trainer`), reads
either pre-serialised episodes from ``data/episodes/meta_test/episodes.jsonl``
OR re-samples episodes with the saved config, then reports

- per-episode accuracies,
- mean ± 95% CI (Student-t),
- per-class confusion matrix.

Outputs land under ``outputs/results/<run_id>/`` as ``metrics.json`` plus a
confusion-matrix PNG.
'''
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ''):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from src.data.episode_sampler import EpisodeSampler  # noqa: E402
from src.data.loader import (  # noqa: E402
    ACTION_LABELS,
    PATH_COLUMNS,
    ThetisDataset,
    load_exclusions,
)
from src.models.factory import build_encoder, build_model  # noqa: E402
from src.training.meta_trainer import (  # noqa: E402
    EpisodeLoader,
    assemble_episode_tensors,
    build_eval_transform,
    select_device,
    set_global_seed,
)
from src.utils.metrics import accuracy_with_ci, per_class_confusion  # noqa: E402
from src.utils.viz import plot_confusion  # noqa: E402


def _iter_episodes_from_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_episodes_from_sampler(
    cfg: dict[str, Any],
    n_episodes: int,
) -> tuple[list[dict[str, Any]], int]:
    '''Re-sample meta_test episodes using the saved config.

    Returns ``(episodes, n_way)``. Falls back to ``n_way`` from ``episode.n_way``
    when the config does not specify a test-time override.
    '''
    import pandas as pd  # noqa: PLC0415

    from src.data.episode_sampler import split_classes  # noqa: PLC0415

    seed = int(cfg['seed'])
    manifest_classes = sorted(
        pd.read_csv(Path(cfg['data']['manifest_path']).resolve(), usecols=['action_label'])['action_label'].unique()
    )
    universe = manifest_classes if manifest_classes else list(ACTION_LABELS)
    splits = split_classes(
        universe,
        train_n=int(cfg['data'].get('train_classes', 6)),
        val_n=int(cfg['data'].get('val_classes', 3)),
        test_n=int(cfg['data'].get('test_classes', 3)),
        seed=seed,
    )
    ep_cfg = cfg['episode']
    n_way_test = int(ep_cfg.get('n_way_test', ep_cfg['n_way']))
    manifest_path = Path(cfg['data']['manifest_path']).resolve()
    modality = cfg['modalities'][0]
    sampler = EpisodeSampler(
        manifest_path=manifest_path,
        n_way=n_way_test,
        k_shot=int(ep_cfg['k_shot']),
        q_query=int(ep_cfg['q_query']),
        splits=splits,
        seed=seed + 2,
        speed_split='none',
        modality=modality,
        # same filter as training, so meta_test is not scored on blank clips or
        # on footage duplicated into both support and query
        exclude=load_exclusions(
            manifest_path, modality, enabled=bool(cfg['data'].get('exclude_defective', True))
        ),
        strict=False,
    )
    eps = [sampler.sample_episode('meta_test', i) for i in range(n_episodes)]
    return eps, n_way_test


def run_eval(
    checkpoint_path: str | Path,
    episodes_jsonl: str | Path | None,
    output_root: str | Path,
    n_episodes: int | None,
    device_arg: str | None,
) -> dict[str, Any]:
    '''Evaluate a checkpoint against ``meta_test`` episodes.'''
    ckpt_path = Path(checkpoint_path).resolve()
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg: dict[str, Any] = blob['config']
    modality = cfg['modalities'][0]
    if modality not in PATH_COLUMNS:
        raise ValueError(f'unknown modality {modality!r}')

    seed = int(cfg['seed'])
    set_global_seed(seed)
    device = select_device(device_arg)

    encoder_cfg = cfg.get('encoder', {})
    encoder = build_encoder(encoder_cfg, pretrained=False)
    # Chunked encoding only bounds memory here: in eval() BatchNorm uses running
    # stats, so the chunk size does not change the embeddings.
    model = build_model(cfg, encoder, encoder_batch_size=encoder_cfg.get('batch_size')).to(device)
    model.load_state_dict(blob['model_state'])
    model.eval()

    eval_transform = build_eval_transform(cfg['data'])
    # Same decode cache as training: meta_test re-samples the small test-class
    # pool (~495 clips) across ~1000 episodes, so caching each clip once —
    # resized to resize_size, which matches build_eval_transform's ResizeVideo,
    # so results are byte-identical — removes the redundant-decode cost (~0.4 GB
    # RAM). Honours data.cache_decoded from the saved config (default on).
    resize_size = int(cfg['data'].get('resize_size', 128))
    cache_decoded = bool(cfg['data'].get('cache_decoded', True))
    dataset = ThetisDataset(
        manifest_path=Path(cfg['data']['manifest_path']).resolve(),
        modalities=[modality],
        dataset_root=Path(cfg['data']['dataset_root']).resolve(),
        transform=None,
        frame_count=int(cfg['data'].get('frame_count', 16)),
        return_tensors=True,
        cache=cache_decoded,
        cache_resize=resize_size if cache_decoded else None,
    )
    loader = EpisodeLoader(dataset, modality=modality, transform=eval_transform)

    if episodes_jsonl is not None:
        ep_path = Path(episodes_jsonl)
        episodes = list(_iter_episodes_from_jsonl(ep_path))
        if n_episodes is not None:
            episodes = episodes[:n_episodes]
        n_way = episodes[0]['n_way'] if episodes else int(cfg['episode'].get('n_way_test', cfg['episode']['n_way']))
    else:
        episodes, n_way = _iter_episodes_from_sampler(
            cfg, n_episodes if n_episodes is not None else int(cfg['episode'].get('episodes_meta_test', 1000))
        )

    accs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    canonical_classes: list[str] = []

    with torch.no_grad():
        for ep in episodes:
            tensors = assemble_episode_tensors(loader, ep, device)
            out = model(
                tensors['support'],
                tensors['query'],
                tensors['support_labels'],
                tensors['query_labels'],
                n_way=int(ep.get('n_way', n_way)),
            )
            accs.append(out['accuracy'])
            # Map local (episode) class indices back to canonical class labels
            # for a global confusion matrix.
            classes_sorted: list[str] = list(ep['classes'])
            for cls in classes_sorted:
                if cls not in canonical_classes:
                    canonical_classes.append(cls)
            local_preds = out['preds'].detach().cpu().tolist()
            local_labels = tensors['query_labels'].detach().cpu().tolist()
            for p, l in zip(local_preds, local_labels, strict=True):
                pred_cls = classes_sorted[int(p)]
                true_cls = classes_sorted[int(l)]
                all_preds.append(canonical_classes.index(pred_cls))
                all_labels.append(canonical_classes.index(true_cls))

    mean, ci = accuracy_with_ci(accs)
    confusion = per_class_confusion(all_preds, all_labels, canonical_classes)

    run_id = cfg.get('run_id') or ckpt_path.parent.name
    results_dir = Path(output_root) / 'results' / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'metrics.json'
    confusion_csv = results_dir / 'confusion.csv'
    confusion_png = results_dir / 'confusion.png'

    payload = {
        'run_id': run_id,
        'checkpoint': str(ckpt_path),
        'evaluated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'n_episodes': len(accs),
        'n_way': int(n_way),
        'k_shot': int(cfg['episode']['k_shot']),
        'q_query': int(cfg['episode']['q_query']),
        'mean_accuracy': mean,
        'ci95_half_width': ci,
        'per_episode_accuracies': accs,
        'classes': canonical_classes,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    confusion.to_csv(confusion_csv)
    if confusion.shape[0] >= 2:
        try:
            plot_confusion(confusion, confusion_png, normalize=True)
        except Exception as exc:  # noqa: BLE001 — plotting failure shouldn't kill the eval
            print(f'warning: failed to render confusion PNG: {exc}')

    print(f'meta_test  acc = {mean:.4f}  ±  {ci:.4f}  (95% CI, n={len(accs)} episodes)')
    print(f'metrics → {metrics_path}')
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate a trained FSAR checkpoint on meta_test.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--episodes',
        type=Path,
        default=None,
        help='Optional JSONL of pre-sampled episodes (e.g. data/episodes/meta_test/episodes.jsonl). '
        'If omitted, episodes are re-sampled deterministically from the saved config.',
    )
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--n-episodes', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_eval(
        checkpoint_path=args.checkpoint,
        episodes_jsonl=args.episodes,
        output_root=args.output_root,
        n_episodes=args.n_episodes,
        device_arg=args.device,
    )
    return 0


__all__ = ['main', 'run_eval']


if __name__ == '__main__':
    raise SystemExit(main())
