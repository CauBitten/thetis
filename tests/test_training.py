'''Smoke tests for src/training/meta_trainer.py and src/utils/metrics.py.

The meta_trainer smoke test stubs the heavy bits (encoder + video loading) so
it runs in seconds without touching the real dataset/ tree.
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from src.data.augment import RandomSegmentSample, RandomTemporalCrop
from src.training.meta_trainer import (
    EpisodeLoader,
    assemble_episode_tensors,
    build_eval_transform,
    build_train_transform,
    load_config,
    run_training,
    temporal_oversample,
)
from src.utils.metrics import accuracy_with_ci, per_class_confusion


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_accuracy_with_ci_zero_variance() -> None:
    mean, ci = accuracy_with_ci([0.5, 0.5, 0.5, 0.5])
    assert mean == pytest.approx(0.5)
    assert ci == pytest.approx(0.0, abs=1e-9)


def test_accuracy_with_ci_handles_single_sample() -> None:
    mean, ci = accuracy_with_ci([0.7])
    assert mean == pytest.approx(0.7)
    assert ci == 0.0


def test_accuracy_with_ci_positive_half_width() -> None:
    accs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.4, 0.6]
    mean, ci = accuracy_with_ci(accs, confidence=0.95)
    assert 0.0 < ci < 1.0
    assert mean == pytest.approx(float(np.mean(accs)))


def test_per_class_confusion_counts() -> None:
    preds = [0, 1, 0, 2, 2]
    labels = [0, 1, 1, 2, 0]
    cm = per_class_confusion(preds, labels, class_names=['a', 'b', 'c'])
    assert cm.loc['a', 'a'] == 1
    assert cm.loc['b', 'b'] == 1
    assert cm.loc['b', 'a'] == 1
    assert cm.loc['c', 'c'] == 1
    assert cm.loc['a', 'c'] == 1
    assert int(cm.to_numpy().sum()) == len(preds)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_load_config_valid(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(
        'method: protonet\n'
        'modalities: [rgb]\n'
        'episode: {n_way: 5, k_shot: 5, q_query: 15}\n'
        'optim: {epochs: 1}\n'
        'data: {manifest_path: data/processed/manifest.csv, dataset_root: dataset}\n'
        'seed: 42\n'
    )
    cfg = load_config(cfg_path)
    assert cfg['method'] == 'protonet'
    assert cfg['modalities'] == ['rgb']


def test_load_config_rejects_unknown_method(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(
        'method: not_a_method\n'
        'modalities: [rgb]\n'
        'episode: {n_way: 5, k_shot: 5, q_query: 15}\n'
        'optim: {epochs: 1}\n'
        'data: {manifest_path: x, dataset_root: y}\n'
        'seed: 1\n'
    )
    with pytest.raises(NotImplementedError):
        load_config(cfg_path)


def test_load_config_rejects_unknown_modality(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(
        'method: protonet\n'
        'modalities: [nope]\n'
        'episode: {n_way: 5, k_shot: 5, q_query: 15}\n'
        'optim: {epochs: 1}\n'
        'data: {manifest_path: x, dataset_root: y}\n'
        'seed: 1\n'
    )
    with pytest.raises(ValueError, match='unknown modality'):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def test_build_train_transform_runs() -> None:
    transform = build_train_transform({'frame_count': 8, 'resize_size': 32, 'spatial_size': 24}, seed=0)
    # Synthetic sample shaped like ThetisDataset output (16 frames at 64x64).
    sample = {'rgb': np.zeros((16, 64, 64, 3), dtype=np.uint8)}
    out = transform(sample)
    rgb = out['rgb']
    arr = rgb.numpy() if isinstance(rgb, torch.Tensor) else rgb
    assert arr.shape == (8, 24, 24, 3)


def _frame_indexed_video(t: int) -> np.ndarray:
    '''``(t, 4, 4, 3)`` clip whose frame ``i`` is filled with ``i`` — output pixels reveal the indices.'''
    return np.broadcast_to(np.arange(t, dtype=np.uint8)[:, None, None, None], (t, 4, 4, 3)).copy()


def test_random_segment_sample_draws_one_frame_per_segment() -> None:
    sampler = RandomSegmentSample(num_frames=8, seed=0)
    for _ in range(20):
        out = sampler({'rgb': _frame_indexed_video(32), 'depth': _frame_indexed_video(32)})
        idx = out['rgb'][:, 0, 0, 0].astype(int)
        assert idx.shape == (8,)
        # 32 frames / 8 segments: segment i is [4i, 4i+4).
        assert all(4 * i <= v < 4 * i + 4 for i, v in enumerate(idx))
        np.testing.assert_array_equal(out['depth'], out['rgb'])


def test_random_segment_sample_short_clip_repeats_on_uniform_grid() -> None:
    out = RandomSegmentSample(num_frames=8, seed=0)({'rgb': _frame_indexed_video(4)})
    idx = out['rgb'][:, 0, 0, 0].astype(int).tolist()
    assert idx == [0, 0, 1, 1, 2, 2, 3, 3]


def test_build_train_transform_temporal_sampling_option() -> None:
    sizes = {'frame_count': 8, 'resize_size': 32, 'spatial_size': 24}
    assert isinstance(build_train_transform(sizes, seed=0).transforms[0], RandomTemporalCrop)
    segment = build_train_transform({**sizes, 'temporal_sampling': 'segment'}, seed=0)
    assert isinstance(segment.transforms[0], RandomSegmentSample)
    out = segment({'rgb': np.zeros((32, 64, 64, 3), dtype=np.uint8)})
    rgb = out['rgb']
    arr = rgb.numpy() if isinstance(rgb, torch.Tensor) else rgb
    assert arr.shape == (8, 24, 24, 3)
    with pytest.raises(ValueError, match='temporal_sampling'):
        build_train_transform({**sizes, 'temporal_sampling': 'tsn'}, seed=0)


def test_temporal_oversample_default_and_validation() -> None:
    assert temporal_oversample({}) == 2
    assert temporal_oversample({'temporal_oversample': 4}) == 4
    with pytest.raises(ValueError, match='temporal_oversample'):
        temporal_oversample({'temporal_oversample': 0})


def test_build_eval_transform_is_deterministic() -> None:
    transform = build_eval_transform({'resize_size': 32, 'spatial_size': 24})
    sample = {'rgb': np.full((4, 64, 64, 3), 42, dtype=np.uint8)}
    out_a = transform(sample.copy())
    out_b = transform(sample.copy())
    arr_a = out_a['rgb'].numpy() if isinstance(out_a['rgb'], torch.Tensor) else out_a['rgb']
    arr_b = out_b['rgb'].numpy() if isinstance(out_b['rgb'], torch.Tensor) else out_b['rgb']
    assert arr_a.shape == (4, 24, 24, 3)
    np.testing.assert_array_equal(arr_a, arr_b)


# ---------------------------------------------------------------------------
# End-to-end smoke for run_training with stubbed encoder + dataset
# ---------------------------------------------------------------------------


class _ToyEncoder(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.embed_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.proj(x.to(torch.float32).mean(dim=(1, 2, 3)))


class _SyntheticDataset:
    '''Stand-in for :class:`ThetisDataset` that fabricates per-class tinted videos.'''

    def __init__(self, manifest_df_subset: Any, modality: str) -> None:
        self.df = manifest_df_subset.reset_index(drop=True)
        self.modality = modality
        rng = np.random.default_rng(0)
        self._class_colors = {
            cls: rng.integers(20, 235, size=3).astype(np.uint8)
            for cls in self.df['action_label'].unique()
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        cls = str(row['action_label'])
        color = self._class_colors[cls]
        video = np.broadcast_to(color, (16, 32, 32, 3)).astype(np.uint8)
        return {
            'sample_id': str(row['sample_id']),
            'label': int(row['action_index']),
            'action_label': cls,
            'action_index': int(row['action_index']),
            'actor': str(row['actor']),
            'actor_index': int(row['actor_index']),
            'skill_level': str(row['skill_level']),
            'sequence_idx': int(row['sequence_idx']),
            'rgb': torch.from_numpy(video.copy()),
        }


def _make_synthetic_manifest(tmp_path: Path) -> Path:
    '''Hand-roll a tiny manifest with 6 classes × 6 samples each (3 beg / 3 exp).

    Splits at 6/3/3 with seed=42 will assign 3 of these 6 classes to meta_train
    and the rest to val/test; n_way=3 in train and 3 in val keeps it viable.
    '''
    import pandas as pd  # noqa: PLC0415

    from src.data.loader import ACTION_INDEX, ACTION_LABEL_TO_CODE  # noqa: PLC0415

    rows = []
    classes = [
        'backhand',
        'backhand2hands',
        'forehand_flat',
        'forehand_volley',
        'smash',
        'kick_service',
    ]
    for cls in classes:
        for actor_idx in (1, 10, 20, 33, 44, 55):
            rows.append({
                'sample_id': f'p{actor_idx}_{cls}_s1',
                'actor': f'p{actor_idx}',
                'actor_index': actor_idx,
                'skill_level': 'beginner' if actor_idx <= 31 else 'expert',
                'action_code': ACTION_LABEL_TO_CODE[cls],
                'action_label': cls,
                'action_index': ACTION_INDEX[cls],
                'sequence_idx': 1,
                'path_rgb': f'VIDEO_RGB/{cls}/p{actor_idx}_{cls}_s1.avi',
                'path_depth': '',
                'path_mask': '',
                'path_skeleton_2d': '',
                'path_skeleton_3d': '',
                'n_modalities': 1,
            })
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_run_training_smoke_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = _make_synthetic_manifest(tmp_path)
    dataset_root = tmp_path  # paths inside the synthetic manifest aren't read by _SyntheticDataset.

    cfg = {
        'method': 'protonet',
        'modalities': ['rgb'],
        'encoder': {'name': 'r2plus1d_18', 'pretrained': False},
        'episode': {
            'n_way': 2,
            'n_way_val': 2,
            'k_shot': 2,
            'q_query': 2,
            'episodes_per_epoch': 2,
            'episodes_meta_val': 2,
        },
        'optim': {'epochs': 1, 'learning_rate': 1e-3, 'eval_every': 1},
        'data': {
            'manifest_path': str(manifest_path),
            'dataset_root': str(dataset_root),
            'train_classes': 2,
            'val_classes': 2,
            'test_classes': 2,
            'frame_count': 4,
            'resize_size': 24,
            'spatial_size': 16,
        },
        'seed': 42,
        'output_root': str(tmp_path / 'outputs'),
        'log_root': str(tmp_path / 'logs'),
        'run_id': 'unittest_run',
    }

    # Patch the heavy bits: dataset → synthetic, encoder → toy.
    import pandas as pd  # noqa: PLC0415

    df_full = pd.read_csv(manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False)

    def _fake_dataset(*args: Any, **kwargs: Any) -> Any:
        return _SyntheticDataset(df_full, modality='rgb')

    monkeypatch.setattr('src.training.meta_trainer.ThetisDataset', _fake_dataset)
    monkeypatch.setattr('src.models.factory.VideoEncoder', lambda **_kw: _ToyEncoder())

    log = run_training(cfg, smoke=True, device_arg='cpu')

    assert log['epochs'], 'training log should have at least one epoch entry'
    epoch_entry = log['epochs'][0]
    assert 'train_loss' in epoch_entry
    assert np.isfinite(epoch_entry['train_loss'])
    assert 'val_acc' in epoch_entry  # smoke forces eval every epoch

    # Best checkpoint should be saved + loadable
    ckpt = tmp_path / 'outputs' / 'checkpoints' / 'smoke_unittest_run' / 'best.pt'
    assert ckpt.exists(), f'expected checkpoint at {ckpt}'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert 'model_state' in blob
    assert blob['config']['method'] == 'protonet'

    # Training log written to disk
    log_path = tmp_path / 'logs' / 'smoke_unittest_run' / 'training.json'
    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert payload['epochs'][0]['train_loss'] == epoch_entry['train_loss']


# ---------------------------------------------------------------------------
# EpisodeLoader unit test (no torchvision required)
# ---------------------------------------------------------------------------


def test_episode_loader_returns_stacked_tensor() -> None:
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame({
        'sample_id': ['a', 'b', 'c'],
        'action_label': ['x', 'x', 'y'],
        'action_index': [0, 0, 1],
        'actor': ['p1', 'p2', 'p3'],
        'actor_index': [1, 2, 3],
        'skill_level': ['beginner', 'beginner', 'beginner'],
        'sequence_idx': [1, 1, 1],
    })

    class _Stub:
        def __init__(self) -> None:
            self.df = df

        def __len__(self) -> int:
            return len(df)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = df.iloc[idx]
            return {'rgb': torch.full((4, 4, 4, 3), int(row['action_index']) + 1, dtype=torch.uint8)}

    loader = EpisodeLoader(_Stub(), modality='rgb', transform=None)
    out = loader.load(['a', 'c', 'b'])
    assert out.shape == (3, 4, 4, 4, 3)
    assert int(out[0, 0, 0, 0, 0]) == 1
    assert int(out[1, 0, 0, 0, 0]) == 2
    assert int(out[2, 0, 0, 0, 0]) == 1


# ---------------------------------------------------------------------------
# Streaming grad-accumulation ≡ classic full-batch step
# ---------------------------------------------------------------------------


def _mini_episode() -> dict[str, torch.Tensor]:
    '''Tiny 2-way / 2-shot / 4-query episode of per-class solid-colour clips.'''
    rng = np.random.default_rng(3)
    colors = rng.integers(30, 220, size=(2, 3)).astype(np.uint8)
    support = np.stack(
        [np.broadcast_to(colors[c], (4, 6, 6, 3)) for c in range(2) for _ in range(2)]
    ).astype(np.uint8)
    query = np.stack(
        [np.broadcast_to(colors[c], (4, 6, 6, 3)) for c in range(2) for _ in range(4)]
    ).astype(np.uint8)
    return {
        'support': torch.from_numpy(support),
        'query': torch.from_numpy(query),
        'support_labels': torch.tensor([0, 0, 1, 1], dtype=torch.long),
        'query_labels': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
    }


def _build_protonet(per_frame: bool = False) -> Any:
    from src.models.protonet import ProtoNet  # noqa: PLC0415

    torch.manual_seed(7)

    class _Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # per-frame: (B, T, 5), averaged over T by ProtoNet; else (B, 5)
            pooled = x.to(torch.float32).mean(dim=(2, 3) if per_frame else (1, 2, 3))
            return self.lin(pooled)

    return ProtoNet(_Enc(), encoder_batch_size=3)


@pytest.mark.parametrize('per_frame', [False, True])
def test_streaming_step_matches_classic_step(per_frame: bool) -> None:
    '''Streaming grad-accum must match a full-batch backward step: same loss,
    same accuracy, same post-step parameters (CPU / fp32). Holds for BatchNorm-free
    encoders only — with BatchNorm the two paths normalise different batches.'''
    from src.training.meta_trainer import _train_step_streaming  # noqa: PLC0415

    tensors = _mini_episode()
    device = torch.device('cpu')

    model_stream = _build_protonet(per_frame)
    opt_stream = torch.optim.SGD(model_stream.parameters(), lr=0.1)
    loss_s, acc_s = _train_step_streaming(
        model_stream, tensors, n_way=2, optimizer=opt_stream,
        scaler=None, use_amp=False, device=device,
    )

    model_classic = _build_protonet(per_frame)
    opt_classic = torch.optim.SGD(model_classic.parameters(), lr=0.1)
    opt_classic.zero_grad(set_to_none=True)
    out = model_classic(
        tensors['support'], tensors['query'],
        tensors['support_labels'], tensors['query_labels'], n_way=2,
    )
    out['loss'].backward()
    opt_classic.step()

    assert loss_s == pytest.approx(float(out['loss'].detach()), rel=1e-5, abs=1e-6)
    assert acc_s == pytest.approx(out['accuracy'], abs=1e-6)
    for (name, p_s), (_, p_c) in zip(
        model_stream.named_parameters(), model_classic.named_parameters()
    ):
        assert torch.allclose(p_s, p_c, atol=1e-6), f'param {name} diverged'


# ---------------------------------------------------------------------------
# --resume: continue an interrupted run from last.pt
# ---------------------------------------------------------------------------


def _resume_cfg(tmp_path: Path, manifest_path: Path, epochs: int) -> dict[str, Any]:
    return {
        'method': 'protonet',
        'modalities': ['rgb'],
        'encoder': {'name': 'r2plus1d_18', 'pretrained': False, 'batch_size': 2},
        'episode': {
            'n_way': 2,
            'n_way_val': 2,
            'k_shot': 1,
            'q_query': 2,
            'episodes_per_epoch': 1,
            'episodes_meta_val': 1,
        },
        'optim': {
            'epochs': epochs,
            'learning_rate': 1e-3,
            'eval_every': 1,
            'fp16': False,
            'stream_query': False,
        },
        'data': {
            'manifest_path': str(manifest_path),
            'dataset_root': str(tmp_path),
            'train_classes': 2,
            'val_classes': 2,
            'test_classes': 2,
            'frame_count': 4,
            'resize_size': 24,
            'spatial_size': 16,
        },
        'seed': 42,
        'output_root': str(tmp_path / 'outputs'),
        'log_root': str(tmp_path / 'logs'),
        'run_id': 'unittest_resume',
    }


def test_run_training_resume_continues_from_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    '''A run stopped at epoch N picks back up at N+1 and keeps its history.'''
    import pandas as pd  # noqa: PLC0415

    manifest_path = _make_synthetic_manifest(tmp_path)
    df_full = pd.read_csv(
        manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False
    )
    monkeypatch.setattr(
        'src.training.meta_trainer.ThetisDataset',
        lambda *a, **kw: _SyntheticDataset(df_full, modality='rgb'),
    )
    monkeypatch.setattr('src.models.factory.VideoEncoder', lambda **_kw: _ToyEncoder())

    ckpt_dir = tmp_path / 'outputs' / 'checkpoints' / 'unittest_resume'

    # --resume with nothing to resume from: starts at epoch 1 anyway.
    first = run_training(_resume_cfg(tmp_path, manifest_path, epochs=2), False, 'cpu', resume='auto')
    assert [e['epoch'] for e in first['epochs']] == [1, 2]
    assert 'resumed_from' not in first
    # last.pt + the JSON log are written every epoch, not only at the end.
    assert (ckpt_dir / 'last.pt').exists()
    assert torch.load(ckpt_dir / 'last.pt', map_location='cpu', weights_only=False)['epoch'] == 2

    # Raising the epoch target and resuming continues at 3 with history intact.
    second = run_training(_resume_cfg(tmp_path, manifest_path, epochs=4), False, 'cpu', resume='auto')
    assert [e['epoch'] for e in second['epochs']] == [1, 2, 3, 4]
    assert second['resumed_from']['at_epoch'] == 3
    assert second['started_at'] == first['started_at']
    logged = json.loads((tmp_path / 'logs' / 'unittest_resume' / 'training.json').read_text())
    assert [e['epoch'] for e in logged['epochs']] == [1, 2, 3, 4]

    # Target already met: no further epochs, log preserved.
    third = run_training(_resume_cfg(tmp_path, manifest_path, epochs=4), False, 'cpu', resume='auto')
    assert [e['epoch'] for e in third['epochs']] == [1, 2, 3, 4]


def test_run_training_resume_missing_explicit_checkpoint_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    '''An explicit --resume path that does not exist is an error, not a silent restart.'''
    import pandas as pd  # noqa: PLC0415

    manifest_path = _make_synthetic_manifest(tmp_path)
    df_full = pd.read_csv(
        manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False
    )
    monkeypatch.setattr(
        'src.training.meta_trainer.ThetisDataset',
        lambda *a, **kw: _SyntheticDataset(df_full, modality='rgb'),
    )
    monkeypatch.setattr('src.models.factory.VideoEncoder', lambda **_kw: _ToyEncoder())

    with pytest.raises(FileNotFoundError):
        run_training(
            _resume_cfg(tmp_path, manifest_path, epochs=1),
            False,
            'cpu',
            resume=str(tmp_path / 'nope.pt'),
        )
