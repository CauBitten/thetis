'''Tests for the supervised baseline (src/models/supervised.py + meta_trainer's supervised branch).'''
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.episode_sampler import split_classes
from src.models.supervised import SupervisedBaseline
from tests.test_training import _make_synthetic_manifest, _SyntheticDataset, _ToyEncoder


def test_probe_fits_support_under_no_grad() -> None:
    '''Eval runs under no_grad: the per-episode probe must still train and separate easy classes.'''
    torch.manual_seed(0)
    model = SupervisedBaseline(_ToyEncoder(dim=4), train_classes=['a', 'b'])
    centers = torch.eye(4)[:3] * 3
    support_labels = torch.arange(3).repeat_interleave(5)
    query_labels = torch.arange(3).repeat_interleave(4)
    with torch.no_grad():
        head = model.support_state(centers[support_labels] + 0.1 * torch.randn(15, 4), support_labels, n_way=3)
        preds = model.query_logits(centers[query_labels] + 0.1 * torch.randn(12, 4), head).argmax(dim=-1)
    assert torch.equal(preds, query_labels)


def test_supervised_train_then_eval_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.eval_episodic import run_eval  # noqa: PLC0415
    from src.training.meta_trainer import run_training  # noqa: PLC0415

    manifest_path = _make_synthetic_manifest(tmp_path)
    df_full = pd.read_csv(manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False)
    fake = lambda *a, **kw: _SyntheticDataset(df_full, modality='rgb')  # noqa: E731
    monkeypatch.setattr('src.training.meta_trainer.ThetisDataset', fake)
    monkeypatch.setattr('src.training.eval_episodic.ThetisDataset', fake)
    monkeypatch.setattr('src.models.factory.VideoEncoder', lambda **_kw: _ToyEncoder())
    meta_train = split_classes(sorted(df_full['action_label'].unique()), 2, 2, 2, seed=42)['meta_train']
    cfg = {
        'method': 'supervised',
        'modalities': ['rgb'],
        'encoder': {'name': 'r2plus1d_18', 'pretrained': False, 'batch_size': 4},
        'model': {'train_classes': meta_train},
        'episode': {'n_way': 2, 'k_shot': 2, 'q_query': 2, 'episodes_meta_val': 2},
        'optim': {'epochs': 1, 'learning_rate': 1e-3, 'eval_every': 1},
        'data': {'manifest_path': str(manifest_path), 'dataset_root': str(tmp_path),
                 'train_classes': 2, 'val_classes': 2, 'test_classes': 2,
                 'frame_count': 4, 'resize_size': 24, 'spatial_size': 16},
        'seed': 42,
        'output_root': str(tmp_path / 'outputs'),
        'log_root': str(tmp_path / 'logs'),
        'run_id': 'unittest_supervised',
    }
    log = run_training(cfg, smoke=True, device_arg='cpu')
    assert np.isfinite(log['epochs'][0]['train_loss']) and 'val_acc' in log['epochs'][0]

    ckpt = tmp_path / 'outputs' / 'checkpoints' / 'smoke_unittest_supervised' / 'best.pt'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert blob['model_state']['classifier.weight'].shape == (2, 4)
    # one model, tested at both K
    for k in (1, 3):
        metrics = run_eval(ckpt, None, tmp_path / 'outputs', n_episodes=3, device_arg='cpu', k_shot=k)
        assert metrics['k_shot'] == k and metrics['n_episodes'] == 3
        assert (tmp_path / 'outputs' / 'results' / 'unittest_supervised' / f'metrics_{k}shot.json').exists()

    # supervised encoder + prototypes: only the encoder weights are loaded
    metrics = run_eval(ckpt, None, tmp_path / 'outputs', n_episodes=3, device_arg='cpu', k_shot=1, head='protonet')
    assert metrics['method'] == 'protonet' and metrics['n_episodes'] == 3
    assert (tmp_path / 'outputs' / 'results' / 'unittest_supervised' / 'metrics_1shot_protonet.json').exists()
