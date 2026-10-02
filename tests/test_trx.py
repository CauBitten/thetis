'''Tests for src/models/trx.py — TRX head (Perrett et al. 2021).

The reference check ports ``TemporalCrossTransformer.forward`` from the
authors' repository (github.com/tobyperrett/trx) line by line — per-query
softmax loop, ``index_select`` tuples, ``torch.norm`` distance — and compares it
with the vectorised head on shared weights.
'''
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.models.factory import build_model
from src.models.protonet import ProtoNet
from src.models.trx import TRX, TemporalCrossTransformer
from tests.test_training import _make_synthetic_manifest, _SyntheticDataset


class _FrameMeanEncoder(nn.Module):
    '''Per-frame encoder ``(B, T, H, W, 3)`` → ``(B, T, dim)``: a linear map of each frame's mean colour.'''

    per_frame = True

    def __init__(self, dim: int = 6, trainable: bool = True) -> None:
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Linear(3, dim)
        self.proj.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.to(torch.float32).mean(dim=(2, 3)) / 255.0)


def _reference_branch_logits(
    branch: TemporalCrossTransformer,
    support_set: torch.Tensor,
    support_labels: torch.Tensor,
    queries: torch.Tensor,
    way: int,
) -> torch.Tensor:
    '''Verbatim port of the original ``TemporalCrossTransformer.forward`` (dropout off).'''
    n_queries = queries.shape[0]
    n_support = support_set.shape[0]
    support_set = branch.pe(support_set)
    queries = branch.pe(queries)
    tuples = [torch.tensor(comb) for comb in itertools.combinations(range(support_set.shape[1]), branch.set_size)]
    tuples_len = len(tuples)
    s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in tuples]
    q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in tuples]
    support_set = torch.stack(s, dim=-2)
    queries = torch.stack(q, dim=-2)
    mh_support_set_ks = branch.norm_k(branch.k_linear(support_set))
    mh_queries_ks = branch.norm_k(branch.k_linear(queries))
    mh_support_set_vs = branch.v_linear(support_set)
    mh_queries_vs = branch.v_linear(queries)
    d_model = mh_queries_ks.shape[-1]
    all_distances_tensor = torch.zeros(n_queries, way)
    for c in torch.unique(support_labels):
        idx = torch.reshape(torch.nonzero(torch.eq(support_labels, c)), (-1,))
        class_k = torch.index_select(mh_support_set_ks, 0, idx)
        class_v = torch.index_select(mh_support_set_vs, 0, idx)
        class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(d_model)
        class_scores = class_scores.permute(0, 2, 1, 3)
        class_scores = class_scores.reshape(n_queries, tuples_len, -1)
        class_scores = [torch.softmax(class_scores[i], dim=1) for i in range(n_queries)]
        class_scores = torch.cat(class_scores)
        class_scores = class_scores.reshape(n_queries, tuples_len, -1, tuples_len)
        class_scores = class_scores.permute(0, 2, 1, 3)
        query_prototype = torch.matmul(class_scores, class_v)
        query_prototype = torch.sum(query_prototype, dim=1)
        diff = mh_queries_vs - query_prototype
        norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
        distance = torch.div(norm_sq, tuples_len)
        all_distances_tensor[:, c.long()] = distance * -1
    return all_distances_tensor


def test_trx_matches_reference_implementation() -> None:
    torch.manual_seed(0)
    model = TRX(_FrameMeanEncoder(dim=6), temporal_set_sizes=(2, 3), d_model=8).eval()
    n_way, k_shot, n_query, t = 3, 2, 4, 5
    support = torch.randn(n_way * k_shot, t, 6)
    queries = torch.randn(n_query, t, 6)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)

    with torch.no_grad():
        ours = model.query_logits(queries, model.support_state(support, support_labels, n_way))
        reference = torch.stack(
            [_reference_branch_logits(b, support, support_labels, queries, n_way) for b in model.transformers]
        ).mean(dim=0)
    assert ours.shape == (n_query, n_way)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_trx_tuples_cover_all_ordered_combinations() -> None:
    branch = TemporalCrossTransformer(in_dim=4, d_model=8, set_size=3, dropout=0.0, pe_scale=0.0)
    x = torch.arange(6, dtype=torch.float32).view(1, 6, 1).expand(2, 6, 4)  # frame i filled with i
    tuples = branch._tuples(x)
    assert tuples.shape == (2, math.comb(6, 3), 3 * 4)
    frames = tuples[0, :, ::4].tolist()  # first feature of each frame in the tuple
    assert frames == [list(map(float, c)) for c in itertools.combinations(range(6), 3)]
    with pytest.raises(ValueError, match='need T >= 3'):
        branch._tuples(torch.zeros(1, 2, 4))


def _ramp_episode(k_shot: int = 2, n_query: int = 3, t: int = 6) -> dict[str, torch.Tensor]:
    '''Class 0 brightens over time, class 1 darkens: same frames, reversed order.'''
    ramp = np.linspace(40, 220, t)
    up = np.broadcast_to(ramp[:, None, None, None], (t, 4, 4, 3))
    down = up[::-1]
    support = np.stack([up] * k_shot + [down] * k_shot).astype(np.uint8)
    query = np.stack([up] * n_query + [down] * n_query).astype(np.uint8)
    return {
        'support': torch.from_numpy(support),
        'query': torch.from_numpy(query),
        'support_labels': torch.tensor([0] * k_shot + [1] * k_shot),
        'query_labels': torch.tensor([0] * n_query + [1] * n_query),
    }


def test_trx_separates_temporal_order_that_mean_pooling_cannot() -> None:
    ep = _ramp_episode()
    args = (ep['support'], ep['query'], ep['support_labels'], ep['query_labels'])

    # ProtoNet averages frames over time: both classes collapse to the same prototype.
    torch.manual_seed(0)
    with torch.no_grad():
        proto_out = ProtoNet(_FrameMeanEncoder(trainable=False))(*args, n_way=2)
    assert torch.allclose(proto_out['logits'][:, 0], proto_out['logits'][:, 1])

    torch.manual_seed(0)
    model = TRX(_FrameMeanEncoder(trainable=False), d_model=16, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        model(*args, n_way=2)['loss'].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        assert model(*args, n_way=2)['accuracy'] == pytest.approx(1.0)


def test_trx_forward_backward_reaches_encoder_and_head() -> None:
    torch.manual_seed(0)
    model = TRX(_FrameMeanEncoder(), d_model=8)
    ep = _ramp_episode()
    out = model(ep['support'], ep['query'], ep['support_labels'], ep['query_labels'], n_way=2)
    assert out['logits'].shape == (6, 2)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.encoder.proj.weight.grad is not None
    for branch in model.transformers:
        assert branch.k_linear.weight.grad is not None and branch.v_linear.weight.grad is not None


def test_trx_rejects_clip_level_features() -> None:
    model = TRX(_FrameMeanEncoder(), d_model=8)
    with pytest.raises(ValueError, match='per-frame'):
        model.support_state(torch.zeros(4, 6), torch.tensor([0, 0, 1, 1]), n_way=2)

    class _ClipEncoder(nn.Module):
        embed_dim = 6

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 6)

    with pytest.raises(ValueError, match='per-frame'):
        build_model({'method': 'trx'}, _ClipEncoder())


def test_trx_class_with_no_support_raises() -> None:
    model = TRX(_FrameMeanEncoder(), d_model=8)
    with pytest.raises(ValueError, match='no support samples'):
        model.support_state(torch.zeros(4, 5, 6), torch.tensor([0, 0, 1, 1]), n_way=3)


def test_build_model_trx_reads_model_section() -> None:
    cfg = {'method': 'trx', 'model': {'temporal_set_sizes': [2], 'd_model': 16, 'dropout': 0.0}}
    model = build_model(cfg, _FrameMeanEncoder(dim=6), encoder_batch_size=4)
    assert isinstance(model, TRX)
    assert [b.set_size for b in model.transformers] == [2]
    assert model.transformers[0].k_linear.weight.shape == (16, 2 * 6)


def test_trx_streaming_step_matches_classic_step() -> None:
    from src.training.meta_trainer import _train_step_streaming  # noqa: PLC0415

    def _build() -> TRX:
        torch.manual_seed(3)
        return TRX(_FrameMeanEncoder(), encoder_batch_size=2, d_model=8, dropout=0.0)

    ep = _ramp_episode(k_shot=2, n_query=3)
    model_stream = _build()
    opt_stream = torch.optim.SGD(model_stream.parameters(), lr=0.1)
    loss_s, acc_s = _train_step_streaming(
        model_stream, ep, n_way=2, optimizer=opt_stream, scaler=None, use_amp=False, device=torch.device('cpu')
    )

    model_classic = _build()
    opt_classic = torch.optim.SGD(model_classic.parameters(), lr=0.1)
    out = model_classic(ep['support'], ep['query'], ep['support_labels'], ep['query_labels'], n_way=2)
    out['loss'].backward()
    opt_classic.step()

    assert loss_s == pytest.approx(float(out['loss'].detach()), rel=1e-5, abs=1e-6)
    assert acc_s == pytest.approx(out['accuracy'], abs=1e-6)
    for (name, p_s), (_, p_c) in zip(model_stream.named_parameters(), model_classic.named_parameters()):
        assert torch.allclose(p_s, p_c, atol=1e-6), f'param {name} diverged'


def test_trx_train_then_eval_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    '''method: trx through run_training (smoke) and run_eval, with a toy per-frame encoder.'''
    from src.training.eval_episodic import run_eval  # noqa: PLC0415
    from src.training.meta_trainer import run_training  # noqa: PLC0415

    manifest_path = _make_synthetic_manifest(tmp_path)
    df_full = pd.read_csv(manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False)

    def _fake_dataset(*args: Any, **kwargs: Any) -> Any:
        return _SyntheticDataset(df_full, modality='rgb')

    monkeypatch.setattr('src.training.meta_trainer.ThetisDataset', _fake_dataset)
    monkeypatch.setattr('src.training.eval_episodic.ThetisDataset', _fake_dataset)
    monkeypatch.setattr('src.models.factory.FrameEncoder', lambda **_kw: _FrameMeanEncoder(dim=4))

    cfg = {
        'method': 'trx',
        'modalities': ['rgb'],
        'encoder': {'name': 'resnet18', 'pretrained': False, 'batch_size': 2},
        'model': {'temporal_set_sizes': [2, 3], 'd_model': 8},
        'episode': {'n_way': 2, 'n_way_val': 2, 'n_way_test': 2, 'k_shot': 2, 'q_query': 2,
                    'episodes_per_epoch': 2, 'episodes_meta_val': 2},
        'optim': {'epochs': 1, 'learning_rate': 1e-3, 'eval_every': 1},
        'data': {'manifest_path': str(manifest_path), 'dataset_root': str(tmp_path),
                 'train_classes': 2, 'val_classes': 2, 'test_classes': 2,
                 'frame_count': 4, 'resize_size': 24, 'spatial_size': 16,
                 'temporal_sampling': 'segment', 'temporal_oversample': 4},
        'seed': 42,
        'output_root': str(tmp_path / 'outputs'),
        'log_root': str(tmp_path / 'logs'),
        'run_id': 'unittest_trx',
    }
    log = run_training(cfg, smoke=True, device_arg='cpu')
    assert np.isfinite(log['epochs'][0]['train_loss'])

    ckpt = tmp_path / 'outputs' / 'checkpoints' / 'smoke_unittest_trx' / 'best.pt'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert blob['config']['method'] == 'trx'
    assert any(k.startswith('transformers.1.k_linear') for k in blob['model_state'])

    payload = run_eval(ckpt, None, tmp_path / 'outputs', n_episodes=3, device_arg='cpu')
    assert payload['n_episodes'] == 3
    assert 0.0 <= payload['mean_accuracy'] <= 1.0
