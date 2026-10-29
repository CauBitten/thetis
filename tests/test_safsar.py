'''Tests for src/models/safsar.py (SAFSAR) and the VideoMAE encoder it runs on.

VideoMAE is built with random weights (no download); the real-BERT check is
skipped when the tokenizer is not cached.
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data.episode_sampler import split_classes
from src.models.encoders import IMAGENET_MEAN, IMAGENET_STD, VideoMAEEncoder, preprocess_video_batch
from src.models.factory import build_encoder, build_model
from src.models.safsar import CLASS_DESCRIPTIONS, SAFSAR
from tests.test_training import _make_synthetic_manifest, _SyntheticDataset


class _ToyVideoMAE(nn.Module):
    '''Clip-level mean colour → Linear, plus deterministic token features with uneven padding.'''

    per_frame = False

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Linear(3, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.to(torch.float32).mean(dim=(1, 2, 3)) / 255.0)

    def encode_text_tokens(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.randn(len(sentences), 5, self.embed_dim, generator=torch.Generator().manual_seed(0))
        padding = torch.zeros(len(sentences), 5, dtype=torch.bool)
        padding[::2, 3:] = True  # every other description is shorter
        return tokens, padding


def _toy_model(**kw: Any) -> SAFSAR:
    torch.manual_seed(0)
    kw.setdefault('train_classes', ['flat_service', 'smash', 'backhand'])
    return SAFSAR(_ToyVideoMAE(), heads=2, mlp_dim=16, **kw)


def _episode(k_shot: int = 2, n_query: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    colours = rng.integers(30, 220, size=(2, 3))
    clip = lambda c: np.broadcast_to(colours[c], (4, 4, 4, 3)).astype(np.uint8)  # noqa: E731
    return {
        'support': torch.from_numpy(np.stack([clip(c) for c in (0, 1) for _ in range(k_shot)])),
        'query': torch.from_numpy(np.stack([clip(c) for c in (0, 1) for _ in range(n_query)])),
        'support_labels': torch.tensor([0] * k_shot + [1] * k_shot),
        'query_labels': torch.tensor([0] * n_query + [1] * n_query),
        'classes': ['flat_service', 'smash'],
    }


# ---------------------------------------------------------------------------
# SAFSAR head (toy encoder)
# ---------------------------------------------------------------------------


def test_safsar_stores_description_tokens() -> None:
    model = _toy_model()
    assert tuple(model.text_tokens.shape) == (len(CLASS_DESCRIPTIONS), 5, 8)
    assert {'text_tokens', 'text_padding'} <= set(model.state_dict())  # eval needs no BERT
    assert tuple(model.global_head.weight.shape) == (3, 8)


def test_safsar_forward_backward() -> None:
    model = _toy_model()
    ep = _episode()
    out = model(ep['support'], ep['query'], ep['support_labels'], ep['query_labels'], n_way=2, class_names=ep['classes'])
    assert out['logits'].shape == (6, 2)
    assert float(out['logits'].abs().max()) <= 1.0 + 1e-6  # cosine logits
    out['loss'].backward()
    for p in (model.encoder.proj.weight, model.fusion.layers[0].linear1.weight, model.tlm.linear1.weight,
              model.global_head.weight):
        assert p.grad is not None


def test_safsar_fuses_each_prototype_with_its_own_description() -> None:
    model = _toy_model().eval()
    emb, labels = torch.randn(4, 8), torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        fused = model.support_state(emb, labels, 2, ['flat_service', 'smash'])
        other = model.support_state(emb, labels, 2, ['kick_service', 'smash'])
    assert not torch.allclose(fused[0], other[0])  # class 0's description changed
    assert torch.allclose(fused[1], other[1])      # class 1 untouched: fusion is per class


def test_safsar_requires_known_class_names_and_clip_features() -> None:
    model = _toy_model()
    emb, labels = torch.randn(4, 8), torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match='class_names'):
        model.support_state(emb, labels, 2)
    with pytest.raises(ValueError, match='no class description'):
        model.support_state(emb, labels, 2, ['flat_service', 'cartwheel'])
    with pytest.raises(ValueError, match='clip-level'):
        model.support_state(torch.randn(4, 3, 8), labels, 2, ['flat_service', 'smash'])


def test_safsar_global_loss_only_on_meta_train_classes() -> None:
    model = _toy_model()
    emb, labels = torch.randn(4, 8), torch.tensor([0, 0, 1, 1])
    assert model.auxiliary_loss(emb, labels, ['flat_service', 'smash']) is not None
    assert model.auxiliary_loss(emb, labels, ['flat_service', 'kick_service']) is None  # val/test classes
    assert _toy_model(train_classes=None).auxiliary_loss(emb, labels, ['flat_service', 'smash']) is None


def test_build_model_safsar_needs_a_videomae_encoder() -> None:
    from tests.test_mvp_shot import _ToyClipEncoder  # noqa: PLC0415

    with pytest.raises(ValueError, match='VideoMAE encoder'):
        build_model({'method': 'safsar'}, _ToyClipEncoder())
    model = build_model({'method': 'safsar', 'model': {'heads': 2, 'mlp_dim': 16, 'fusion_layers': 1}}, _ToyVideoMAE())
    assert isinstance(model, SAFSAR) and len(model.fusion.layers) == 1 and model.train_classes is None


def test_safsar_streaming_step_matches_classic_step() -> None:
    '''Includes the global loss: support term once, query term spread over micro-batches.'''
    from src.training.meta_trainer import _train_step_streaming  # noqa: PLC0415

    def _build() -> SAFSAR:
        model = _toy_model(dropout=0.0)
        model.encoder_batch_size = 2
        return model

    ep = _episode()
    model_stream = _build()
    opt_stream = torch.optim.SGD(model_stream.parameters(), lr=0.1)
    loss_s, acc_s = _train_step_streaming(
        model_stream, ep, n_way=2, optimizer=opt_stream, scaler=None, use_amp=False, device=torch.device('cpu')
    )
    model_classic = _build()
    opt_classic = torch.optim.SGD(model_classic.parameters(), lr=0.1)
    out = model_classic(ep['support'], ep['query'], ep['support_labels'], ep['query_labels'], n_way=2,
                        class_names=ep['classes'])
    out['loss'].backward()
    opt_classic.step()

    assert loss_s == pytest.approx(float(out['loss'].detach()), rel=1e-5, abs=1e-6)
    assert acc_s == pytest.approx(out['accuracy'], abs=1e-6)
    for (name, p_s), (_, p_c) in zip(model_stream.named_parameters(), model_classic.named_parameters()):
        assert torch.allclose(p_s, p_c, atol=1e-6), f'param {name} diverged'


def _cfg(tmp_path: Path, manifest_path: Path, train_classes: list[str]) -> dict[str, Any]:
    return {
        'method': 'safsar',
        'modalities': ['rgb'],
        'encoder': {'name': 'videomae_base', 'pretrained': False, 'batch_size': 2},
        'model': {'heads': 2, 'mlp_dim': 16, 'train_classes': train_classes},
        'episode': {'n_way': 2, 'n_way_val': 2, 'n_way_test': 2, 'k_shot': 2, 'q_query': 2,
                    'episodes_per_epoch': 2, 'episodes_meta_val': 2},
        'optim': {'epochs': 1, 'learning_rate': 1e-3, 'eval_every': 1},
        'data': {'manifest_path': str(manifest_path), 'dataset_root': str(tmp_path),
                 'train_classes': 2, 'val_classes': 2, 'test_classes': 2,
                 'frame_count': 4, 'resize_size': 24, 'spatial_size': 16},
        'seed': 42,
        'output_root': str(tmp_path / 'outputs'),
        'log_root': str(tmp_path / 'logs'),
        'run_id': 'unittest_safsar',
    }


def test_safsar_train_then_eval_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.eval_episodic import run_eval  # noqa: PLC0415
    from src.training.meta_trainer import run_training  # noqa: PLC0415

    manifest_path = _make_synthetic_manifest(tmp_path)
    df_full = pd.read_csv(manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False)
    fake = lambda *a, **kw: _SyntheticDataset(df_full, modality='rgb')  # noqa: E731
    monkeypatch.setattr('src.training.meta_trainer.ThetisDataset', fake)
    monkeypatch.setattr('src.training.eval_episodic.ThetisDataset', fake)
    monkeypatch.setattr('src.models.factory.VideoMAEEncoder', lambda **_kw: _ToyVideoMAE())
    meta_train = split_classes(sorted(df_full['action_label'].unique()), 2, 2, 2, seed=42)['meta_train']

    with pytest.raises(ValueError, match='meta_train split'):
        run_training(_cfg(tmp_path, manifest_path, ['smash', 'backhand']), smoke=True, device_arg='cpu')

    log = run_training(_cfg(tmp_path, manifest_path, meta_train), smoke=True, device_arg='cpu')
    assert np.isfinite(log['epochs'][0]['train_loss'])
    ckpt = tmp_path / 'outputs' / 'checkpoints' / 'smoke_unittest_safsar' / 'best.pt'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert blob['config']['method'] == 'safsar' and 'text_tokens' in blob['model_state']
    assert run_eval(ckpt, None, tmp_path / 'outputs', n_episodes=3, device_arg='cpu')['n_episodes'] == 3


# ---------------------------------------------------------------------------
# VideoMAEEncoder (random init, no download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def videomae() -> VideoMAEEncoder:
    torch.manual_seed(0)
    return VideoMAEEncoder(pretrained=False).eval()


def test_videomae_matches_hugging_face_at_native_size(videomae: VideoMAEEncoder) -> None:
    x = torch.randint(0, 256, (1, 16, 224, 224, 3), dtype=torch.uint8)
    with torch.no_grad():
        ours = videomae(x)
        pixels = preprocess_video_batch(x, mean=IMAGENET_MEAN, std=IMAGENET_STD).permute(0, 2, 1, 3, 4)
        theirs = videomae.backbone(pixel_values=pixels).logits  # classifier is Identity → pooled features
    assert ours.shape == (1, 768)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_videomae_resamples_positions_at_112(videomae: VideoMAEEncoder) -> None:
    with torch.no_grad():
        out = videomae(torch.randint(0, 256, (2, 8, 112, 112, 3), dtype=torch.uint8))
    assert out.shape == (2, 768) and torch.isfinite(out).all()
    assert not any(p.requires_grad for p in videomae.backbone.videomae.embeddings.patch_embeddings.parameters())


def test_videomae_checkpointing_matches_plain_forward() -> None:
    torch.manual_seed(0)
    enc = build_encoder({'name': 'videomae_base'}, pretrained=False, use_checkpointing=True).train()
    assert isinstance(enc, VideoMAEEncoder) and not enc.per_frame
    x = torch.randint(0, 256, (1, 4, 64, 64, 3), dtype=torch.uint8)
    ckpt = enc(x)
    ckpt.sum().backward()
    assert enc.backbone.fc_norm.weight.grad is not None
    enc.backbone.gradient_checkpointing_disable()
    assert torch.allclose(enc(x), ckpt, atol=1e-5)  # no BatchNorm/dropout: checkpointing is result-neutral


def test_videomae_bert_tokens(videomae: VideoMAEEncoder) -> None:
    try:
        tokens, padding = videomae.encode_text_tokens(['a tennis smash', 'a kick tennis serve hit with topspin'])
    except OSError as exc:  # tokenizer not cached and no network
        pytest.skip(f'BERT tokenizer unavailable: {exc}')
    assert tokens.shape[0] == 2 and tokens.shape[2] == 768 and tokens.dtype == torch.float32
    assert padding[0].any() and not padding[1].any()  # the shorter sentence is padded
