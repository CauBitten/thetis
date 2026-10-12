'''Tests for src/models/mvp_shot.py (MVP-Shot) and the CLIP encoder it runs on.

The OTAM check compares against a line-by-line port of ``OTAM_cum_dist_v2``
from CLIP-FSAR (github.com/alibaba-mmai-research/CLIP-FSAR,
``models/base/few_shot.py``). CLIP tests build the model with random weights,
so no download is needed.
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.encoders import CLIP_MEAN, CLIP_STD, ClipFrameEncoder, FrameEncoder, preprocess_video_batch
from src.models.factory import build_encoder, build_model
from src.models.mvp_shot import CLASS_TEXTS, PSTI, MVPShot, otam_distance
from tests.test_training import _make_synthetic_manifest, _SyntheticDataset


def _reference_otam(dists: torch.Tensor, lbda: float = 0.5) -> torch.Tensor:
    '''Verbatim port of CLIP-FSAR's ``OTAM_cum_dist_v2`` (input ``[n_q, n_s, L_q, L_s]``).'''
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]
    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda)
            + torch.exp(- cum_dists[:, :, l, 0] / lbda))
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda)
            + torch.exp(- cum_dists[:, :, l, -2] / lbda))
    return cum_dists[:, :, -1, -1]


# ---------------------------------------------------------------------------
# OTAM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('length', [8, 4, 2, 1])
def test_otam_matches_clipfsar_reference(length: int) -> None:
    torch.manual_seed(length)
    dists = 2 * torch.rand(3, 4, length, length)  # cosine distances live in [0, 2]
    assert torch.allclose(otam_distance(dists), _reference_otam(dists), atol=1e-5)


def test_otam_prefers_the_aligned_order() -> None:
    frames = F.normalize(torch.randn(8, 16), dim=-1)
    def dist(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(otam_distance((1 - a @ b.T)[None, None])[0, 0])
    assert dist(frames, frames) < dist(frames, frames.flip(0))


# ---------------------------------------------------------------------------
# PSTI
# ---------------------------------------------------------------------------


def test_psti_outputs_one_tensor_per_velocity() -> None:
    torch.manual_seed(0)
    psti = PSTI(dim=8, n_scales=3, heads=2, mlp_dim=16, attn_dropout=0.0, ffn_dropout=0.0).eval()
    x, text = torch.randn(2, 8, 8), torch.randn(2, 8)
    with torch.no_grad():
        with_text = psti(x, text)
        without_text = psti(x)
    assert [tuple(v.shape) for v in with_text] == [(2, 8, 8), (2, 4, 8), (2, 2, 8)]
    assert [tuple(v.shape) for v in without_text] == [(2, 8, 8), (2, 4, 8), (2, 2, 8)]
    assert not torch.allclose(with_text[0], without_text[0])  # the text token changes the support path
    with pytest.raises(ValueError, match='divisible by 4'):
        psti(torch.randn(2, 6, 8))


# ---------------------------------------------------------------------------
# MVP-Shot head (toy CLIP-like encoder)
# ---------------------------------------------------------------------------


class _ToyClipEncoder(nn.Module):
    '''Per-frame mean colour → Linear, plus a deterministic ``encode_text`` that records its prompts.'''

    per_frame = True

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Linear(3, dim)
        self.prompts: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.to(torch.float32).mean(dim=(2, 3)) / 255.0)

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        self.prompts.extend(prompts)
        return torch.randn(len(prompts), self.embed_dim, generator=torch.Generator().manual_seed(0))


def _toy_model(**kw: Any) -> MVPShot:
    torch.manual_seed(0)
    return MVPShot(_ToyClipEncoder(), heads=2, mlp_dim=16, **kw)


def _episode(k_shot: int = 2, n_query: int = 2, t: int = 8) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    colours = rng.integers(30, 220, size=(2, 3))
    def clip(c: int) -> np.ndarray:
        ramp = np.linspace(0.6, 1.4, t)[:, None, None, None]
        return np.clip(colours[c] * ramp * np.ones((t, 4, 4, 3)), 0, 255).astype(np.uint8)
    return {
        'support': torch.from_numpy(np.stack([clip(c) for c in (0, 1) for _ in range(k_shot)])),
        'query': torch.from_numpy(np.stack([clip(c) for c in (0, 1) for _ in range(n_query)])),
        'support_labels': torch.tensor([0] * k_shot + [1] * k_shot),
        'query_labels': torch.tensor([0] * n_query + [1] * n_query),
        'classes': ['flat_service', 'smash'],
    }


def test_mvp_shot_embeds_every_class_prompt_once() -> None:
    model = _toy_model(prompt_template='a photo of {}', class_texts={'smash': 'an overhead smash'})
    prompts = model.encoder.prompts
    assert len(prompts) == len(CLASS_TEXTS)
    assert 'a photo of a tennis flat serve' in prompts and 'a photo of an overhead smash' in prompts
    assert tuple(model.text_embeddings.shape) == (len(CLASS_TEXTS), 8)
    assert 'text_embeddings' in model.state_dict()  # saved with the checkpoint → eval needs no text tower


def test_mvp_shot_forward_backward() -> None:
    model = _toy_model()
    ep = _episode()
    out = model(ep['support'], ep['query'], ep['support_labels'], ep['query_labels'], n_way=2, class_names=ep['classes'])
    assert out['logits'].shape == (4, 2)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.encoder.proj.weight.grad is not None
    assert model.psti.blocks[0].channel_mlp[0].weight.grad is not None
    assert model.psti.downsample[0].logits.grad is not None


def test_mvp_shot_uses_the_support_class_text() -> None:
    model = _toy_model().eval()
    ep = _episode()
    emb_s, emb_q = model.encoder(ep['support']), model.encoder(ep['query'])
    with torch.no_grad():
        logits = model.query_logits(emb_q, model.support_state(emb_s, ep['support_labels'], 2, ['flat_service', 'smash']))
        other = model.query_logits(emb_q, model.support_state(emb_s, ep['support_labels'], 2, ['kick_service', 'smash']))
    assert not torch.allclose(logits[:, 0], other[:, 0])  # class 0's text changed
    assert torch.allclose(logits[:, 1], other[:, 1])      # class 1's prototype did not


def test_mvp_shot_requires_known_class_names() -> None:
    model = _toy_model()
    emb = torch.randn(4, 8, 8)
    labels = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match='class_names'):
        model.support_state(emb, labels, 2)
    with pytest.raises(ValueError, match='no class text'):
        model.support_state(emb, labels, 2, ['flat_service', 'cartwheel'])
    with pytest.raises(ValueError, match='per-frame'):
        model.support_state(torch.randn(4, 8), labels, 2, ['flat_service', 'smash'])


def test_build_model_mvp_shot_needs_a_clip_encoder() -> None:
    with pytest.raises(ValueError, match='CLIP encoder'):
        build_model({'method': 'mvp_shot'}, FrameEncoder(name='resnet18', pretrained=False))
    cfg = {'method': 'mvp_shot', 'model': {'heads': 2, 'mlp_dim': 16, 'scale_weights': [1.0, 0.5, 0.25]}}
    model = build_model(cfg, _ToyClipEncoder(), encoder_batch_size=2)
    assert isinstance(model, MVPShot) and model.scale_weights == (1.0, 0.5, 0.25)


def test_mvp_shot_streaming_step_matches_classic_step() -> None:
    from src.training.meta_trainer import _train_step_streaming  # noqa: PLC0415

    def _build() -> MVPShot:
        model = _toy_model(attn_dropout=0.0, ffn_dropout=0.0)
        model.encoder_batch_size = 1
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


def test_mvp_shot_train_then_eval_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    '''method: mvp_shot through run_training (smoke) and run_eval: episode class names reach the head.'''
    from src.training.eval_episodic import run_eval  # noqa: PLC0415
    from src.training.meta_trainer import run_training  # noqa: PLC0415

    manifest_path = _make_synthetic_manifest(tmp_path)
    df_full = pd.read_csv(manifest_path, dtype={'actor': str, 'action_code': str}, keep_default_na=False)

    def _fake_dataset(*args: Any, **kwargs: Any) -> Any:
        return _SyntheticDataset(df_full, modality='rgb')

    monkeypatch.setattr('src.training.meta_trainer.ThetisDataset', _fake_dataset)
    monkeypatch.setattr('src.training.eval_episodic.ThetisDataset', _fake_dataset)
    monkeypatch.setattr('src.models.factory.ClipFrameEncoder', lambda **_kw: _ToyClipEncoder())

    cfg = {
        'method': 'mvp_shot',
        'modalities': ['rgb'],
        'encoder': {'name': 'clip_rn50', 'pretrained': False, 'batch_size': 2},
        'model': {'heads': 2, 'mlp_dim': 16},
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
        'run_id': 'unittest_mvp',
    }
    log = run_training(cfg, smoke=True, device_arg='cpu')
    assert np.isfinite(log['epochs'][0]['train_loss'])

    ckpt = tmp_path / 'outputs' / 'checkpoints' / 'smoke_unittest_mvp' / 'best.pt'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert blob['config']['method'] == 'mvp_shot' and 'text_embeddings' in blob['model_state']

    payload = run_eval(ckpt, None, tmp_path / 'outputs', n_episodes=3, device_arg='cpu')
    assert payload['n_episodes'] == 3


# ---------------------------------------------------------------------------
# ClipFrameEncoder (random init, no download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def clip_encoder() -> ClipFrameEncoder:
    torch.manual_seed(0)
    return ClipFrameEncoder(pretrained=False).eval()


def test_clip_encoder_matches_open_clip_at_native_size(clip_encoder: ClipFrameEncoder) -> None:
    x = torch.randint(0, 256, (1, 2, 224, 224, 3), dtype=torch.uint8)
    with torch.no_grad():
        ours = clip_encoder(x)
        frames = preprocess_video_batch(x, mean=CLIP_MEAN, std=CLIP_STD).transpose(1, 2).flatten(0, 1)
        theirs = clip_encoder.backbone(frames).view(1, 2, -1)  # open_clip's own forward
    assert ours.shape == (1, 2, 1024)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_clip_encoder_interpolates_position_embedding_at_112(clip_encoder: ClipFrameEncoder) -> None:
    with torch.no_grad():
        out = clip_encoder(torch.randint(0, 256, (2, 3, 112, 112, 3), dtype=torch.uint8))
    assert out.shape == (2, 3, 1024) and torch.isfinite(out).all()
    assert clip_encoder.backbone.attnpool.positional_embedding.shape == (50, 2048)  # stored grid unchanged


def test_clip_encoder_checkpointing_matches_plain_forward() -> None:
    torch.manual_seed(0)
    enc = build_encoder({'name': 'clip_rn50'}, pretrained=False)
    assert isinstance(enc, ClipFrameEncoder) and enc.per_frame
    x = torch.randint(0, 256, (1, 2, 64, 64, 3), dtype=torch.uint8)
    plain = enc(x)
    enc.use_checkpointing = True
    ckpt = enc(x)
    assert torch.allclose(plain, ckpt, atol=1e-5)
    ckpt.sum().backward()
    assert enc.backbone.attnpool.positional_embedding.grad is not None  # gradient flows through the resize


def test_clip_encoder_text_tower_is_not_a_submodule(clip_encoder: ClipFrameEncoder) -> None:
    text = clip_encoder.encode_text(['a photo of a tennis smash', 'a photo of a tennis kick serve'])
    assert text.shape == (2, 1024) and text.dtype == torch.float32
    assert not any(k.startswith(('transformer', 'token_embedding')) for k in clip_encoder.state_dict())
