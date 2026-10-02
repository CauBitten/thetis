'''Config → encoder/model wiring shared by the meta-trainer and the evaluator.

``encoder.name`` picks the encoder family (:data:`VIDEO_BACKBONES` → clip-level
:class:`VideoEncoder`, :data:`FRAME_BACKBONES` → per-frame
:class:`FrameEncoder`), ``method`` picks the head from :data:`METHODS`, and
the optional ``model`` mapping of the config is forwarded to the head's
constructor as keyword arguments (method-specific hyperparameters).
'''
from __future__ import annotations

from typing import Any

from torch import nn

from src.models.base import EpisodicModel
from src.models.encoders import FRAME_BACKBONES, VIDEO_BACKBONES, FrameEncoder, VideoEncoder
from src.models.protonet import ProtoNet
from src.models.trx import TRX

METHODS: dict[str, type[EpisodicModel]] = {
    'protonet': ProtoNet,
    'trx': TRX,
}


def build_encoder(encoder_cfg: dict[str, Any], pretrained: bool, use_checkpointing: bool = False) -> nn.Module:
    name = str(encoder_cfg.get('name', 'r2plus1d_18')).lower()
    if name in VIDEO_BACKBONES:
        return VideoEncoder(name=name, pretrained=pretrained, use_checkpointing=use_checkpointing)
    if name in FRAME_BACKBONES:
        return FrameEncoder(name=name, pretrained=pretrained, use_checkpointing=use_checkpointing)
    raise ValueError(
        f'unknown encoder.name {name!r}; video: {list(VIDEO_BACKBONES)}, per-frame: {list(FRAME_BACKBONES)}'
    )


def build_model(cfg: dict[str, Any], encoder: nn.Module, encoder_batch_size: int | None = None) -> EpisodicModel:
    method = cfg['method']
    model_cls = METHODS.get(method)
    if model_cls is None:
        raise NotImplementedError(f'method {method!r} not implemented; available: {list(METHODS)}')
    if model_cls.needs_frame_features and not getattr(encoder, 'per_frame', False):
        raise ValueError(
            f'method {method!r} needs per-frame features; set encoder.name to one of {list(FRAME_BACKBONES)}'
        )
    return model_cls(encoder, encoder_batch_size=encoder_batch_size, **cfg.get('model', {}))


__all__ = ['METHODS', 'build_encoder', 'build_model']
