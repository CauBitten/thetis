'''Video encoders used as backbones for FSAR methods.

Two families, both taking ``(B, T, H, W, 3)`` uint8 input — the same layout
returned by :class:`src.data.loader.ThetisDataset` — and handling the permute +
normalisation internally via :func:`preprocess_video_batch`:

- :class:`VideoEncoder` (clip-level, ``(B, embed_dim)``): a torchvision 3D
  backbone (``r2plus1d_18`` / ``r3d_18``) pre-trained on Kinetics-400 with its
  classifier replaced by an identity. The trunk pools time away (it strides T
  by 8), so it only serves heads that compare whole clips (ProtoNet).
- :class:`FrameEncoder` (per-frame, ``(B, T, embed_dim)``): a torchvision 2D
  ResNet pre-trained on ImageNet, applied to every frame. Keeps the temporal
  axis that frame-matching heads (TRX) need; ProtoNet averages it.

Gradient checkpointing (``use_checkpointing=True``) trades ~30% extra
compute for ~70% less activation memory by re-running each ResNet stage
during the backward pass. Essential on ≤6 GB GPUs. It is *not* result-neutral:
the re-run happens in ``train()`` mode, so every BatchNorm updates its running
statistics twice per step — keep the setting fixed across compared runs.
'''
from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

# Kinetics-400 normalisation statistics (torchvision R(2+1)D / R3D defaults).
KINETICS_MEAN: tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
KINETICS_STD: tuple[float, float, float] = (0.22803, 0.22145, 0.216989)
# ImageNet normalisation statistics (torchvision 2D classification defaults).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

VIDEO_BACKBONES: tuple[str, ...] = ('r2plus1d_18', 'r3d_18')
FRAME_BACKBONES: tuple[str, ...] = ('resnet18', 'resnet34', 'resnet50')


def preprocess_video_batch(
    x: torch.Tensor,
    mean: tuple[float, float, float] = KINETICS_MEAN,
    std: tuple[float, float, float] = KINETICS_STD,
) -> torch.Tensor:
    '''Convert ``(B, T, H, W, 3)`` uint8/float into ``(B, 3, T, H, W)`` float32, normalized.

    Accepts uint8 (divides by 255) or float (assumed in [0, 1] or [0, 255] —
    inferred from max). Normalises with ``mean``/``std`` (Kinetics by default).
    '''
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError(f'expected (B, T, H, W, 3) tensor, got shape {tuple(x.shape)}')
    # Target dtype honours an enclosing autocast: under AMP the clip enters the
    # backbone in fp16 (half the bytes) instead of a full fp32 copy the conv
    # would immediately re-cast anyway. Falls back to fp32 (no autocast / CPU).
    out_dtype = torch.float32
    if x.is_cuda and torch.is_autocast_enabled():
        out_dtype = (
            torch.get_autocast_dtype('cuda')
            if hasattr(torch, 'get_autocast_dtype')
            else torch.get_autocast_gpu_dtype()
        )
    # uint8 always scales by 255; float is assumed in [0, 1] unless it looks like
    # 0-255 (short-circuits so the uint8 path never pays the max() device sync).
    divide = x.dtype == torch.uint8 or (x.numel() > 0 and float(x.max()) > 1.5)
    # Single cast+copy straight into channel-first contiguous layout (B, 3, T, H, W),
    # then normalise in place — no extra full-size temporaries.
    y = x.permute(0, 4, 1, 2, 3).to(out_dtype, memory_format=torch.contiguous_format)
    if divide:
        y = y.div_(255.0)
    mean_t = torch.tensor(mean, device=y.device, dtype=y.dtype).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=y.device, dtype=y.dtype).view(1, 3, 1, 1, 1)
    return y.sub_(mean_t).div_(std_t)


class VideoEncoder(nn.Module):
    '''Wrapper around a torchvision video backbone returning ``(B, embed_dim)``.

    Args:
        name: backbone name. Currently ``'r2plus1d_18'`` (default) and
            ``'r3d_18'`` (fallback). Both expose a 512-dim feature.
        pretrained: load Kinetics-400 weights via torchvision's enum API.
            If the download fails, prints a warning and continues with
            random init.
        use_checkpointing: if True, applies gradient checkpointing on
            ``layer1..layer4`` of the ResNet to slash activation memory.
            Off by default (turn on for small GPUs / large batches).
    '''

    per_frame = False

    def __init__(
        self,
        name: str = 'r2plus1d_18',
        pretrained: bool = True,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        backbone, embed_dim = _load_backbone(name, pretrained=pretrained)
        self.name = name
        self.backbone = backbone
        self.embed_dim: int = embed_dim
        self.use_checkpointing = bool(use_checkpointing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''``(B, T, H, W, 3)`` uint8/float in → ``(B, embed_dim)`` float32 out.'''
        x = preprocess_video_batch(x)
        if not self.use_checkpointing or not torch.is_grad_enabled():
            return self.backbone(x)
        return self._forward_checkpointed(x)

    def _forward_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        '''Per-stage checkpoint: stem → layer1 → ... → layer4 → avgpool → fc.

        Each block's activations are re-computed during backward instead of
        kept in memory. Cuts activation memory ~70% at the cost of one extra
        forward per stage.
        '''
        b = self.backbone
        x = _grad_checkpoint(b.stem, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer1, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer2, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer3, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer4, x, use_reentrant=False)
        x = b.avgpool(x)
        x = x.flatten(1)
        return b.fc(x)


class FrameEncoder(nn.Module):
    '''2D ImageNet ResNet applied to every frame: ``(B, T, H, W, 3)`` → ``(B, T, embed_dim)``.

    Frames are folded into the batch axis, so in ``train()`` mode each
    BatchNorm normalises over the ``B*T`` frames of a chunk (frames of the same
    clip are highly correlated). As with :class:`VideoEncoder`, the chunk size
    (``encoder.batch_size``, in videos) is therefore part of the protocol.

    Args:
        name: ``'resnet18'``, ``'resnet34'`` (512-dim) or ``'resnet50'``
            (2048-dim, the TRX backbone).
        pretrained: load ImageNet weights (``IMAGENET1K_V1``, the ones TRX
            used). If the download fails, warns and continues with random init.
        use_checkpointing: gradient checkpointing on the stem and
            ``layer1..layer4``, same trade-off as :class:`VideoEncoder`.
    '''

    per_frame = True

    def __init__(
        self,
        name: str = 'resnet50',
        pretrained: bool = True,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        backbone, embed_dim = _load_frame_backbone(name, pretrained=pretrained)
        self.name = name
        self.backbone = backbone
        self.embed_dim: int = embed_dim
        self.use_checkpointing = bool(use_checkpointing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''``(B, T, H, W, 3)`` uint8/float in → ``(B, T, embed_dim)`` out.'''
        b, t = x.shape[0], x.shape[1]
        y = preprocess_video_batch(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)  # (B, 3, T, H, W)
        y = y.transpose(1, 2).flatten(0, 1)                                  # (B*T, 3, H, W)
        if not self.use_checkpointing or not torch.is_grad_enabled():
            feats = self.backbone(y)
        else:
            feats = self._forward_checkpointed(y)
        return feats.view(b, t, -1)

    def _forward_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = _grad_checkpoint(_resnet2d_stem, b, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer1, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer2, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer3, x, use_reentrant=False)
        x = _grad_checkpoint(b.layer4, x, use_reentrant=False)
        x = b.avgpool(x)
        x = x.flatten(1)
        return b.fc(x)


def _resnet2d_stem(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return net.maxpool(net.relu(net.bn1(net.conv1(x))))


def _instantiate(ctor: Any, weights: Any, name: str) -> nn.Module:
    try:
        return ctor(weights=weights)
    except Exception as exc:  # noqa: BLE001 — pretrained download can fail in many ways
        if weights is None:
            raise
        warnings.warn(f'failed to load pretrained {name}: {exc}; falling back to random init', stacklevel=3)
        return ctor(weights=None)


def _load_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import video as tv_video  # noqa: PLC0415

    name = name.lower()
    if name == 'r2plus1d_18':
        weights_enum = tv_video.R2Plus1D_18_Weights
        ctor = tv_video.r2plus1d_18
    elif name == 'r3d_18':
        weights_enum = tv_video.R3D_18_Weights
        ctor = tv_video.r3d_18
    else:
        raise ValueError(f'unknown video backbone: {name!r}')

    model = _instantiate(ctor, weights_enum.KINETICS400_V1 if pretrained else None, name)
    embed_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, embed_dim


def _load_frame_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision import models as tv_models  # noqa: PLC0415

    name = name.lower()
    ctors = {
        'resnet18': (tv_models.resnet18, tv_models.ResNet18_Weights),
        'resnet34': (tv_models.resnet34, tv_models.ResNet34_Weights),
        'resnet50': (tv_models.resnet50, tv_models.ResNet50_Weights),
    }
    if name not in ctors:
        raise ValueError(f'unknown frame backbone: {name!r}')
    ctor, weights_enum = ctors[name]

    model = _instantiate(ctor, weights_enum.IMAGENET1K_V1 if pretrained else None, name)
    embed_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, embed_dim


__all__ = [
    'FRAME_BACKBONES',
    'FrameEncoder',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'KINETICS_MEAN',
    'KINETICS_STD',
    'VIDEO_BACKBONES',
    'VideoEncoder',
    'preprocess_video_batch',
]
