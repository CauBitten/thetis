'''Spatio-temporal augmentations for THETIS samples.

All transforms operate on a sample dict (the same schema returned by
:class:`src.data.loader.ThetisDataset`). Video transforms key off
:data:`VIDEO_KEYS` and apply synchronously across every video modality
present in the sample (so RGB/depth/mask/skeleton-video stay aligned).
Coordinate transforms operate on :data:`COORD_KEYS_2D`/:data:`COORD_KEYS_3D`
and are no-ops when the keys are absent — they exist now so a future
``src/features/pose.py`` can drop coordinate arrays into the dict without
changing the augmentation pipeline.

Tensor convention for video keys: ``(T, H, W, 3)`` uint8, RGB. Both
``np.ndarray`` and ``torch.Tensor`` are accepted; the output type matches
the input type.
'''
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# Video keys (output by ThetisDataset.__getitem__)
VIDEO_KEYS: tuple[str, ...] = (
    'rgb',
    'depth',
    'mask',
    'skeleton_2d_video',
    'skeleton_3d_video',
)

# Coordinate keys (currently not produced by the loader; reserved for pose features)
COORD_KEYS_2D: tuple[str, ...] = ('skeleton_2d_coords',)
COORD_KEYS_3D: tuple[str, ...] = ('skeleton_3d_coords',)
COORD_KEYS: tuple[str, ...] = COORD_KEYS_2D + COORD_KEYS_3D


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith('torch')


def _to_numpy(x: Any) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _from_numpy_like(arr: np.ndarray, ref: Any) -> Any:
    if _is_torch(ref):
        import torch  # noqa: PLC0415

        return torch.from_numpy(np.ascontiguousarray(arr))
    return arr


def _video_keys_present(sample: dict[str, Any]) -> list[str]:
    return [k for k in VIDEO_KEYS if k in sample]


def _video_shape(video: Any) -> tuple[int, int, int, int]:
    '''Return ``(T, H, W, C)`` for either ndarray or torch tensor.'''
    shape = tuple(video.shape)
    if len(shape) != 4:
        raise ValueError(f'expected (T,H,W,C) tensor, got shape {shape}')
    return shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class Compose:
    '''Compose a list of sample-dict transforms into one callable.'''

    def __init__(self, transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        self.transforms: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...] = tuple(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        body = ',\n  '.join(repr(t) for t in self.transforms)
        return f'Compose([\n  {body}\n])'


# ---------------------------------------------------------------------------
# Video transforms (synchronised across modalities)
# ---------------------------------------------------------------------------


class RandomTemporalCrop:
    '''Pick a contiguous window of ``num_frames`` frames, identical across modalities.

    If a video has fewer than ``num_frames`` frames, it's padded by repeating
    the last frame (rather than dropping the sample) so the dataset stays usable.
    '''

    def __init__(self, num_frames: int, seed: int | None = None) -> None:
        if num_frames <= 0:
            raise ValueError('num_frames must be positive')
        self.num_frames = int(num_frames)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        keys = _video_keys_present(sample)
        if not keys:
            return sample
        # Use the first video's T to choose the window; all modalities of the
        # same sample share the same number of frames.
        ref = sample[keys[0]]
        t = int(_video_shape(ref)[0])
        if t >= self.num_frames:
            start = int(self._rng.integers(0, t - self.num_frames + 1))
        else:
            start = 0
        for key in keys:
            video = sample[key]
            arr = _to_numpy(video)
            T = arr.shape[0]
            if T >= self.num_frames:
                cropped = arr[start : start + self.num_frames]
            else:
                pad_len = self.num_frames - T
                pad = np.repeat(arr[-1:], pad_len, axis=0)
                cropped = np.concatenate([arr, pad], axis=0)
            sample[key] = _from_numpy_like(cropped, video)
        return sample


class RandomSegmentSample:
    '''TSN-style sampling: one random frame from each of ``num_frames`` equal segments.

    Unlike :class:`RandomTemporalCrop`, the output always spans the whole clip,
    so training sees the full stroke at the same temporal extent as evaluation
    (which samples ``num_frames`` uniformly over the clip). The randomness is
    only the jitter inside each segment. Indices are drawn once per sample and
    shared across modalities.

    If a video has fewer than ``num_frames`` frames, frames are repeated on a
    uniform grid (same as the loader's ``linspace`` resampling) instead of
    padding the tail, so temporal coverage is preserved.
    '''

    def __init__(self, num_frames: int, seed: int | None = None) -> None:
        if num_frames <= 0:
            raise ValueError('num_frames must be positive')
        self.num_frames = int(num_frames)
        self._rng = np.random.default_rng(seed)

    def _indices(self, t: int) -> np.ndarray:
        if t < self.num_frames:
            return np.linspace(0, t - 1, num=self.num_frames).round().astype(int)
        # Integer segment edges; each segment has >= 1 frame because t >= num_frames.
        edges = (np.arange(self.num_frames + 1) * t) // self.num_frames
        return self._rng.integers(edges[:-1], edges[1:])

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        keys = _video_keys_present(sample)
        if not keys:
            return sample
        indices = self._indices(int(_video_shape(sample[keys[0]])[0]))
        for key in keys:
            video = sample[key]
            arr = _to_numpy(video)
            sampled = arr[np.minimum(indices, arr.shape[0] - 1)]
            sample[key] = _from_numpy_like(sampled, video)
        return sample


class RandomSpatialCrop:
    '''Pick a random ``(h,w)`` crop region, identical across modalities.'''

    def __init__(self, size: int | tuple[int, int], seed: int | None = None) -> None:
        if isinstance(size, int):
            self.height = self.width = int(size)
        else:
            self.height, self.width = int(size[0]), int(size[1])
        if self.height <= 0 or self.width <= 0:
            raise ValueError('crop size must be positive')
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        keys = _video_keys_present(sample)
        if not keys:
            return sample
        ref = sample[keys[0]]
        _, h, w, _ = _video_shape(ref)
        target_h = min(self.height, int(h))
        target_w = min(self.width, int(w))
        top = int(self._rng.integers(0, h - target_h + 1)) if h > target_h else 0
        left = int(self._rng.integers(0, w - target_w + 1)) if w > target_w else 0
        for key in keys:
            video = sample[key]
            arr = _to_numpy(video)
            cropped = arr[:, top : top + target_h, left : left + target_w, :]
            sample[key] = _from_numpy_like(cropped, video)
        return sample


class CenterSpatialCrop:
    '''Deterministic center crop to ``(h, w)``, identical across modalities.'''

    def __init__(self, size: int | tuple[int, int]) -> None:
        if isinstance(size, int):
            self.height = self.width = int(size)
        else:
            self.height, self.width = int(size[0]), int(size[1])
        if self.height <= 0 or self.width <= 0:
            raise ValueError('crop size must be positive')

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        keys = _video_keys_present(sample)
        if not keys:
            return sample
        ref = sample[keys[0]]
        _, h, w, _ = _video_shape(ref)
        target_h = min(self.height, int(h))
        target_w = min(self.width, int(w))
        top = max(0, (int(h) - target_h) // 2)
        left = max(0, (int(w) - target_w) // 2)
        for key in keys:
            video = sample[key]
            arr = _to_numpy(video)
            cropped = arr[:, top : top + target_h, left : left + target_w, :]
            sample[key] = _from_numpy_like(cropped, video)
        return sample


class ResizeVideo:
    '''Bilinear resize to ``(h, w)``, identical across modalities.

    Uses cv2.resize per frame; output dtype matches input. The first axis
    (time) is preserved.
    '''

    def __init__(self, size: int | tuple[int, int]) -> None:
        if isinstance(size, int):
            self.height = self.width = int(size)
        else:
            self.height, self.width = int(size[0]), int(size[1])
        if self.height <= 0 or self.width <= 0:
            raise ValueError('resize size must be positive')

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        keys = _video_keys_present(sample)
        if not keys:
            return sample
        try:
            import cv2  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError('ResizeVideo requires opencv-python') from exc

        for key in keys:
            video = sample[key]
            arr = _to_numpy(video)
            T, H, W, C = arr.shape
            if H == self.height and W == self.width:
                continue
            out = np.empty((T, self.height, self.width, C), dtype=arr.dtype)
            for t in range(T):
                out[t] = cv2.resize(arr[t], (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            sample[key] = _from_numpy_like(out, video)
        return sample


class HorizontalFlip:
    '''Flip along width axis with probability ``p``, identical across modalities.

    Sets ``sample['flipped'] = True`` whenever a flip is applied so that
    downstream coordinate transforms (e.g. joint x-mirror) can react.
    '''

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError('p must be in [0,1]')
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self._rng.random() >= self.p:
            return sample
        for key in _video_keys_present(sample):
            video = sample[key]
            arr = _to_numpy(video)
            sample[key] = _from_numpy_like(arr[:, :, ::-1, :].copy(), video)
        # Mirror coord X axes if present
        for key in COORD_KEYS:
            if key not in sample:
                continue
            coords = sample[key]
            arr = _to_numpy(coords).copy()
            arr[..., 0] = -arr[..., 0]
            sample[key] = _from_numpy_like(arr, coords)
        sample['flipped'] = bool(sample.get('flipped', False)) ^ True
        return sample


class ColorJitter:
    '''Per-frame brightness/contrast/saturation/hue jitter on the ``rgb`` key.

    Only touches ``sample['rgb']``. Depth/mask/skeleton videos are left alone
    because color-space jitter is meaningless or destructive on them.

    Implementation is in numpy/HSV; a single jitter setting is sampled per
    sample and applied uniformly across all frames (so motion stays coherent).
    '''

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        seed: int | None = None,
    ) -> None:
        for name, val in [('brightness', brightness), ('contrast', contrast), ('saturation', saturation)]:
            if val < 0.0:
                raise ValueError(f'{name} must be >= 0')
        if not 0.0 <= hue <= 0.5:
            raise ValueError('hue must be in [0, 0.5]')
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self._rng = np.random.default_rng(seed)

    def _factor(self, magnitude: float) -> float:
        if magnitude <= 0.0:
            return 1.0
        return float(self._rng.uniform(max(0.0, 1.0 - magnitude), 1.0 + magnitude))

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if 'rgb' not in sample:
            return sample
        try:
            import cv2  # type: ignore  # noqa: PLC0415
        except ImportError:
            return sample

        video = sample['rgb']
        arr = _to_numpy(video).astype(np.float32)
        b = self._factor(self.brightness)
        c = self._factor(self.contrast)
        s = self._factor(self.saturation)
        h_shift = float(self._rng.uniform(-self.hue, self.hue)) if self.hue > 0 else 0.0

        if b != 1.0:
            arr = arr * b
        if c != 1.0:
            mean = arr.mean(axis=(1, 2, 3), keepdims=True)
            arr = (arr - mean) * c + mean
        arr = np.clip(arr, 0.0, 255.0)

        if s != 1.0 or h_shift != 0.0:
            T, H, W, _ = arr.shape
            flat = arr.reshape(T * H, W, 3).astype(np.uint8)
            hsv = cv2.cvtColor(flat, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + h_shift * 180.0) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * s, 0.0, 255.0)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            arr = rgb.reshape(T, H, W, 3).astype(np.float32)

        sample['rgb'] = _from_numpy_like(arr.astype(np.uint8), video)
        return sample


# ---------------------------------------------------------------------------
# Coordinate transforms (active only when coord keys are present)
# ---------------------------------------------------------------------------


def _coord_targets(target: str | Sequence[str] | None) -> tuple[str, ...]:
    if target is None:
        return COORD_KEYS
    if isinstance(target, str):
        return (target,)
    return tuple(target)


class JointJitter:
    '''Add Gaussian noise N(0, sigma) to coordinate arrays.'''

    def __init__(self, sigma: float, target: str | Sequence[str] | None = None, seed: int | None = None) -> None:
        if sigma < 0:
            raise ValueError('sigma must be >= 0')
        self.sigma = float(sigma)
        self.targets = _coord_targets(target)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for key in self.targets:
            if key not in sample:
                continue
            coords = sample[key]
            arr = _to_numpy(coords).astype(np.float32)
            arr = arr + self._rng.normal(0.0, self.sigma, size=arr.shape).astype(np.float32)
            sample[key] = _from_numpy_like(arr, coords)
        return sample


class RandomScale:
    '''Multiply all coordinates by a random isotropic factor in ``[low, high]``.'''

    def __init__(
        self,
        low: float = 0.9,
        high: float = 1.1,
        target: str | Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if low <= 0 or high < low:
            raise ValueError('require 0 < low <= high')
        self.low = float(low)
        self.high = float(high)
        self.targets = _coord_targets(target)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        factor = float(self._rng.uniform(self.low, self.high))
        for key in self.targets:
            if key not in sample:
                continue
            coords = sample[key]
            arr = _to_numpy(coords).astype(np.float32) * factor
            sample[key] = _from_numpy_like(arr, coords)
        return sample


class RandomRotationXY:
    '''Rotate coordinates in the XY plane by a random angle in [-max_deg, max_deg].'''

    def __init__(
        self,
        max_deg: float = 15.0,
        target: str | Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if max_deg < 0:
            raise ValueError('max_deg must be >= 0')
        self.max_deg = float(max_deg)
        self.targets = _coord_targets(target)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        theta = float(self._rng.uniform(-self.max_deg, self.max_deg)) * np.pi / 180.0
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for key in self.targets:
            if key not in sample:
                continue
            coords = sample[key]
            arr = _to_numpy(coords).astype(np.float32)
            if arr.shape[-1] < 2:
                continue
            x = arr[..., 0]
            y = arr[..., 1]
            arr_xy = np.stack([cos_t * x - sin_t * y, sin_t * x + cos_t * y], axis=-1)
            new = arr.copy()
            new[..., :2] = arr_xy
            sample[key] = _from_numpy_like(new, coords)
        return sample


class JointDropout:
    '''Zero-out each joint independently with probability ``p`` (per sample, not per frame).'''

    def __init__(
        self,
        p: float = 0.1,
        target: str | Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError('p must be in [0,1]')
        self.p = float(p)
        self.targets = _coord_targets(target)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for key in self.targets:
            if key not in sample:
                continue
            coords = sample[key]
            arr = _to_numpy(coords).astype(np.float32)
            # Expect shape (..., J, C) where J is the joint axis. Drop along that axis.
            if arr.ndim < 2:
                continue
            J = arr.shape[-2]
            mask = self._rng.random(J) >= self.p
            keep = mask[None, :, None] if arr.ndim == 3 else np.broadcast_to(mask, arr.shape[:-1])[..., None]
            arr = arr * keep
            sample[key] = _from_numpy_like(arr, coords)
        return sample


__all__ = [
    'VIDEO_KEYS',
    'COORD_KEYS',
    'COORD_KEYS_2D',
    'COORD_KEYS_3D',
    'Compose',
    'RandomTemporalCrop',
    'RandomSegmentSample',
    'RandomSpatialCrop',
    'CenterSpatialCrop',
    'ResizeVideo',
    'HorizontalFlip',
    'ColorJitter',
    'JointJitter',
    'RandomScale',
    'RandomRotationXY',
    'JointDropout',
]
