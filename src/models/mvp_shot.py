'''MVP-Shot: Multi-Velocity Progressive-Alignment (Qu et al., IEEE TMM 2025).

There is no official code release, so this follows the paper (arXiv
2405.02077v4, §III) and, where the paper is silent, the code of CLIP-FSAR
(github.com/alibaba-mmai-research/CLIP-FSAR) — the CLIP-based method MVP-Shot
extends and compares against, with the same encoder, text and OTAM metric.

Pipeline for an episode (per-frame CLIP features ``(B, T, C)``, T = 8):

1. **PSTI** — Progressive Semantic-Tailored Interaction: ``n_scales`` STI
   blocks, one per velocity (8, 4 and 2 tokens). Each block runs

   - the *Temporal Relation Network* (Eq. 1-3): the class-name text token is
     concatenated to the frame tokens and one pre-norm transformer layer
     (multi-head self-attention + FFN) runs over the ``T+1`` tokens;
   - the *Channel Correction Network* (Eq. 4-7): ``MLP([mean frame token;
     enhanced text token])`` gives a vector that is added to every frame token.

   The next velocity's input is ``Linear(TM(X_out) + Downsample(X_in))``
   (Eq. 8-9), with half the tokens, and the enhanced text token is carried to
   the next block. Query videos run the same blocks without text (§III-C).
2. **MVFA** — Multi-Velocity Feature Alignment (Eq. 10-11): at each velocity,
   support videos are averaged per class into prototypes and compared with the
   query by the OTAM temporal metric; the logits are ``-Σ_n α_n D_n``.

Choices where the paper gives no detail (also listed in
``experiments/configs/README.md``):

- query videos carry no text: no text token in the temporal transformer, and
  a zero vector in place of the text in the channel-correction MLP;
- ``Downsample`` is a learnable softmax-weighted pooling of each pair of
  neighbouring tokens (it starts as the mean); ``TM`` is the mean of each pair;
- ``α_n = 1`` for every velocity (``scale_weights``);
- from CLIP-FSAR: OTAM is its ``OTAM_cum_dist_v2`` (λ = 0.5) summed over both
  directions; the transformer has 8 heads, FFN width 2048 and dropouts
  0.2 (attention output) / 0.05 (FFN); the text token goes after the frames;
  the prompt is ``"a photo of {}"`` over the CLIP text embedding, unnormalised;
- class names are spelled out in tennis terms (:data:`CLASS_TEXTS`).
'''
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.base import EpisodicModel

#: canonical THETIS label → the text that fills the prompt template.
CLASS_TEXTS: dict[str, str] = {
    'backhand': 'a tennis backhand',
    'backhand2hands': 'a two-handed tennis backhand',
    'backhand_slice': 'a tennis backhand slice',
    'backhand_volley': 'a tennis backhand volley',
    'flat_service': 'a tennis flat serve',
    'forehand_flat': 'a tennis flat forehand',
    'forehand_openstands': 'an open stance tennis forehand',
    'forehand_slice': 'a tennis forehand slice',
    'forehand_volley': 'a tennis forehand volley',
    'kick_service': 'a tennis kick serve',
    'slice_service': 'a tennis slice serve',
    'smash': 'a tennis smash',
}


def otam_distance(dists: torch.Tensor, lbda: float = 0.5) -> torch.Tensor:
    '''One-directional OTAM distance: ``(..., L_query, L_support)`` frame distances → ``(...)``.

    Same recursion as CLIP-FSAR's ``OTAM_cum_dist_v2``: the support axis is
    zero-padded at both ends so the alignment may start and end anywhere, the
    top row is a plain cumulative sum, and every other cell adds its frame
    distance to a soft minimum (temperature ``lbda``) over its allowed
    predecessors. The soft minimum uses ``logsumexp`` (stable where
    ``-λ·log(Σ exp(-x/λ))`` would underflow); the result is otherwise identical.
    '''
    d = F.pad(dists, (1, 1))
    n_rows, n_cols = d.shape[-2], d.shape[-1]

    def softmin(*xs: torch.Tensor) -> torch.Tensor:
        return -lbda * torch.logsumexp(torch.stack(xs) / -lbda, dim=0)

    zero = torch.zeros_like(d[..., 0, 0])
    prev = [zero]
    for m in range(1, n_cols):
        prev.append(d[..., 0, m] + prev[m - 1])
    for l in range(1, n_rows):
        row = [zero]  # column 0 is never updated in the original recursion
        row.append(d[..., l, 1] + softmin(prev[0], prev[1], row[0]))
        for m in range(2, n_cols - 1):
            row.append(d[..., l, m] + softmin(prev[m - 1], row[m - 1]))
        row.append(d[..., l, n_cols - 1] + softmin(prev[n_cols - 2], prev[n_cols - 1], row[n_cols - 2]))
        prev = row
    return prev[-1]


def _pair_mean(x: torch.Tensor) -> torch.Tensor:
    '''TM (Eq. 8): ``(B, T, C)`` → ``(B, T/2, C)``, mean of each pair of neighbouring tokens.'''
    b, t, c = x.shape
    return x.reshape(b, t // 2, 2, c).mean(dim=2)


class WeightedPairPool(nn.Module):
    '''Downsample (Eq. 9): learnable softmax-weighted pooling of each pair of neighbouring tokens.'''

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        weights = torch.softmax(self.logits, dim=0).to(x.dtype)
        return torch.einsum('bpkc,k->bpc', x.reshape(b, t // 2, 2, c), weights)


class STIBlock(nn.Module):
    '''Semantic-Tailored Interaction at one velocity (Fig. 3): temporal relation + channel correction.'''

    def __init__(self, dim: int, heads: int, mlp_dim: int, attn_dropout: float, ffn_dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(ffn_dropout), nn.Linear(mlp_dim, dim), nn.Dropout(ffn_dropout)
        )
        self.channel_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, text: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        '''Frame tokens ``(B, T, C)`` and text token ``(B, C)`` (``None`` for queries) → ``X_out``, enhanced text.'''
        n_frames = x.shape[1]
        tokens = x if text is None else torch.cat([x, text.to(x.dtype).unsqueeze(1)], dim=1)
        h = self.norm_attn(tokens)
        tokens = tokens + self.attn_drop(self.attn(h, h, h, need_weights=False)[0])  # Eq. 1-2
        tokens = tokens + self.ffn(self.norm_ffn(tokens))                            # Eq. 3
        frames = tokens[:, :n_frames]
        text_out = None if text is None else tokens[:, n_frames]
        context = frames.mean(dim=1)                                                 # Eq. 4
        text_in = torch.zeros_like(context) if text_out is None else text_out
        modulation = self.channel_mlp(torch.cat([context, text_in], dim=-1))         # Eq. 5-6
        return frames + modulation.unsqueeze(1), text_out                            # Eq. 7


class PSTI(nn.Module):
    '''Progressive Semantic-Tailored Interaction: one STI block per velocity, halving T between blocks.'''

    def __init__(
        self, dim: int, n_scales: int, heads: int, mlp_dim: int, attn_dropout: float, ffn_dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            STIBlock(dim, heads, mlp_dim, attn_dropout, ffn_dropout) for _ in range(n_scales)
        )
        self.downsample = nn.ModuleList(WeightedPairPool() for _ in range(n_scales - 1))
        self.fuse = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_scales - 1))

    def forward(self, x: torch.Tensor, text: torch.Tensor | None = None) -> list[torch.Tensor]:
        '''``(B, T, C)`` → one ``(B, T / 2**n, C)`` tensor per velocity ``n``.'''
        divisor = 2 ** (len(self.blocks) - 1)
        if x.shape[1] % divisor:
            raise ValueError(f'{len(self.blocks)} velocities need T divisible by {divisor}, got T={x.shape[1]}')
        outputs = []
        x_in = x
        for n, block in enumerate(self.blocks):
            x_out, text = block(x_in, text)
            outputs.append(x_out)
            if n + 1 < len(self.blocks):
                x_in = self.fuse[n](_pair_mean(x_out) + self.downsample[n](x_in))  # Eq. 8-9
        return outputs


class MVPShot(EpisodicModel):
    '''MVP-Shot head over a CLIP per-frame encoder.

    Args:
        encoder: per-frame encoder exposing ``embed_dim`` and ``encode_text``
            (:class:`src.models.encoders.ClipFrameEncoder`).
        encoder_batch_size: see :class:`src.models.base.EpisodicModel`.
        n_scales: number of velocities ``N`` (T must be divisible by ``2**(N-1)``).
        scale_weights: ``α_n`` of Eq. 11 (default: all 1).
        heads, mlp_dim, attn_dropout, ffn_dropout: the STI transformer layer.
        otam_lambda: temperature of the OTAM soft minimum.
        prompt_template: filled with :data:`CLASS_TEXTS` and embedded once by
            the frozen CLIP text tower; the embeddings are a buffer saved with
            the model, so evaluation does not need the pretrained text tower.
        class_texts: overrides/extends :data:`CLASS_TEXTS`.
    '''

    needs_frame_features = True

    def __init__(
        self,
        encoder: nn.Module,
        encoder_batch_size: int | None = None,
        n_scales: int = 3,
        scale_weights: Sequence[float] | None = None,
        heads: int = 8,
        mlp_dim: int = 2048,
        attn_dropout: float = 0.2,
        ffn_dropout: float = 0.05,
        otam_lambda: float = 0.5,
        prompt_template: str = 'a photo of {}',
        class_texts: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(encoder, encoder_batch_size)
        dim = getattr(encoder, 'embed_dim', None)
        encode_text = getattr(encoder, 'encode_text', None)
        if dim is None or encode_text is None:
            raise ValueError('MVP-Shot needs a CLIP encoder with embed_dim and encode_text (encoder.name: clip_rn50)')
        weights = [1.0] * int(n_scales) if scale_weights is None else [float(w) for w in scale_weights]
        if len(weights) != int(n_scales):
            raise ValueError(f'scale_weights has {len(weights)} values for n_scales={n_scales}')
        self.scale_weights = tuple(weights)
        self.otam_lambda = float(otam_lambda)
        self.psti = PSTI(int(dim), int(n_scales), int(heads), int(mlp_dim), float(attn_dropout), float(ffn_dropout))

        texts = {**CLASS_TEXTS, **(class_texts or {})}
        names = sorted(texts)
        self.class_index = {name: i for i, name in enumerate(names)}
        prompts = [prompt_template.format(texts[name]) for name in names]
        self.register_buffer('text_embeddings', encode_text(prompts).float())

    @staticmethod
    def _check_frames(emb: torch.Tensor) -> None:
        if emb.ndim != 3:
            raise ValueError(f'MVP-Shot needs per-frame (B, T, D) features; got shape {tuple(emb.shape)}')

    def support_state(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        class_names: Sequence[str] | None = None,
    ) -> list[torch.Tensor]:
        '''Per velocity, class prototypes ``(n_way, T_n, C)``: PSTI with each video's class text, then the class mean.'''
        self._check_frames(support_emb)
        if class_names is None or len(class_names) < n_way:
            raise ValueError('MVP-Shot needs the class_names of the episode (the support text)')
        missing = [c for c in class_names[:n_way] if c not in self.class_index]
        if missing:
            raise ValueError(f'no class text for {missing}; add them to model.class_texts')
        masks = []
        for c in range(n_way):
            mask = support_labels == c
            if not bool(mask.any()):
                raise ValueError(f'class {c} has no support samples in this episode')
            masks.append(mask)
        episode_text = self.text_embeddings[[self.class_index[c] for c in class_names[:n_way]]]  # (n_way, C)
        velocities = self.psti(support_emb, episode_text[support_labels])
        return [torch.stack([v[m].mean(dim=0) for m in masks]) for v in velocities]

    def query_logits(self, query_emb: torch.Tensor, prototypes: list[torch.Tensor]) -> torch.Tensor:
        '''``(n_query, n_way)`` logits ``-Σ α_n D_n`` (queries run PSTI without text).'''
        self._check_frames(query_emb)
        distance = None
        for alpha, query, protos in zip(self.scale_weights, self.psti(query_emb), prototypes, strict=True):
            d = alpha * self._otam(query, protos)
            distance = d if distance is None else distance + d
        return -distance

    def _otam(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        '''Bidirectional OTAM between every query ``(n_q, T, C)`` and prototype ``(n_way, T, C)`` → ``(n_q, n_way)``.'''
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = F.normalize(query.float(), dim=-1)
            p = F.normalize(prototypes.float(), dim=-1)
            dists = 1 - torch.einsum('qtc,nsc->qnts', q, p)  # rows: query frames, columns: support frames
            return otam_distance(dists, self.otam_lambda) + otam_distance(dists.transpose(-2, -1), self.otam_lambda)


__all__ = ['CLASS_TEXTS', 'MVPShot', 'PSTI', 'STIBlock', 'WeightedPairPool', 'otam_distance']
