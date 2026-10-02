'''Temporal-Relational CrossTransformers (TRX; Perrett et al., CVPR 2021).

Ported from the authors' implementation (github.com/tobyperrett/trx). For each
tuple cardinality ``ω`` in ``Ω`` (``temporal_set_sizes``, default ``{2, 3}``):

1. per-frame features ``(B, T, D)`` get a fixed sinusoidal positional encoding
   (scaled by ``pe_scale``) and dropout;
2. every ordered frame tuple — all ``P = C(T, ω)`` of them — is concatenated
   into a ``ω·D`` vector and projected to keys (``LayerNorm(W_k ·)``) and
   values (``W_v ·``) of size ``d_model``;
3. for each class, every query tuple attends (softmax over *all* tuples of all
   that class's support videos) and builds a query-specific class prototype as
   the attention-weighted sum of the support values;
4. the class logit is ``-‖V_q - prototype‖² / P`` (mean over the query tuples).

The final logits average the per-cardinality logits. Hyperparameter names map
to the original ``args``: ``d_model`` = ``trans_linear_out_dim`` (1152),
``dropout`` = ``trans_dropout`` (0.1), ``pe_scale`` = ``pe_scale_factor``
(0.1), ``temporal_set_sizes`` = ``temp_set``.

Adaptations to this codebase:

- built on :class:`src.models.base.EpisodicModel`: the support keys/values are
  the ``support_state``, and each query is scored independently, so the query
  set can be streamed in micro-batches;
- ``T`` comes from the input (tuples are enumerated per call) instead of being
  fixed at construction, so smoke runs with fewer frames work;
- the distance is computed in fp32 with autocast off (AMP-safe), as in ProtoNet;
- the per-query softmax loop of the original is vectorised (same math).
'''
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from src.models.base import EpisodicModel


class PositionalEncoding(nn.Module):
    '''Fixed sinusoidal encoding scaled by ``scale``, followed by dropout. ``(B, T, D) → (B, T, D)``.'''

    def __init__(self, dim: int, dropout: float = 0.1, scale: float = 0.1, max_len: int = 64) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f'positional encoding needs an even feature size, got {dim}')
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term) * scale
        pe[:, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[0]:
            raise ValueError(f'sequence of {x.shape[1]} frames exceeds max_len={self.pe.shape[0]}')
        return self.dropout(x + self.pe[: x.shape[1]].to(x.dtype))


class TemporalCrossTransformer(nn.Module):
    '''One TRX branch: tuples of ``set_size`` frames → keys/values → per-class logits.'''

    def __init__(self, in_dim: int, d_model: int, set_size: int, dropout: float, pe_scale: float) -> None:
        super().__init__()
        self.set_size = int(set_size)
        self.pe = PositionalEncoding(in_dim, dropout=dropout, scale=pe_scale)
        self.k_linear = nn.Linear(in_dim * self.set_size, d_model)
        self.v_linear = nn.Linear(in_dim * self.set_size, d_model)
        self.norm_k = nn.LayerNorm(d_model)

    def _tuples(self, x: torch.Tensor) -> torch.Tensor:
        '''``(B, T, D)`` → ``(B, C(T, set_size), set_size·D)``, tuples in lexicographic order.'''
        x = self.pe(x)
        b, t, _ = x.shape
        if t < self.set_size:
            raise ValueError(f'TRX tuples of {self.set_size} frames need T >= {self.set_size}, got T={t}')
        idx = torch.combinations(torch.arange(t, device=x.device), r=self.set_size)  # (P, set_size)
        return x[:, idx].reshape(b, idx.shape[0], -1)

    def keys_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''``(B, T, D)`` → keys and values, each ``(B, P, d_model)``.'''
        tuples = self._tuples(x)
        return self.norm_k(self.k_linear(tuples)), self.v_linear(tuples)

    @staticmethod
    def class_logits(
        query_k: torch.Tensor,
        query_v: torch.Tensor,
        support_k: torch.Tensor,
        support_v: torch.Tensor,
    ) -> torch.Tensor:
        '''Logits ``(n_query,)`` of every query against one class.

        ``query_*`` are ``(n_query, P, d)``; ``support_*`` are ``(K, P, d)`` —
        the K support videos of the class. Each query tuple's softmax runs over
        all ``K·P`` support tuples of the class.
        '''
        _, p, d = query_k.shape
        scores = torch.matmul(query_k, support_k.reshape(-1, d).transpose(0, 1)) / math.sqrt(d)  # (nq, P, K*P)
        attn = torch.softmax(scores, dim=-1)
        prototype = torch.matmul(attn, support_v.reshape(-1, d))  # (nq, P, d)
        with torch.autocast(device_type=query_v.device.type, enabled=False):
            diff = query_v.float() - prototype.float()
            return -(diff * diff).sum(dim=(-2, -1)) / p


class TRX(EpisodicModel):
    '''TRX head over a per-frame encoder (``(B, T, H, W, 3)`` → ``(B, T, D)``).

    Args:
        encoder: per-frame encoder exposing ``embed_dim`` (e.g.
            :class:`src.models.encoders.FrameEncoder`).
        encoder_batch_size: see :class:`src.models.base.EpisodicModel`.
        temporal_set_sizes: tuple cardinalities ``Ω`` (one branch each).
        d_model: key/value size.
        dropout: dropout after the positional encoding.
        pe_scale: amplitude of the positional encoding.
    '''

    needs_frame_features = True

    def __init__(
        self,
        encoder: nn.Module,
        encoder_batch_size: int | None = None,
        temporal_set_sizes: Sequence[int] = (2, 3),
        d_model: int = 1152,
        dropout: float = 0.1,
        pe_scale: float = 0.1,
    ) -> None:
        super().__init__(encoder, encoder_batch_size)
        in_dim = getattr(encoder, 'embed_dim', None)
        if in_dim is None:
            raise ValueError('TRX needs an encoder exposing embed_dim (per-frame feature size)')
        if not temporal_set_sizes:
            raise ValueError('temporal_set_sizes must name at least one tuple cardinality')
        self.transformers = nn.ModuleList(
            TemporalCrossTransformer(int(in_dim), int(d_model), int(s), float(dropout), float(pe_scale))
            for s in temporal_set_sizes
        )

    @staticmethod
    def _check_frames(emb: torch.Tensor) -> None:
        if emb.ndim != 3:
            raise ValueError(f'TRX needs per-frame (B, T, D) features; got shape {tuple(emb.shape)}')

    def support_state(
        self, support_emb: torch.Tensor, support_labels: torch.Tensor, n_way: int
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        '''Per branch, per class: the ``(K, P, d_model)`` keys and values of that class's support videos.'''
        self._check_frames(support_emb)
        masks = []
        for c in range(n_way):
            mask = support_labels == c
            if not bool(mask.any()):
                raise ValueError(f'class {c} has no support samples in this episode')
            masks.append(mask)
        state = []
        for branch in self.transformers:
            keys, values = branch.keys_values(support_emb)
            state.append([(keys[m], values[m]) for m in masks])
        return state

    def query_logits(
        self, query_emb: torch.Tensor, state: list[list[tuple[torch.Tensor, torch.Tensor]]]
    ) -> torch.Tensor:
        '''``(n_query, n_way)`` logits, averaged over the tuple cardinalities.'''
        self._check_frames(query_emb)
        per_branch = []
        for branch, classes in zip(self.transformers, state, strict=True):
            query_k, query_v = branch.keys_values(query_emb)
            per_branch.append(
                torch.stack([branch.class_logits(query_k, query_v, k, v) for k, v in classes], dim=1)
            )
        return torch.stack(per_branch).mean(dim=0)


__all__ = ['PositionalEncoding', 'TRX', 'TemporalCrossTransformer']
