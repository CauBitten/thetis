'''Episodic interface shared by every FSAR head (ProtoNet, TRX, ...).

A head scores an episode in two steps, so the trainer can stream the query set:

1. ``support_state(support_emb, support_labels, n_way)`` — everything that
   depends only on the support set (ProtoNet: class prototypes; TRX: the
   support frame features grouped by class);
2. ``query_logits(query_emb, state)`` — ``(n_query, n_way)`` logits for any
   slice of the query set. Each query is scored independently of the others,
   which is what makes the micro-batched backward of
   ``meta_trainer._train_step_streaming`` exact.

``forward`` composes the two over the whole episode and adds the
cross-entropy; subclasses implement only the two steps. The encoder is
decoupled: clip-level encoders return ``(B, D)``, per-frame encoders
``(B, T, D)`` (see :mod:`src.models.encoders`), and each head declares
which one it needs via :attr:`EpisodicModel.needs_frame_features`.
'''
from __future__ import annotations

from typing import Any, TypedDict

import torch
from torch import nn
from torch.nn import functional as F


class EpisodeOutput(TypedDict):
    logits: torch.Tensor   # (N*Q, N)
    loss: torch.Tensor     # scalar
    accuracy: float        # scalar in [0, 1]
    preds: torch.Tensor    # (N*Q,) int64


class EpisodicModel(nn.Module):
    '''Base class for few-shot heads wrapping an arbitrary video encoder.

    Args:
        encoder: ``nn.Module`` with ``forward(x: (B, T, H, W, 3)) -> (B, D)``
            or ``-> (B, T, D)``.
        encoder_batch_size: if set, :meth:`forward` encodes at most this many
            videos per encoder call (see :meth:`_encode`).

    Call signature:
        ``forward(support, query, support_labels, query_labels, n_way)`` where
        ``support`` is ``(N*K, T, H, W, 3)``, ``query`` is ``(N*Q, T, H, W, 3)``,
        and labels are int tensors in ``[0, N)``. Returns an
        :class:`EpisodeOutput` dict so callers can choose to backprop on
        ``loss`` and log ``accuracy``.
    '''

    #: True when the head needs per-frame ``(B, T, D)`` features (e.g. TRX);
    #: False when clip-level ``(B, D)`` features are enough.
    needs_frame_features: bool = False

    def __init__(self, encoder: nn.Module, encoder_batch_size: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_batch_size = encoder_batch_size

    def _encode(self, videos: torch.Tensor, encoder_batch_size: int | None = None) -> torch.Tensor:
        '''Encode ``(B, T, H, W, 3)`` videos, optionally in chunks.

        Chunking caps the *transient* forward footprint, but under autograd each
        chunk's activation graph is retained until backward — so this only bounds
        peak memory in ``no_grad`` (eval). The training loop streams support/query
        with gradient accumulation instead (see
        ``meta_trainer._train_step_streaming``).
        '''
        batch_size = encoder_batch_size if encoder_batch_size is not None else self.encoder_batch_size
        if batch_size is None or batch_size >= videos.shape[0]:
            return self.encoder(videos)
        chunks: list[torch.Tensor] = []
        for start in range(0, videos.shape[0], batch_size):
            chunks.append(self.encoder(videos[start : start + batch_size]))
        return torch.cat(chunks, dim=0)

    def support_state(self, support_emb: torch.Tensor, support_labels: torch.Tensor, n_way: int) -> Any:
        '''Everything the head precomputes from the support set alone.'''
        raise NotImplementedError

    def query_logits(self, query_emb: torch.Tensor, state: Any) -> torch.Tensor:
        '''``(n_query, n_way)`` logits for a slice of the query set, given ``support_state``.'''
        raise NotImplementedError

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: torch.Tensor,
        query_labels: torch.Tensor,
        n_way: int,
        encoder_batch_size: int | None = None,
    ) -> EpisodeOutput:
        if support.ndim != 5 or query.ndim != 5:
            raise ValueError(
                f'expected (B,T,H,W,3) tensors; got support={tuple(support.shape)}, query={tuple(query.shape)}'
            )
        n_support = support.shape[0]
        n_query = query.shape[0]
        if support_labels.shape != (n_support,):
            raise ValueError(f'support_labels shape {tuple(support_labels.shape)} != ({n_support},)')
        if query_labels.shape != (n_query,):
            raise ValueError(f'query_labels shape {tuple(query_labels.shape)} != ({n_query},)')

        combined = torch.cat([support, query], dim=0)
        embeddings = self._encode(combined, encoder_batch_size)

        state = self.support_state(embeddings[:n_support], support_labels, n_way)
        logits = self.query_logits(embeddings[n_support:], state)
        loss = F.cross_entropy(logits, query_labels)
        preds = logits.argmax(dim=-1)
        accuracy = float((preds == query_labels).float().mean().item())

        return {
            'logits': logits,
            'loss': loss,
            'accuracy': accuracy,
            'preds': preds,
        }


__all__ = ['EpisodeOutput', 'EpisodicModel']
