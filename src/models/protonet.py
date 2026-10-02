'''Prototypical Networks (Snell et al. 2017) for few-shot action recognition.

Given a single episode with ``N`` classes, ``K`` support samples per class and
``Q`` query samples per class, the head:

1. encodes support and query (``encoder(...)`` → ``(N*(K+Q), D)``);
2. averages each class's K support embeddings → prototypes ``(N, D)``;
3. classifies each query by ``-‖q - c_i‖²`` as logits (negative squared L2);
4. cross-entropy against ``[0, N)`` labels.

Steps 2 and 3 are :meth:`ProtoNet.support_state` and
:meth:`ProtoNet.query_logits`; the episode plumbing lives in
:class:`src.models.base.EpisodicModel`. Clip-level ``(B, D)`` embeddings are
used as is; per-frame ``(B, T, D)`` ones (from
:class:`src.models.encoders.FrameEncoder`) are averaged over ``T`` first —
the ProtoNet baseline of the TRX paper.
'''
from __future__ import annotations

import torch

from src.models.base import EpisodicModel


class ProtoNet(EpisodicModel):
    '''ProtoNet head wrapping an arbitrary video encoder (clip-level or per-frame).'''

    @staticmethod
    def _pool(emb: torch.Tensor) -> torch.Tensor:
        '''``(B, D)`` as is; ``(B, T, D)`` averaged over frames.'''
        if emb.ndim == 3:
            return emb.mean(dim=1)
        if emb.ndim != 2:
            raise ValueError(f'encoder must return (B, D) or (B, T, D); got shape {tuple(emb.shape)}')
        return emb

    def support_state(self, support_emb: torch.Tensor, support_labels: torch.Tensor, n_way: int) -> torch.Tensor:
        '''Per-class mean of the support embeddings → prototypes ``(n_way, D)``.'''
        support_emb = self._pool(support_emb)
        embed_dim = support_emb.shape[1]
        prototypes = torch.zeros(n_way, embed_dim, device=support_emb.device, dtype=support_emb.dtype)
        for c in range(n_way):
            mask = support_labels == c
            if not bool(mask.any()):
                raise ValueError(f'class {c} has no support samples in this episode')
            prototypes[c] = support_emb[mask].mean(dim=0)
        return prototypes

    def query_logits(self, query_emb: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        '''Negative squared L2 distance ``(N*Q, N)``, forced to fp32.

        Under AMP the embeddings are fp16 and a sum of ``D`` squared differences
        can exceed fp16's 65504 ceiling → ``inf`` → ``NaN`` loss. Computing the
        distance with autocast disabled keeps the logits/softmax numerically safe
        at negligible (few-KB) memory cost.
        '''
        query_emb = self._pool(query_emb)
        with torch.autocast(device_type=query_emb.device.type, enabled=False):
            q = query_emb.float().unsqueeze(1)   # (N*Q, 1, D)
            p = prototypes.float().unsqueeze(0)  # (1, N, D)
            diffs = q - p
            return -(diffs * diffs).sum(dim=-1)  # (N*Q, N)


__all__ = ['ProtoNet']
