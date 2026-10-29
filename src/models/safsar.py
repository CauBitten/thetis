'''SAFSAR: Semantic-Aware Few-Shot Action Recognition (Tang, Béjar & Vidal, WACV 2024).

There is no official code release; this follows the paper (arXiv 2311.06218,
§3). Per episode, on clip-level VideoMAE features ``(B, C)``:

1. **Prototypes** (Eq. 2): mean support feature per class.
2. **MM Fusion** (Eq. 3): each prototype is prepended, as one extra token, to
   the frozen BERT token features of its class description, and a Fusion
   Transformer of ``fusion_layers`` layers runs over them; the output at the
   prototype's slot is the semantic-aware prototype ``f*_c``. Queries skip it.
3. **TLM** (Eq. 4): one Transformer layer over ``[f*_1, …, f*_N, f_q]``.
4. **Distance** (Eq. 5-6): the logits are the cosine similarities between
   the query's and each class's TLM outputs (no temperature, as in the paper).
5. **Loss** (Eq. 7-9): episode cross-entropy plus ``global_loss_weight`` × a
   global linear classifier over the meta-train classes, applied to the
   VideoMAE features of support and query videos (:meth:`SAFSAR.auxiliary_loss`).

Choices where the paper gives no detail (also listed in
``experiments/configs/README.md``):

- the paper's episodes have one query; here each query runs the TLM with the N
  prototypes on its own, which keeps queries independent (exact streaming);
- Transformer layers: d = 768, 12 heads, FFN 3072, dropout 0.1, post-norm;
- ``global_loss_weight`` (λ) = 1;
- class descriptions written for THETIS (:data:`CLASS_DESCRIPTIONS`), as the
  paper did with ChatGPT for EPIC-KITCHENS.
'''
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.base import EpisodicModel

#: canonical THETIS label → the description BERT encodes.
CLASS_DESCRIPTIONS: dict[str, str] = {
    'backhand': 'a one-handed tennis backhand, a groundstroke hit on the non-dominant side by swinging the racket '
                'across the body',
    'backhand2hands': 'a two-handed tennis backhand, a groundstroke hit on the non-dominant side holding the racket '
                      'with both hands',
    'backhand_slice': 'a tennis backhand slice, a backhand hit with an open racket face moving from high to low that '
                      'puts backspin on the ball',
    'backhand_volley': 'a tennis backhand volley, a short punching backhand near the net that hits the ball before it '
                       'bounces',
    'flat_service': 'a flat tennis serve, hit overhead with little spin, striking the ball flat for a fast and straight '
                    'trajectory',
    'forehand_flat': 'a flat tennis forehand, a groundstroke on the dominant side with a square racket face and a '
                     'frontal impact for a fast, flat ball',
    'forehand_openstands': 'an open stance tennis forehand, a forehand groundstroke hit with the feet parallel to the '
                           'baseline and the hips rotating open',
    'forehand_slice': 'a tennis forehand slice, a forehand hit with an open racket face moving from high to low that '
                      'puts backspin on the ball',
    'forehand_volley': 'a tennis forehand volley, a short punching forehand near the net that hits the ball before it '
                       'bounces',
    'kick_service': 'a kick tennis serve, hit overhead brushing up the back of the ball with heavy topspin so it '
                    'bounces high',
    'slice_service': 'a slice tennis serve, hit overhead brushing the side of the ball with sidespin so it curves away',
    'smash': 'a tennis smash, an overhead stroke hitting a high ball above the head with a powerful downward motion',
}


class SAFSAR(EpisodicModel):
    '''SAFSAR head over a clip-level VideoMAE encoder.

    Args:
        encoder: clip-level encoder exposing ``embed_dim`` and
            ``encode_text_tokens`` (:class:`src.models.encoders.VideoMAEEncoder`).
        encoder_batch_size: see :class:`src.models.base.EpisodicModel`.
        fusion_layers: ℓ, layers of the Fusion Transformer.
        heads, mlp_dim, dropout: every Transformer layer (fusion and TLM).
        train_classes: the meta-train classes, rows of the global classifier;
            the trainer checks them against the split. ``None`` disables L2.
        global_loss_weight: λ of Eq. 9.
        class_descriptions: overrides/extends :data:`CLASS_DESCRIPTIONS`. They
            are encoded once; tokens and padding mask are buffers saved with
            the model, so evaluation does not need BERT.
    '''

    def __init__(
        self,
        encoder: nn.Module,
        encoder_batch_size: int | None = None,
        fusion_layers: int = 2,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        train_classes: Sequence[str] | None = None,
        global_loss_weight: float = 1.0,
        class_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(encoder, encoder_batch_size)
        dim = getattr(encoder, 'embed_dim', None)
        encode_text_tokens = getattr(encoder, 'encode_text_tokens', None)
        if dim is None or encode_text_tokens is None:
            raise ValueError('SAFSAR needs a VideoMAE encoder with embed_dim and encode_text_tokens '
                             '(encoder.name: videomae_base)')

        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(int(dim), int(heads), int(mlp_dim), float(dropout), batch_first=True)

        self.fusion = nn.TransformerEncoder(layer(), int(fusion_layers), enable_nested_tensor=False)
        self.tlm = layer()
        self.train_classes = list(train_classes) if train_classes else None
        self._train_index = {name: i for i, name in enumerate(self.train_classes or [])}
        self.global_head = nn.Linear(int(dim), len(self.train_classes)) if self.train_classes else None
        self.global_loss_weight = float(global_loss_weight)

        texts = {**CLASS_DESCRIPTIONS, **(class_descriptions or {})}
        names = sorted(texts)
        self.class_index = {name: i for i, name in enumerate(names)}
        tokens, padding = encode_text_tokens([texts[name] for name in names])
        self.register_buffer('text_tokens', tokens.float())
        self.register_buffer('text_padding', padding)

    @staticmethod
    def _check_clip(emb: torch.Tensor) -> None:
        if emb.ndim != 2:
            raise ValueError(f'SAFSAR needs clip-level (B, D) features; got shape {tuple(emb.shape)}')

    def support_state(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        class_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        '''Semantic-aware prototypes ``f*_c`` ``(n_way, C)``: class mean (Eq. 2) fused with its description (Eq. 3).'''
        self._check_clip(support_emb)
        if class_names is None or len(class_names) < n_way:
            raise ValueError('SAFSAR needs the class_names of the episode (the support text)')
        missing = [c for c in class_names[:n_way] if c not in self.class_index]
        if missing:
            raise ValueError(f'no class description for {missing}; add them to model.class_descriptions')
        prototypes = []
        for c in range(n_way):
            mask = support_labels == c
            if not bool(mask.any()):
                raise ValueError(f'class {c} has no support samples in this episode')
            prototypes.append(support_emb[mask].mean(dim=0))
        prototypes = torch.stack(prototypes)
        idx = [self.class_index[c] for c in class_names[:n_way]]
        tokens = torch.cat([prototypes.unsqueeze(1), self.text_tokens[idx].to(prototypes.dtype)], dim=1)
        padding = torch.cat([torch.zeros_like(self.text_padding[idx][:, :1]), self.text_padding[idx]], dim=1)
        return self.fusion(tokens, src_key_padding_mask=padding)[:, 0]

    def query_logits(self, query_emb: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        '''``(n_query, n_way)`` cosine logits; each query runs the TLM with all ``f*_c`` on its own (Eq. 4-6).'''
        self._check_clip(query_emb)
        n_query = query_emb.shape[0]
        tokens = torch.cat([fused.unsqueeze(0).expand(n_query, -1, -1), query_emb.unsqueeze(1).to(fused.dtype)], dim=1)
        out = self.tlm(tokens)
        with torch.autocast(device_type=query_emb.device.type, enabled=False):
            return F.cosine_similarity(out[:, :-1].float(), out[:, -1:].float(), dim=-1)

    def auxiliary_loss(
        self, emb: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str] | None
    ) -> torch.Tensor | None:
        '''λ·L2 (Eq. 8): global classifier on the VideoMAE features; ``None`` for episodes outside meta-train.'''
        if self.global_head is None or class_names is None or any(c not in self._train_index for c in class_names):
            return None
        targets = torch.tensor([self._train_index[c] for c in class_names], device=labels.device)[labels]
        return self.global_loss_weight * F.cross_entropy(self.global_head(emb), targets)


__all__ = ['CLASS_DESCRIPTIONS', 'SAFSAR']
