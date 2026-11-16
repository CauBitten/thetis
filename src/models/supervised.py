'''Supervised baseline, the "Baseline" of Chen et al. (ICLR 2019): no episodes in training.

- **Training** (``meta_trainer`` with ``method: supervised``): encoder + a linear
  :attr:`SupervisedBaseline.classifier` over the meta-train classes, plain
  cross-entropy on shuffled mini-batches of clips.
- **Test** (the episodic interface, so ``eval_episodic`` works unchanged): the
  encoder is frozen and a *new* linear classifier is fit on each episode's K
  support embeddings, with the authors' fine-tuning settings (100 epochs,
  batch 4, SGD lr 0.01, momentum 0.9, dampening 0.9, weight decay 1e-3); the
  training classifier is discarded.
'''
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.base import EpisodicModel


class SupervisedBaseline(EpisodicModel):
    '''Clip-level encoder + meta-train classifier; per-episode linear probe at test time.'''

    def __init__(
        self,
        encoder: nn.Module,
        encoder_batch_size: int | None = None,
        train_classes: Sequence[str] = (),
        finetune_epochs: int = 100,
        finetune_batch_size: int = 4,
        finetune_lr: float = 0.01,
    ) -> None:
        super().__init__(encoder, encoder_batch_size)
        if getattr(encoder, 'per_frame', False):
            raise ValueError('supervised needs a clip-level encoder (encoder.name: r2plus1d_18)')
        if not train_classes:
            raise ValueError('supervised needs model.train_classes (the meta_train split)')
        self.train_classes = list(train_classes)
        self.classifier = nn.Linear(int(encoder.embed_dim), len(self.train_classes))
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_batch_size = int(finetune_batch_size)
        self.finetune_lr = float(finetune_lr)

    def support_state(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        class_names: Sequence[str] | None = None,
    ) -> nn.Linear:
        '''Fit a fresh ``n_way`` linear classifier on the (detached) support embeddings.'''
        # Evaluation runs under no_grad and fp16 autocast; the probe needs both off.
        with torch.enable_grad(), torch.autocast(device_type=support_emb.device.type, enabled=False):
            x = support_emb.detach().float()
            head = nn.Linear(x.shape[1], n_way).to(x.device)
            opt = torch.optim.SGD(
                head.parameters(), lr=self.finetune_lr, momentum=0.9, dampening=0.9, weight_decay=1e-3
            )
            for _ in range(self.finetune_epochs):
                for idx in torch.randperm(x.shape[0], device=x.device).split(self.finetune_batch_size):
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(head(x[idx]), support_labels[idx]).backward()
                    opt.step()
        return head

    def query_logits(self, query_emb: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        with torch.autocast(device_type=query_emb.device.type, enabled=False):
            return head(query_emb.float())


__all__ = ['SupervisedBaseline']
