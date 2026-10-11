"""Contrastive loss: InfoNCE + optional memory queue.

  L = -log( exp(sim(student, teacher)/τ) / Σ_j exp(sim(student, teacher_j)/τ) )

The queue supplements negatives (plan §12: use a memory bank when the batch is
not large enough).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ContrastiveQueue:
    """FIFO queue of teacher features, used as extra negatives."""
    def __init__(self, feature_dim: int, queue_size: int, device: torch.device):
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.device = device
        self.buffer = None
        if queue_size > 0:
            self.buffer = torch.empty((0, feature_dim), dtype=torch.float32, device=device)

    def get(self):
        return self.buffer if (self.buffer is not None and self.buffer.numel() > 0) else None

    def update(self, features: torch.Tensor):
        if self.queue_size <= 0:
            return
        feats = F.normalize(features.detach().to(self.device), dim=-1)
        if self.buffer.numel() == 0:
            self.buffer = feats[-self.queue_size:]
        else:
            self.buffer = torch.cat([self.buffer, feats], dim=0)[-self.queue_size:]


def info_nce_loss(student_embedding, teacher_embedding, temperature: float, queue_features=None):
    """student retrieves teacher; the positive pair shares the same index.
    Returns (loss, logits, targets)."""
    student = F.normalize(student_embedding, dim=-1)
    teacher = F.normalize(teacher_embedding, dim=-1)

    logits = student @ teacher.t()
    targets = torch.arange(student.shape[0], device=student.device)

    if queue_features is not None and queue_features.numel() > 0:
        queue_logits = student @ queue_features.t()
        logits = torch.cat([logits, queue_logits], dim=1)

    logits = logits / temperature
    loss = F.cross_entropy(logits, targets)
    return loss, logits, targets
