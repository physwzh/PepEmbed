"""Evaluation metrics.

- recall_at_1: within-batch top1 retrieval hit rate for contrastive learning
- fingerprint_per_bit_auc: Stage 2 fingerprint prediction per-bit AUC (ignoring constant bits)
- property_mae: Stage 2 property prediction MAE (normalized space)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def recall_at_1(student_embedding, teacher_embedding):
    student = F.normalize(student_embedding, dim=-1)
    teacher = F.normalize(teacher_embedding, dim=-1)
    sim = student @ teacher.t()
    pred = sim.argmax(dim=1)
    target = torch.arange(sim.shape[0], device=sim.device)
    return (pred == target).float().mean().item()


def fingerprint_per_bit_auc(fp_logits, fp_target, return_mean=True):
    """per-bit AUC. Constant bits (all 0 or all 1, AUC undefined) are ignored.

    fp_logits: [B, n_bits] logits
    fp_target: [B, n_bits] 0/1
    """
    from sklearn.metrics import roc_auc_score

    logits = fp_logits.detach().cpu().numpy()
    target = fp_target.detach().cpu().numpy()
    n_bits = logits.shape[1]
    aucs = []
    for b in range(n_bits):
        col = target[:, b]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue  # constant bit, skip
        try:
            aucs.append(roc_auc_score(col, logits[:, b]))
        except ValueError:
            continue
    if not aucs:
        return float("nan")
    mean = float(np.mean(aucs))
    if return_mean:
        return mean
    return aucs


def property_mae(prop_pred, prop_target):
    return float(torch.abs(prop_pred - prop_target).mean().item())
