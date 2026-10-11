"""Shared training loop.

Wraps optimizer/scheduler construction, the single-epoch train/val loop, ckpt
saving, and early stopping. The three stage entry points pass different
loss_fn / metric_fn and reuse the same skeleton.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .utils import setup_logger, save_checkpoint, save_metrics_line


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_optimizer(model, config, total_steps):
    """Two learning-rate groups: one for the backbone (including pooling), one for the heads."""
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.") or name.startswith("pooling."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = [{"params": backbone_params, "lr": config["lr_backbone"]}]
    if head_params:
        groups.append({"params": head_params, "lr": config["lr_head"]})
    optimizer = AdamW(groups, weight_decay=config.get("weight_decay", 0.01))

    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.0))

    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_epoch(
    model, loader, optimizer, scheduler, device, config,
    loss_fn: Callable,           # (model, batch) -> (loss, metrics_dict, do_backward=True)
    logger, train: bool, queue=None, queue_update_fn: Optional[Callable] = None,
):
    model.train() if train else model.eval()
    total_loss = 0.0
    metric_accum = {}
    total_steps = 0
    max_steps = config["max_steps_per_epoch"] if train else config["max_val_steps"]
    log_every = config.get("log_every", 100)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, start=1):
            if max_steps is not None and step > max_steps:
                break
            batch = move_batch_to_device(batch, device)
            loss, metrics, _ = loss_fn(model, batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = config.get("gradient_clip_norm")
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                if queue is not None and queue_update_fn is not None:
                    queue_update_fn(queue, batch)
            total_loss += loss.item()
            for k, v in metrics.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + v
            total_steps += 1
            if train and step % log_every == 0:
                logger.info(f"train step={step} loss={loss.item():.4f} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    denom = max(1, total_steps)
    result = {"loss": total_loss / denom, "steps": total_steps}
    for k, v in metric_accum.items():
        result[k] = v / denom
    return result


def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, device, config, training_cfg,
    loss_fn, logger, output_dir, metrics_path,
    queue=None, queue_update_fn=None,
    val_extra_metrics_fn: Optional[Callable] = None,
):
    """Generic training loop + early stopping + ckpt saving.

    val_extra_metrics_fn(model, device): optional, extra metrics computed and
    printed each epoch (e.g. Stage1b's chem monitoring).
    """
    best_val_loss = float("inf")
    patience = 0
    epochs = training_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, device, training_cfg,
                                  loss_fn, logger, train=True, queue=queue, queue_update_fn=queue_update_fn)
        val_metrics = run_epoch(model, val_loader, optimizer, scheduler, device, training_cfg,
                                loss_fn, logger, train=False)

        merged = {"epoch": epoch,
                  "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
                  "train_steps": train_metrics["steps"], "val_steps": val_metrics["steps"]}
        for k in train_metrics:
            if k not in ("loss", "steps"):
                merged[f"train_{k}"] = train_metrics[k]
        for k in val_metrics:
            if k not in ("loss", "steps"):
                merged[f"val_{k}"] = val_metrics[k]

        if val_extra_metrics_fn is not None:
            extra = val_extra_metrics_fn(model, device)
            merged.update(extra)

        save_metrics_line(metrics_path, merged)
        msg = f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}"
        for k in val_metrics:
            if k not in ("loss", "steps"):
                msg += f" val_{k}={val_metrics[k]:.4f}"
        if val_extra_metrics_fn is not None:
            for ek, ev in (extra or {}).items():
                msg += f" {ek}={ev:.4f}" if isinstance(ev, float) else f" {ek}={ev}"
        logger.info(msg)

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, merged)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, merged)
        else:
            patience += 1
            if patience >= training_cfg.get("early_stopping_patience", 3):
                logger.info("early stopping triggered")
                break

    logger.info(f"finished training, best_val_loss={best_val_loss:.4f}")
    return best_val_loss
