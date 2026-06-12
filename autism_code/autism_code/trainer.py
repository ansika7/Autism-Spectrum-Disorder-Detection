"""
trainer.py — Training Engine
==============================
Manages the complete supervised training loop:

  • Warm-up phase  : backbones frozen for first N epochs
                     (only projection + fusion + head are trained)
  • Fine-tune phase: all parameters unfrozen
  • Optimiser      : AdamW / Adam / SGD
  • Scheduler      : CosineAnnealing / StepLR / ReduceLROnPlateau
  • Loss           : FocalLoss or CrossEntropyLoss (see losses.py)
  • Gradient clip  : configurable max-norm
  • Early stopping : patience on validation accuracy
  • Checkpointing  : saves best and last checkpoints
  • TensorBoard    : loss / accuracy / F1 / AUC / LR per epoch
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)
from tqdm import tqdm

log = logging.getLogger(__name__)


# ── Metric Helpers ────────────────────────────────────────────────────────────

def _metrics(labels: np.ndarray,
             preds:  np.ndarray,
             probs:  np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}


# ── Optimiser / Scheduler Factories ──────────────────────────────────────────

def _build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    t   = cfg["training"]
    lr  = t["learning_rate"]
    wd  = t.get("weight_decay", 1e-4)
    opt = t.get("optimizer", "adamw").lower()
    params = filter(lambda p: p.requires_grad, model.parameters())

    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {opt!r}")


def _build_scheduler(cfg: dict, opt: torch.optim.Optimizer):
    t     = cfg["training"]
    sched = t.get("scheduler", "cosine").lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max   = t.get("scheduler_t_max", t["epochs"]),
            eta_min = t.get("scheduler_eta_min", 1e-6),
        )
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, verbose=True)
    raise ValueError(f"Unknown scheduler: {sched!r}")


# ── Single Epoch ──────────────────────────────────────────────────────────────

def _run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    clip:      float = 1.0,
    phase:     str   = "train",
) -> Tuple[float, Dict[str, float]]:

    is_train = (phase == "train")
    model.train(is_train)

    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, desc=f"  {phase:>5}", leave=False, unit="batch")

    with torch.set_grad_enabled(is_train):
        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            probs = torch.softmax(logits.detach(), dim=1)
            preds = probs.argmax(dim=1)

            running_loss += loss.item() * imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend( preds.cpu().numpy())
            all_probs.append( probs.cpu().numpy())   # use append not extend

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n   = max(len(all_labels), 1)
    avg = running_loss / n
    met = _metrics(
    np.array(all_labels),
    np.array(all_preds),
    np.vstack(all_probs),   # vstack keeps 2D shape (N, C)
)
    return avg, met


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training + validation loop.

    Usage
    -----
        trainer = Trainer(cfg, model, loaders, criterion, device)
        history = trainer.fit()
    """

    def __init__(
        self,
        cfg:       dict,
        model:     nn.Module,
        loaders:   Dict[str, DataLoader],
        criterion: nn.Module,
        device:    torch.device,
    ):
        self.cfg       = cfg
        self.model     = model
        self.loaders   = loaders
        self.criterion = criterion
        log.info(f"Criterion: {criterion.__class__.__name__}")
        self.device    = device

        t                    = cfg["training"]
        self.epochs          = t["epochs"]
        self.warmup_epochs   = t.get("warmup_epochs", 5)
        self.clip            = t.get("gradient_clip", 1.0)
        self.patience        = t.get("early_stopping_patience", 10)

        save_dir = Path(t.get("save_dir", "outputs/checkpoints"))
        log_dir  = Path(t.get("log_dir",  "outputs/logs"))
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Build optimizer/scheduler for the warm-up phase first
        self.optimizer = _build_optimizer(cfg, model)
        self.scheduler = _build_scheduler(cfg, self.optimizer)

        self.best_val_acc     = 0.0
        self.patience_counter = 0

        self.history: Dict[str, list] = {
            k: [] for k in [
                "train_loss", "val_loss",
                "train_acc",  "val_acc",
                "train_f1",   "val_f1",
                "train_auc",  "val_auc",
            ]
        }

    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, list]:
        log.info(
            f"Training {self.epochs} epochs on {self.device} │ "
            f"warm-up for first {self.warmup_epochs} epoch(s)"
        )

        for epoch in range(1, self.epochs + 1):

            # ── Warm-up / fine-tune transitions ─────────────────────
            if epoch == 1:
                self.model.freeze_backbones()

            if epoch == self.warmup_epochs + 1:
                self.model.unfreeze_all()
                # Rebuild with all params now trainable
                self.optimizer = _build_optimizer(self.cfg, self.model)
                self.scheduler = _build_scheduler(self.cfg, self.optimizer)
                log.info(f"Epoch {epoch}: backbones unfrozen — full fine-tuning")

            t0 = time.time()

            # ── Train ────────────────────────────────────────────────
            tr_loss, tr_met = _run_epoch(
                self.model, self.loaders["train"], self.criterion,
                self.optimizer, self.device, self.clip, "train",
            )

            # ── Validate ─────────────────────────────────────────────
            vl_loss, vl_met = _run_epoch(
                self.model, self.loaders["val"], self.criterion,
                None, self.device, phase="val",
            )

            # ── Scheduler step ───────────────────────────────────────
            if self.cfg["training"].get("scheduler", "cosine") == "plateau":
                self.scheduler.step(vl_met["accuracy"])
            else:
                self.scheduler.step()

            elapsed = time.time() - t0
            lr      = self.optimizer.param_groups[0]["lr"]

            self._update_history(tr_loss, tr_met, vl_loss, vl_met)
            self._log(epoch, tr_loss, tr_met, vl_loss, vl_met, elapsed, lr)
            self._tb(epoch, tr_loss, tr_met, vl_loss, vl_met, lr)

            # ── Checkpoint ───────────────────────────────────────────
            if vl_met["accuracy"] > self.best_val_acc:
                self.best_val_acc     = vl_met["accuracy"]
                self.patience_counter = 0
                self._save(epoch, vl_met, "best")
                log.info(f"  ✓ New best val_acc={self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1

            # ── Early stopping ───────────────────────────────────────
            if self.patience_counter >= self.patience:
                log.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        self._save(epoch, vl_met, "last")
        self.writer.close()

        # Save history JSON
        hist_path = Path("outputs") / "training_history.json"
        hist_path.parent.mkdir(exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        log.info(f"Training complete │ best val_acc={self.best_val_acc:.4f}")
        return self.history

    # ------------------------------------------------------------------
    def _log(self, ep, tl, tm, vl, vm, t, lr):
        log.info(
            f"Epoch {ep:>3}/{self.epochs}  "
            f"│ train loss={tl:.4f} acc={tm['accuracy']:.4f} f1={tm['f1']:.4f} "
            f"│ val   loss={vl:.4f} acc={vm['accuracy']:.4f} f1={vm['f1']:.4f} "
            f"auc={vm['auc']:.4f} "
            f"│ lr={lr:.2e} │ {t:.1f}s"
        )

    def _update_history(self, tl, tm, vl, vm):
        self.history["train_loss"].append(tl)
        self.history["val_loss"].append(vl)
        self.history["train_acc"].append(tm["accuracy"])
        self.history["val_acc"].append(vm["accuracy"])
        self.history["train_f1"].append(tm["f1"])
        self.history["val_f1"].append(vm["f1"])
        self.history["train_auc"].append(tm["auc"])
        self.history["val_auc"].append(vm["auc"])

    def _tb(self, ep, tl, tm, vl, vm, lr):
        self.writer.add_scalars("Loss",     {"train": tl, "val": vl}, ep)
        self.writer.add_scalars("Accuracy", {"train": tm["accuracy"], "val": vm["accuracy"]}, ep)
        self.writer.add_scalars("F1",       {"train": tm["f1"],       "val": vm["f1"]},       ep)
        self.writer.add_scalars("AUC",      {"train": tm["auc"],      "val": vm["auc"]},       ep)
        self.writer.add_scalar("LR", lr, ep)

    def _save(self, ep, metrics, tag):
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch":       ep,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics":     metrics,
            "config":      self.cfg,
        }, path)
        log.info(f"  Checkpoint → {path}")
