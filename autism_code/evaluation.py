"""
evaluation.py — Comprehensive Test-Set Evaluation
===================================================
Produces everything needed to assess and report model performance:

Metrics
  Accuracy · F1 (binary) · Precision · Recall / Sensitivity
  Specificity · AUC-ROC · Matthews Correlation Coefficient

Plots (saved to outputs/evaluation/)
  confusion_matrix.png   — normalised heatmap with raw counts
  roc_curve.png          — ROC curve with AUC annotation
  pr_curve.png           — Precision-Recall curve
  training_history.png   — loss / accuracy / AUC across epochs

Text files
  classification_report.txt
  metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
)

log = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
_BG   = "#0d0d12"
_C1   = "#4fc3f7"   # cyan
_C2   = "#f4a261"   # amber
_C3   = "#a8edab"   # green
_GRID = "#1e1e2a"


# ════════════════════════════════════════════════════════════════
#  Inference pass
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pass over the DataLoader.

    Returns
    -------
    labels : (N,)    int   — ground-truth class indices
    preds  : (N,)    int   — predicted class indices
    probs  : (N, C)  float — softmax probabilities
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  Evaluating", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return (
        np.array(all_labels, dtype=int),
        np.array(all_preds,  dtype=int),
        np.array(all_probs,  dtype=np.float32),
    )


# ════════════════════════════════════════════════════════════════
#  Metric computation
# ════════════════════════════════════════════════════════════════

def compute_metrics(
    labels:      np.ndarray,
    preds:       np.ndarray,
    probs:       np.ndarray,
    class_names: List[str],
) -> Tuple[Dict[str, float], str]:

    acc = accuracy_score(labels, preds)
    f1        = f1_score(       labels, preds, average="binary", zero_division=0)
    prec      = precision_score(labels, preds, average="binary", zero_division=0)
    rec       = recall_score(   labels, preds, average="binary", zero_division=0)
    mcc       = matthews_corrcoef(labels, preds)

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(labels, probs[:, 1])
    except ValueError:
        ap = float("nan")

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity  = tp / max(tp + fn, 1)
        specificity  = tn / max(tn + fp, 1)
        ppv          = tp / max(tp + fp, 1)
        npv          = tn / max(tn + fn, 1)
    else:
        sensitivity = specificity = ppv = npv = float("nan")

    metrics = {
        "accuracy":      acc,
        "f1":            f1,
        "precision":     prec,
        "recall":        rec,
        "sensitivity":   sensitivity,
        "specificity":   specificity,
        "ppv":           ppv,
        "npv":           npv,
        "mcc":           mcc,
        "auc_roc":       auc,
        "avg_precision": ap,
    }

    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    log.info("\n== Test-Set Metrics ==")
    for k, v in metrics.items():
        log.info(f"  {k:>18s} : {v:.4f}")
    log.info("\n" + report)

    return metrics, report


# ════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════

def _ax_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(_BG)
    ax.set_title(title,   color="white",  fontsize=10)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for s in ax.spines.values():
        s.set_color(_GRID)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.5)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.5)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: List[str],
    save_path:   Path,
):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=_BG)
    ax.set_facecolor(_BG)

    sns.heatmap(
        cm_norm,
        annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues",
        linewidths=0.6, linecolor=_GRID,
        ax=ax,
        cbar_kws={"shrink": 0.78},
        annot_kws={"color": "white", "fontsize": 13, "fontweight": "bold"},
    )

    # Overlay raw counts in smaller text
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(
                c + 0.5, r + 0.73, f"n = {cm[r, c]}",
                ha="center", va="center", fontsize=8, color="#cccccc",
            )

    ax.set_title("Confusion Matrix (row-normalised)", color="white", pad=10, fontsize=10)
    ax.set_xlabel("Predicted",  color="#aaaaaa", fontsize=8)
    ax.set_ylabel("True Label", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close()
    log.info(f"  confusion_matrix.png → {save_path}")


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(labels, probs, save_path):
    try:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc         = roc_auc_score(labels, probs[:, 1])
    except ValueError as e:
        log.warning(f"ROC curve skipped: {e}")
        return

    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=_BG)
    _ax_style(ax, "ROC Curve", "False Positive Rate", "True Positive Rate")
    ax.plot(fpr, tpr, color=_C1, lw=2.2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color=_C2, lw=1.2, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=_C1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(facecolor="#13131c", edgecolor=_GRID, labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close()
    log.info(f"  roc_curve.png -> {save_path}")
# ── Precision-Recall curve ────────────────────────────────────────────────────

def plot_pr_curve(labels, probs, save_path):
    try:
        prec, rec, _ = precision_recall_curve(labels, probs[:, 1])
        ap            = average_precision_score(labels, probs[:, 1])
    except ValueError as e:
        log.warning(f"PR curve skipped: {e}")
        return

    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=_BG)
    _ax_style(ax, "Precision-Recall Curve", "Recall", "Precision")
    ax.plot(rec, prec, color=_C3, lw=2.2, label=f"AP = {ap:.4f}")
    ax.fill_between(rec, prec, alpha=0.08, color=_C3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(facecolor="#13131c", edgecolor=_GRID, labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close()
    log.info(f"  pr_curve.png -> {save_path}")

# ── Training history ──────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=_BG)
    fig.suptitle("Training History", color="white", fontsize=13, y=1.02)

    triples = [
        ("Loss",     "train_loss", "val_loss"),
        ("Accuracy", "train_acc",  "val_acc"),
        ("AUC-ROC",  "train_auc",  "val_auc"),
    ]

    for ax, (title, tr_k, vl_k) in zip(axes, triples):
        _ax_style(ax, title, "Epoch", title)
        ax.plot(epochs, history[tr_k], color=_C1, lw=2, label="Train")
        ax.plot(epochs, history[vl_k], color=_C2, lw=2, label="Val")
        ax.legend(facecolor="#13131c", edgecolor=_GRID, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close()
    log.info(f"  training_history.png → {save_path}")


# ════════════════════════════════════════════════════════════════
#  Full evaluation pipeline
# ════════════════════════════════════════════════════════════════

def run_evaluation(
    model:   nn.Module,
    loader:  DataLoader,
    cfg:     dict,
    device:  torch.device,
    history: Optional[dict] = None,
) -> Dict[str, float]:
    """
    End-to-end test-set evaluation.

    Runs inference, computes all metrics, saves plots and JSON.
    Returns the metrics dict.
    """
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = cfg["dataset"].get("classes", ["autism", "control"])

    log.info("Running inference on test set …")
    labels, preds, probs = run_inference(model, loader, device)

    metrics, report = compute_metrics(labels, preds, probs, class_names)

    # ── Save text reports ──
    (out_dir / "classification_report.txt").write_text(report)
    safe_metrics = {
        k: (round(float(v), 6) if not np.isnan(v) else None)
        for k, v in metrics.items()
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(safe_metrics, f, indent=2)

    # ── Plots ──
    plot_confusion_matrix(labels, preds, class_names,
                          out_dir / "confusion_matrix.png")
    plot_roc_curve(  labels, probs, out_dir / "roc_curve.png")
    plot_pr_curve(   labels, probs, out_dir / "pr_curve.png")

    if history:
        plot_training_history(history, out_dir / "training_history.png")

    log.info(f"All evaluation artefacts saved → {out_dir}")
    return metrics
