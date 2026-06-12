"""
main.py — Post-Augmentation Pipeline Orchestrator
===================================================
Runs the pipeline from the point where augmented data is ready:

  Stage 5  ─ Hybrid CNN-ViT model (models/hybrid_model.py)
  Stage 6  ─ Training              (trainer.py)
  Stage 7a ─ Explainability        (explainability.py)
  Stage 7b ─ Evaluation & metrics  (evaluation.py)

Usage examples
──────────────
  # Full run (train → evaluate → explain)
  python main.py

  # Specify a config file
  python main.py --config config.yaml

  # Evaluate only (no training) from a checkpoint
  python main.py --eval-only --checkpoint outputs/checkpoints/checkpoint_best.pt

  # Generate explanations from a saved checkpoint
  python main.py --explain-only --checkpoint outputs/checkpoints/checkpoint_best.pt

  # Train then evaluate, skip explainability
  python main.py --no-explain

  # Use a custom augmented-data directory
  python main.py --data-dir /path/to/augmented
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# ── Ensure outputs/ exists before log file is created ────────────────────────
Path("outputs").mkdir(exist_ok=True)

import io
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")),
        logging.FileHandler("outputs/pipeline.log", mode="a", encoding="utf-8"),
    ],
)

log = logging.getLogger("main")


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    d = cfg["training"].get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(d)
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        log.info("Running on CPU")
    return device


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, cfg, path: Path):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer else None,
        "metrics":     metrics,
        "config":      cfg,
    }, path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"│ metrics: {ckpt.get('metrics', {})}"
    )
    return ckpt


# ══════════════════════════════════════════════════════════════════
#  Stage functions
# ══════════════════════════════════════════════════════════════════

def stage_dataloaders(cfg: dict):
    log.info(">>  STAGE 4 (input)  | Loading augmented DataLoaders")
    from dataset import build_dataloaders
    return build_dataloaders(cfg)


def stage_model(cfg: dict, device: torch.device):
    log.info(">>  STAGE 5          | Building Hybrid CNN-ViT Model")
    from models import build_model
    return build_model(cfg, device)


def stage_train(cfg: dict, model, loaders, device: torch.device):
    log.info(">>  STAGE 6          | Training")
    from models import build_criterion
    from trainer import Trainer

    cw        = loaders["train"].dataset.class_weights().to(device)
    criterion = build_criterion(cfg, cw)
    trainer   = Trainer(cfg, model, loaders, criterion, device)
    return trainer.fit()


def stage_evaluate(cfg: dict, model, loaders, device: torch.device, history=None):
    log.info(">>  STAGE 7b         | Evaluation")
    from evaluation import run_evaluation
    return run_evaluation(model, loaders["test"], cfg, device, history)


def stage_explain(cfg: dict, model, loaders, device: torch.device):
    log.info(">>  STAGE 7a         | Explainability (Grad-CAM + Attention Rollout)")
    from explainability import explain_test_samples
    n = cfg.get("explainability", {}).get("num_samples", 16)
    explain_test_samples(model, loaders["test"], cfg, device, n=n)


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ASD Detection — Post-Augmentation Pipeline"
    )
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--data-dir",    default=None,
                        help="Override dataset.augmented_dir in config")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training; evaluate a saved checkpoint")
    parser.add_argument("--explain-only", action="store_true",
                        help="Skip training; run explainability on a checkpoint")
    parser.add_argument("--no-explain",  action="store_true",
                        help="Skip explainability stage")
    parser.add_argument("--checkpoint",  default=None,
                        help="Path to .pt checkpoint for --eval-only / --explain-only")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["dataset"]["augmented_dir"] = args.data_dir

    device = get_device(cfg)

    # ─────────────────────────────────────────────────────────────
    #  Mode: eval-only
    # ─────────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.checkpoint:
            log.error("--checkpoint required for --eval-only")
            sys.exit(1)
        loaders = stage_dataloaders(cfg)
        model   = stage_model(cfg, device)
        load_checkpoint(model, args.checkpoint, device)
        stage_evaluate(cfg, model, loaders, device)
        return

    # ─────────────────────────────────────────────────────────────
    #  Mode: explain-only
    # ─────────────────────────────────────────────────────────────
    if args.explain_only:
        if not args.checkpoint:
            log.error("--checkpoint required for --explain-only")
            sys.exit(1)
        loaders = stage_dataloaders(cfg)
        model   = stage_model(cfg, device)
        load_checkpoint(model, args.checkpoint, device)
        stage_explain(cfg, model, loaders, device)
        return

    # ─────────────────────────────────────────────────────────────
    #  Full pipeline
    # ─────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  ASD Detection Pipeline - Post-Augmentation")
    log.info("=" * 55)
    loaders = stage_dataloaders(cfg)
    model   = stage_model(cfg, device)
    history = stage_train(cfg, model, loaders, device)

    # Load best checkpoint for evaluation
    best_ckpt = Path(cfg["training"]["save_dir"]) / "checkpoint_best.pt"
    if best_ckpt.exists():
        load_checkpoint(model, str(best_ckpt), device)
    else:
        log.warning("Best checkpoint not found — using model in current state")

    stage_evaluate(cfg, model, loaders, device, history)

    if not args.no_explain:
        stage_explain(cfg, model, loaders, device)

    log.info("=" * 55)
    log.info("  Pipeline complete. Results in outputs/")
    log.info("=" * 55)

if __name__ == "__main__":
    main()
