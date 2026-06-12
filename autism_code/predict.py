"""
predict.py — Single-Image Inference
=====================================
Loads a saved checkpoint and runs inference on a single image,
printing class probabilities and generating an explanation figure.

Usage:
    python predict.py \
        --image  path/to/face.jpg \
        --checkpoint  outputs/checkpoints/checkpoint_best.pt \
        --config  config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict")


def predict(image_path: str, checkpoint_path: str, config_path: str):
    # ── Config & device ──
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    d = cfg["training"].get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(d)

    # ── Transform ──
    from dataset import build_inference_transform
    tfm = build_inference_transform(cfg)

    # ── Load & transform image ──
    pil_img = Image.open(image_path).convert("RGB")
    tensor  = tfm(pil_img).unsqueeze(0).to(device)   # (1,3,H,W)

    # ── Model ──
    from models import build_model
    model = build_model(cfg, device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Inference ──
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_names = cfg["dataset"].get("classes", ["autism", "control"])
    pred_idx    = int(probs.argmax())
    pred_label  = class_names[pred_idx]
    confidence  = probs[pred_idx] * 100

    print("\n─────────────────────────────────────────")
    print(f"  Image     : {image_path}")
    print(f"  Prediction: {pred_label.upper()}  ({confidence:.2f}% confident)")
    for c, p in zip(class_names, probs):
        bar = "█" * int(p * 30)
        print(f"  {c:>10s}  {p*100:5.2f}%  {bar}")
    print("─────────────────────────────────────────\n")

    # ── Explanation ──
    from explainability import ExplainabilityEngine
    engine  = ExplainabilityEngine(model, cfg, device)
    stem    = Path(image_path).stem
    engine.explain_batch(
        images  = tensor.cpu(),
        labels  = torch.tensor([pred_idx]),
        indices = [stem],
    )
    out_dir = cfg.get("explainability", {}).get("output_dir", "outputs/explanations")
    log.info(f"Explanation saved → {out_dir}/explanation_{stem}.png")

    return pred_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="config.yaml")
    args = parser.parse_args()

    predict(args.image, args.checkpoint, args.config)
