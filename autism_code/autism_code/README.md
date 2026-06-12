# ASD Detection — Post-Augmentation Pipeline
## Stages 5 → 6 → 7a → 7b

This package covers everything **after** data augmentation is complete.  
It expects your augmented face-crop images already on disk and handles:

```
Stage 5  ─ Hybrid CNN + Vision Transformer model
Stage 6  ─ Supervised training (warm-up → full fine-tune)
Stage 7a ─ Dual explainability  (Grad-CAM + Attention Rollout)
Stage 7b ─ Evaluation & metrics (Accuracy, F1, AUC, MCC, plots)
```

---

## Directory Layout

```
asd_post_aug/
├── config.yaml           ← all hyperparameters
├── main.py               ← pipeline entry point
├── predict.py            ← single-image inference
├── dataset.py            ← DataLoader from augmented images
├── trainer.py            ← training engine
├── explainability.py     ← Grad-CAM + Attention Rollout
├── evaluation.py         ← metrics + plots
├── requirements.txt
└── models/
    ├── __init__.py
    ├── hybrid_model.py   ← CNN + ViT + CrossAttentionFusion
    └── losses.py         ← Focal Loss
```

Expected data layout (set `dataset.augmented_dir` in config):

```
data/augmented/
    autism/
        *.jpg
    control/
        *.jpg
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Full run
python main.py --config config.yaml

# Skip explainability
python main.py --no-explain

# Evaluate a checkpoint
python main.py --eval-only \
    --checkpoint outputs/checkpoints/checkpoint_best.pt

# Explain with a checkpoint
python main.py --explain-only \
    --checkpoint outputs/checkpoints/checkpoint_best.pt

# Single-image prediction
python predict.py \
    --image    path/to/face.jpg \
    --checkpoint outputs/checkpoints/checkpoint_best.pt
```

---

## Model Architecture

### Hybrid CNN + Vision Transformer

```
Input (B, 3, 224, 224)
    │
    ├──────────────────────┐
    ▼                      ▼
CNN Branch             ViT Branch
ResNet-50              ViT-Small/16
(local features)       (global attention)
    │                      │
 proj→LayerNorm→GELU    proj→LayerNorm→GELU
 (B, 512)               (B, 512)
    │                      │
    └──────────┬────────────┘
               ▼
    Cross-Attention Fusion  ← NOVEL
    CNN queries ViT space
    ViT queries CNN space
    cat → Linear → LayerNorm → GELU
         (B, 256)
               ▼
         Dropout + Linear
          (B, 2)  logits
```

### Cross-Attention Fusion
The core novelty: rather than naïve concatenation or addition, each branch
selectively queries the other's feature space via multi-head attention before
the final classification decision.

---

## Outputs

After a full run:

```
outputs/
├── checkpoints/
│   ├── checkpoint_best.pt     ← highest val accuracy
│   └── checkpoint_last.pt
├── logs/                      ← TensorBoard events
├── training_history.json
├── evaluation/
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── training_history.png
└── explanations/
    ├── explanation_0.png      ← Original | Grad-CAM | Attention Rollout
    ├── explanation_1.png
    └── ...
```

Launch TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.augmented_dir` | `data/augmented` | Root of augmented images |
| `model.cnn_backbone` | `resnet50` | `resnet50` / `resnet34` / `efficientnet_b3` / `mobilenet_v3_large` |
| `model.vit_backbone` | `vit_small_patch16_224` | Any timm ViT model name |
| `model.fusion_type` | `attention` | `attention` / `concat` / `add` |
| `training.epochs` | `50` | Max training epochs |
| `training.warmup_epochs` | `5` | Epochs with backbones frozen |
| `training.focal_loss_gamma` | `2.0` | 0 → CrossEntropy |
| `explainability.num_samples` | `16` | Samples to explain from test set |
