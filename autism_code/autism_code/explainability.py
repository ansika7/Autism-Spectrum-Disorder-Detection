"""
explainability.py — Dual Explainability Framework
===================================================
Implements the two complementary explainability mechanisms:

  1. Grad-CAM (CNN branch)
     ─────────────────────
     Computes the gradient of the class score w.r.t. the last
     convolutional feature map, then applies global-average-pooled
     weighting.  Highlights WHICH SPATIAL REGIONS drove the
     CNN prediction (periocular, philtrum, nasal areas).

  2. Attention Rollout (ViT branch)
     ────────────────────────────────
     Propagates attention weights through ALL transformer layers
     (Abnar & Zuidema, 2020), accounting for residual connections.
     Produces a single heatmap showing WHICH IMAGE PATCHES the
     ViT attended to globally.

Both heatmaps are overlaid on the original de-normalised face image
and saved side-by-side as a single PNG for direct clinical comparison.

Output layout per sample:
  [ Original Face ] [ Grad-CAM Overlay ] [ Attention Rollout Overlay ]
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)

# ImageNet inverse-normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BG     = "#0d0d12"
_COL_C  = "#4fc3f7"   # cyan — Grad-CAM title
_COL_V  = "#f4a261"   # amber — Attention Rollout title
_COL_O  = "#aaaaaa"   # grey  — original


# ── Tensor → displayable image ────────────────────────────────────────────────

def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalised float tensor → (H,W,3) uint8 BGR."""
    img = t.cpu().permute(1, 2, 0).numpy()
    img = (img * _STD + _MEAN).clip(0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def overlay_heatmap(
    bgr:     np.ndarray,
    heatmap: np.ndarray,    # (H,W) float [0,1]
    alpha:   float = 0.50,
    cmap:    int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Return a BGR image with the heatmap blended over the original."""
    h, w = bgr.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    hm8  = (hm * 255).astype(np.uint8)
    col  = cv2.applyColorMap(hm8, cmap)
    return cv2.addWeighted(bgr, 1 - alpha, col, alpha, 0)


def _norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


# ════════════════════════════════════════════════════════════════
#  1. Grad-CAM
# ════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Manual hook-based Grad-CAM.

    Optionally wraps pytorch-grad-cam if installed (more robust
    to complex backbone architectures).
    """

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model         = model
        self.target_layers = target_layers
        self._use_lib      = self._try_library()

        if self._use_lib:
            from pytorch_grad_cam import GradCAM as LibGC
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            self._gc          = LibGC(model=model, target_layers=target_layers)
            self._ClsTarget   = ClassifierOutputTarget
            log.info("Grad-CAM: using pytorch-grad-cam library")
        else:
            self._acts: Optional[torch.Tensor] = None
            self._grads: Optional[torch.Tensor] = None
            self._attach_hooks()
            log.info("Grad-CAM: using manual hooks")

    # ------------------------------------------------------------------
    @staticmethod
    def _try_library() -> bool:
        try:
            import pytorch_grad_cam  # noqa
            return True
        except ImportError:
            return False

    def _attach_hooks(self):
        layer = self.target_layers[-1]

        def fwd(_, __, out):
            self._acts = out.detach()

        def bwd(_, __, grad_out):
            self._grads = grad_out[0].detach()

        layer.register_forward_hook(fwd)
        layer.register_full_backward_hook(bwd)

    # ------------------------------------------------------------------
    def compute(
        self,
        input_t:      torch.Tensor,         # (1,3,H,W)
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Returns (H,W) float32 heatmap in [0,1]."""

        if self._use_lib:
            tgt = ([self._ClsTarget(target_class)]
                   if target_class is not None else None)
            cam = self._gc(input_tensor=input_t, targets=tgt)
            return cam[0].astype(np.float32)   # already [0,1]

        # Manual path
        self.model.zero_grad()
        logits = self.model(input_t)
        cls    = int(logits.argmax(1).item()) if target_class is None else target_class
        logits[0, cls].backward()

        grads = self._grads                   # (1,C,h,w)
        acts  = self._acts                    # (1,C,h,w)

        weights  = grads.mean(dim=(2, 3), keepdim=True)
        cam_raw  = F.relu((weights * acts).sum(dim=1).squeeze(0))
        return _norm01(cam_raw.cpu().numpy()).astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  2. Attention Rollout
# ════════════════════════════════════════════════════════════════

class AttentionRollout:
    """
    Attention Rollout across ALL transformer layers.

    Abnar & Zuidema (2020):  A_rollout = ∏_l  (0.5·A_l + 0.5·I)

    The [CLS]→patch row of the final rollout matrix gives the
    global relevance of each image patch.

    discard_ratio: fraction of lowest-attention patches zeroed out
                   to suppress noisy background responses.
    """

    def __init__(self, model: nn.Module, discard_ratio: float = 0.9):
        self.model        = model
        self.discard_ratio = discard_ratio

    # ------------------------------------------------------------------
    def compute(
        self,
        input_t:    torch.Tensor,   # (1,3,H,W)
        patch_size: int = 16,
    ) -> np.ndarray:
        """Returns (H,W) float32 heatmap in [0,1]."""

        attn_list: List[np.ndarray] = []

        def _hook(module, inp, out):
            # out shape depends on timm version:
            #   some return the attended values, some return attention weights.
            # We try to capture the stored attention weight if possible.
            attn_list.append(out.detach().cpu().numpy())

        # Attach hooks to every block's attention sub-module
        vit   = self.model.vit_branch.vit
        hooks = []
        for block in vit.blocks:
            h = block.attn.register_forward_hook(_hook)
            hooks.append(h)

        with torch.no_grad():
            _ = self.model(input_t)

        for h in hooks:
            h.remove()

        # If attn_list elements have shape (B, H, N, N) we do rollout.
        # Otherwise fall back to last-layer single attention.
        rollout_map = self._rollout_from_list(attn_list, input_t.shape[-1])
        if rollout_map is None:
            # Use ViT branch helper (returns last-layer only)
            attn_t = self.model.get_vit_attention(input_t)
            if attn_t is None:
                log.warning("Attention rollout failed — returning zeros")
                H = input_t.shape[-1]
                return np.zeros((H, H), dtype=np.float32)
            attn_np = attn_t[0].cpu().float().numpy()   # (heads, N, N)
            rollout_map = self._single_layer_map(
                attn_np, input_t.shape[-1], patch_size
            )

        return rollout_map

    # ------------------------------------------------------------------
    class AttentionRollout:
    def __init__(self, model: nn.Module, discard_ratio: float = 0.9):
        self.model         = model
        self.discard_ratio = discard_ratio

    def compute(self, input_t: torch.Tensor, patch_size: int = 16) -> np.ndarray:
        attn_list = []

        def _hook(module, inp, out):
            # Get attention weights directly from the attention module
            B, N, C = inp[0].shape
            qkv = module.qkv(inp[0])
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            scale = (C // module.num_heads) ** -0.5
            attn  = (q @ k.transpose(-2, -1)) * scale
            attn  = attn.softmax(dim=-1)
            attn_list.append(attn.detach().cpu().numpy())

        vit   = self.model.vit_branch.vit
        hooks = []
        for block in vit.blocks:
            h = block.attn.register_forward_hook(_hook)
            hooks.append(h)

        with torch.no_grad():
            _ = self.model(input_t)

        for h in hooks:
            h.remove()

        if not attn_list:
            log.warning("No attention maps captured")
            return np.zeros((input_t.shape[-1], input_t.shape[-1]), dtype=np.float32)

        # Attention rollout across all layers
        img_size = input_t.shape[-1]
        I        = None
        rollout  = None

        for attn in attn_list:
            # attn shape: (B, num_heads, N, N)
            avg = attn[0].mean(axis=0)           # (N, N)
            N   = avg.shape[0]
            if I is None:
                I = np.eye(N)
            a_hat = 0.5 * avg + 0.5 * I
            a_hat /= a_hat.sum(axis=-1, keepdims=True)
            rollout = a_hat if rollout is None else a_hat @ rollout

        # CLS token row -> patch relevances
        cls_row = rollout[0, 1:]

        # Discard lowest attention
        thr     = np.quantile(cls_row, self.discard_ratio)
        cls_row = cls_row.copy()
        cls_row[cls_row < thr] = 0.0

        # Reshape to grid and upsample
        num_patches = cls_row.shape[0]
        grid_size   = int(np.sqrt(num_patches))
        grid        = cls_row[:grid_size * grid_size].reshape(grid_size, grid_size)
        heatmap     = cv2.resize(grid.astype(np.float32), (img_size, img_size),
                                 interpolation=cv2.INTER_CUBIC)

        mn, mx = heatmap.min(), heatmap.max()
        if mx > mn:
            heatmap = (heatmap - mn) / (mx - mn)

        return heatmap.astype(np.float32)

    def _single_layer_map(
        self,
        attn_np:   np.ndarray,   # (H, N, N)
        img_size:  int,
        patch_size: int = 16,
    ) -> np.ndarray:
        avg = attn_np.mean(axis=0)           # (N, N)
        cls_row = avg[0, 1:]
        return self._reshape_and_upsample(cls_row, img_size, patch_size)

    def _reshape_and_upsample(
        self,
        cls_row:   np.ndarray,   # (num_patches,)
        img_size:  int,
        patch_size: int = 16,
    ) -> np.ndarray:
        # Discard lowest-attention patches
        thr         = np.quantile(cls_row, self.discard_ratio)
        cls_row     = cls_row.copy()
        cls_row[cls_row < thr] = 0.0

        num_patches = cls_row.shape[0]
        grid_size   = int(np.sqrt(num_patches))

        # Gracefully handle non-square patch grids
        if grid_size * grid_size != num_patches:
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            pad       = grid_size * grid_size - num_patches
            cls_row   = np.concatenate([cls_row, np.zeros(pad)])

        grid    = cls_row.reshape(grid_size, grid_size)
        heatmap = cv2.resize(grid.astype(np.float32), (img_size, img_size),
                             interpolation=cv2.INTER_CUBIC)
        return _norm01(heatmap).astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  Explainability Engine
# ════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """
    Orchestrates Grad-CAM + Attention Rollout for a batch of images,
    renders and saves side-by-side comparison figures.
    """

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device):
        self.model   = model
        self.device  = device
        self.cfg     = cfg

        exp_cfg      = cfg.get("explainability", {})
        self.out_dir = Path(exp_cfg.get("output_dir", "outputs/explanations"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.alpha    = exp_cfg.get("overlay_alpha", 0.5)
        cmap_name     = exp_cfg.get("colormap", "jet").upper()
        self.cv_cmap  = getattr(cv2, f"COLORMAP_{cmap_name}", cv2.COLORMAP_JET)

        self.class_names = cfg["dataset"].get("classes", ["autism", "control"])

        # Initialise explainers
        tgt_layers   = model.cnn_target_layers
        self.gradcam  = GradCAM(model, tgt_layers) if tgt_layers else None
        self.rollout  = AttentionRollout(model)

        if not tgt_layers:
            log.warning("No CNN target layers found — Grad-CAM will be skipped")

    # ------------------------------------------------------------------
    def explain_batch(
        self,
        images:  torch.Tensor,         # (B,3,H,W) normalised
        labels:  torch.Tensor,         # (B,) ground-truth
        indices: Optional[List] = None,
    ):
        """Generate and save an explanation figure for every image."""
        B = images.size(0)
        self.model.eval()

        for i in range(B):
            img_t  = images[i].unsqueeze(0).to(self.device)
            label  = int(labels[i].item())
            name   = str(indices[i]) if indices else str(i)

            # Prediction
            with torch.no_grad():
                logits = self.model(img_t)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred   = int(probs.argmax())

            bgr_img = tensor_to_bgr(images[i])
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # ── Grad-CAM ─────────────────────────────────────────
            if self.gradcam is not None:
                try:
                    gc_map     = self.gradcam.compute(img_t, target_class=pred)
                    gc_overlay = overlay_heatmap(bgr_img, gc_map,
                                                 self.alpha, self.cv_cmap)
                    gc_rgb     = cv2.cvtColor(gc_overlay, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    log.warning(f"Grad-CAM error on sample {name}: {e}")
                    gc_rgb = rgb_img
            else:
                gc_rgb = rgb_img

            # ── Attention Rollout ─────────────────────────────────
            try:
                ro_map     = self.rollout.compute(img_t)
                ro_overlay = overlay_heatmap(bgr_img, ro_map,
                                             self.alpha, self.cv_cmap)
                ro_rgb     = cv2.cvtColor(ro_overlay, cv2.COLOR_BGR2RGB)
            except Exception as e:
                log.warning(f"Rollout error on sample {name}: {e}")
                ro_rgb = rgb_img

            self._save_figure(rgb_img, gc_rgb, ro_rgb,
                              label, pred, probs, name)

        log.info(f"Saved {B} explanation figures → {self.out_dir}")

    # ------------------------------------------------------------------
    def _save_figure(
        self,
        orig:     np.ndarray,
        gc_img:   np.ndarray,
        ro_img:   np.ndarray,
        true_lbl: int,
        pred_lbl: int,
        probs:    np.ndarray,
        name:     str,
    ):
        cls  = self.class_names
        true = cls[true_lbl] if true_lbl < len(cls) else str(true_lbl)
        pred = cls[pred_lbl] if pred_lbl < len(cls) else str(pred_lbl)
        tick = "✓" if true_lbl == pred_lbl else "✗"
        conf = probs[pred_lbl] * 100

        fig = plt.figure(figsize=(13, 4.8), facecolor=_BG)
        fig.suptitle(
            f"Sample {name}   │   True: {true}   Pred: {pred} "
            f"({conf:.1f}%)  {tick}",
            fontsize=12, color="white", y=1.02,
        )

        gs   = gridspec.GridSpec(1, 3, wspace=0.04)
        imgs = [orig,   gc_img,                     ro_img]
        ttls = ["Original Face",
                "Grad-CAM  (CNN Branch)",
                "Attention Rollout  (ViT Branch)"]
        clrs = [_COL_O, _COL_C, _COL_V]

        for col, (im, title, colour) in enumerate(zip(imgs, ttls, clrs)):
            ax = fig.add_subplot(gs[col])
            ax.imshow(im)
            ax.set_title(title, color=colour, fontsize=9, pad=5)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.8)

        plt.tight_layout()
        out = self.out_dir / f"explanation_{name}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def explain_test_samples(
    model:   nn.Module,
    loader,
    cfg:     dict,
    device:  torch.device,
    n:       int = 16,
):
    """Pull `n` samples from `loader` and generate explanation figures."""
    engine = ExplainabilityEngine(model, cfg, device)
    model.eval()

    imgs_list, lbls_list, idx_list = [], [], []
    count = 0

    for batch_imgs, batch_labels in loader:
        for j in range(batch_imgs.size(0)):
            if count >= n:
                break
            imgs_list.append(batch_imgs[j])
            lbls_list.append(batch_labels[j])
            idx_list.append(count)
            count += 1
        if count >= n:
            break

    if not imgs_list:
        log.warning("No samples for explanation")
        return

    imgs_t = torch.stack(imgs_list)
    lbls_t = torch.stack(lbls_list)
    engine.explain_batch(imgs_t, lbls_t, idx_list)
