"""
models/hybrid_model.py — Hybrid CNN + Vision Transformer
==========================================================

Two-branch architecture:

  Branch A ── CNN (ResNet-50 / EfficientNet-B3 / MobileNet-V3)
                 Captures fine-grained LOCAL features:
                 periocular region, midface morphology, philtrum shape.

  Branch B ── Vision Transformer (ViT-Small/16 via timm)
                 Captures GLOBAL spatial relationships
                 through multi-head self-attention.

Both branches project to the same embedding dimension (default 512-d),
then are fused via one of three strategies:

  • "attention"  ← PRIMARY (novel): learnable cross-attention fusion
  • "concat"     : concatenate → MLP → classify
  • "add"        : element-wise sum → classify

Cross-Attention Fusion (novel contribution)
-------------------------------------------
  CNN features query the ViT key/value space  →  CNN-attended vector
  ViT features query the CNN key/value space  →  ViT-attended vector
  Concatenate → project to hidden_dim → classify

This allows each branch to selectively absorb complementary information
from the other before the final decision, rather than naïve late fusion.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  CNN Branch
# ═══════════════════════════════════════════════════════════════

class CNNBranch(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, out_dim: int):
        super().__init__()
        import torchvision.models as tvm

        backbone_name = backbone_name.lower()
        self._is_timm = False

        if backbone_name == "resnet50":
            base   = tvm.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            in_dim = base.fc.in_features
            base.fc = nn.Identity()
            self._last_conv_attr = "layer4"

        elif backbone_name == "resnet34":
            base   = tvm.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            in_dim = base.fc.in_features
            base.fc = nn.Identity()
            self._last_conv_attr = "layer4"

        elif backbone_name == "efficientnet_b3":
            base   = tvm.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
            in_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self._last_conv_attr = "features"

        elif backbone_name == "mobilenet_v3_large":
            base   = tvm.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
            in_dim = base.classifier[3].in_features
            base.classifier = nn.Identity()
            self._last_conv_attr = "features"

        elif backbone_name == "xception":
            import timm
            base   = timm.create_model("xception", pretrained=pretrained, num_classes=0)
            in_dim = base.num_features
            self._last_conv_attr = "conv4"
            self._is_timm = True

        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone_name!r}")

        self.cnn = base   # store as self.cnn to avoid any name conflicts

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        log.info(f"CNN branch | {backbone_name} | {in_dim}->{out_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])  # global average pool
        return self.proj(feats)

    @property
    def grad_cam_target_layers(self) -> List[nn.Module]:
        attr = getattr(self.cnn, self._last_conv_attr, None)
        return [attr] if attr is not None else []

# ═══════════════════════════════════════════════════════════════
#  Vision Transformer Branch
# ═══════════════════════════════════════════════════════════════

class ViTBranch(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, out_dim: int):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required.  Run: pip install timm")

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        in_dim = self.vit.num_features

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        log.info(f"ViT branch  | {model_name} | {in_dim}->{out_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, out_dim)"""
        feats = self.vit(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        return self.proj(feats)

    def get_last_attention(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        attn_output = []

        def _hook(module, inp, out):
            attn_output.append(out)

        last_block = self.vit.blocks[-1]
        handle     = last_block.attn.register_forward_hook(_hook)

        with torch.no_grad():
            self.vit(x)

        handle.remove()

        if not attn_output:
            return None

        return attn_output[0]
# ═══════════════════════════════════════════════════════════════
#  Cross-Attention Fusion  (novel contribution)
# ═══════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion of two feature vectors.

    Each branch queries the other branch's key-value space,
    allowing selective information transfer before classification.

      CNN  →  (Q_cnn, K_vit, V_vit)  →  CNN-attended  ∈ ℝ^feat_dim
      ViT  →  (Q_vit, K_cnn, V_cnn)  →  ViT-attended  ∈ ℝ^feat_dim
      cat  →  Linear(2·feat_dim, hidden_dim)  →  LayerNorm  →  GELU
    """

    def __init__(self, feat_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        assert feat_dim % num_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = feat_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # CNN → ViT direction
        self.Wq_cnn = nn.Linear(feat_dim, feat_dim, bias=False)
        self.Wk_vit = nn.Linear(feat_dim, feat_dim, bias=False)
        self.Wv_vit = nn.Linear(feat_dim, feat_dim, bias=False)

        # ViT → CNN direction
        self.Wq_vit = nn.Linear(feat_dim, feat_dim, bias=False)
        self.Wk_cnn = nn.Linear(feat_dim, feat_dim, bias=False)
        self.Wv_cnn = nn.Linear(feat_dim, feat_dim, bias=False)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    def _single_cross_attn(
        self,
        q_feat: torch.Tensor,   # (B, D)
        k_feat: torch.Tensor,   # (B, D)
        v_feat: torch.Tensor,   # (B, D)
        Wq: nn.Linear,
        Wk: nn.Linear,
        Wv: nn.Linear,
    ) -> torch.Tensor:
        B, D = q_feat.shape
        H, Hd = self.num_heads, self.head_dim

        # Reshape to (B, H, 1, Hd)  — single token per branch
        q = Wq(q_feat).view(B, H, 1, Hd)
        k = Wk(k_feat).view(B, H, 1, Hd)
        v = Wv(v_feat).view(B, H, 1, Hd)

        # Scaled dot-product attention over the single token
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale,
            dim=-1,
        )                               # (B, H, 1, 1)
        out  = torch.matmul(attn, v)   # (B, H, 1, Hd)
        return out.reshape(B, D)       # (B, D)

    # ------------------------------------------------------------------
    def forward(
        self,
        cnn_feat: torch.Tensor,    # (B, D)
        vit_feat: torch.Tensor,    # (B, D)
    ) -> torch.Tensor:
        cnn_attended = self._single_cross_attn(
            cnn_feat, vit_feat, vit_feat,
            self.Wq_cnn, self.Wk_vit, self.Wv_vit,
        )
        vit_attended = self._single_cross_attn(
            vit_feat, cnn_feat, cnn_feat,
            self.Wq_vit, self.Wk_cnn, self.Wv_cnn,
        )
        fused = torch.cat([cnn_attended, vit_attended], dim=-1)   # (B, 2D)
        return self.out_proj(fused)                                # (B, hidden_dim)


# ═══════════════════════════════════════════════════════════════
#  Classifier Head
# ═══════════════════════════════════════════════════════════════

class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
#  Full Hybrid Model
# ═══════════════════════════════════════════════════════════════

class HybridASDModel(nn.Module):
    """
    Full Hybrid CNN + Vision Transformer model.

    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)  raw logits

    Architecture:
        ┌─── CNN Branch ────────────────┐
        │ ResNet-50 → proj → (B,512)    │
        └──────────────┬────────────────┘
                       │
        ┌─── ViT Branch ────────────────┐
        │ ViT-S/16  → proj → (B,512)   │
        └──────────────┬────────────────┘
                       │
              CrossAttentionFusion
                 (B, 256)
                       │
               Dropout + Linear
                 (B, 2)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        feat_dim    = m["cnn_feature_dim"]
        vit_dim     = m["vit_feature_dim"]
        hidden_dim  = m["fusion_hidden_dim"]
        num_heads   = m.get("fusion_num_heads", 4)
        fusion_type = m["fusion_type"]
        num_classes = m["num_classes"]
        dropout     = m["dropout_rate"]

        assert feat_dim == vit_dim, \
            "cnn_feature_dim must equal vit_feature_dim"

        # ── Branches ──────────────────────────────────────────────────
        self.cnn_branch = CNNBranch(
            backbone_name = m["cnn_backbone"],
            pretrained    = m["cnn_pretrained"],
            out_dim       = feat_dim,
        )
        self.vit_branch = ViTBranch(
            model_name = m["vit_backbone"],
            pretrained = m["vit_pretrained"],
            out_dim    = vit_dim,
        )

        # ── Fusion ────────────────────────────────────────────────────
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            self.fusion   = CrossAttentionFusion(feat_dim, hidden_dim, num_heads)
            clf_in        = hidden_dim

        elif fusion_type == "concat":
            self.fusion   = nn.Sequential(
                nn.Linear(feat_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            clf_in = hidden_dim

        elif fusion_type == "add":
            self.fusion = nn.Identity()
            clf_in      = feat_dim

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type!r}")

        # ── Classifier ────────────────────────────────────────────────
        self.classifier = ClassifierHead(clf_in, num_classes, dropout)

        log.info(
            f"HybridASDModel │ fusion={fusion_type} │ "
            f"feat={feat_dim} → clf_in={clf_in} → {num_classes}"
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x:               torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        cnn_feat = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)

        if self.fusion_type == "attention":
            fused = self.fusion(cnn_feat, vit_feat)
        elif self.fusion_type == "concat":
            fused = self.fusion(torch.cat([cnn_feat, vit_feat], dim=-1))
        else:   # add
            fused = cnn_feat + vit_feat

        logits = self.classifier(fused)

        if return_features:
            return logits, fused
        return logits

    # ------------------------------------------------------------------
    def get_vit_attention(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return self.vit_branch.get_last_attention(x)

    @property
    def cnn_target_layers(self) -> List[nn.Module]:
        return self.cnn_branch.grad_cam_target_layers

    # ------------------------------------------------------------------
    def freeze_backbones(self):
        for p in self.cnn_branch.cnn.parameters():
            p.requires_grad = False
        for p in self.vit_branch.vit.parameters():
            p.requires_grad = False
        log.info("Backbones frozen (only proj + fusion + head trainable)")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        log.info("All parameters unfrozen - full fine-tuning")

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: Optional[torch.device] = None) -> HybridASDModel:
    # Force Xception as CNN backbone
    cfg["model"]["cnn_backbone"]    = "xception"
    cfg["model"]["cnn_pretrained"]  = True
    cfg["model"]["cnn_feature_dim"] = 512
    model = HybridASDModel(cfg)
    if device:
        model = model.to(device)
    log.info(f"Trainable parameters: {model.parameter_count():,}")
    return model