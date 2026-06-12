"""
models/losses.py — Loss Functions
===================================
FocalLoss  : down-weights easy/well-classified examples
             so the model focuses on hard misclassified cases.
             Particularly useful for class-imbalanced ASD datasets.

build_criterion() : returns FocalLoss or CrossEntropyLoss per config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al., ICCV 2017.

        FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    Args:
        gamma:     focusing exponent. 0 → standard cross-entropy.
        alpha:     per-class weight tensor shape (C,), or None.
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma:     float                    = 2.0,
        alpha:     Optional[torch.Tensor]   = None,
        reduction: str                      = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C)  — raw un-normalised model output
        targets: (B,)    — integer class indices
        """
        log_p = F.log_softmax(logits, dim=1)                     # (B, C)
        p     = torch.exp(log_p)                                  # (B, C)

        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1) # (B,)
        pt     = p.gather(1,     targets.unsqueeze(1)).squeeze(1) # (B,)

        focal_w = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_w = alpha_t * focal_w

        loss = -(focal_w * log_pt)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    cfg:           dict,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Return FocalLoss (gamma > 0) or CrossEntropyLoss (gamma == 0).
    class_weights are applied in both cases when provided.
    """
    gamma = cfg["training"].get("focal_loss_gamma", 2.0)
    if gamma > 0:
        return FocalLoss(gamma=gamma, alpha=class_weights)
    return nn.CrossEntropyLoss(weight=class_weights)
