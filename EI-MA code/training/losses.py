"""Loss helpers: KGE, generated supervision, total objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kge_1n_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Binary 1-N completion loss on candidate scores.
    scores/labels: [B, |V|] or flattened.
    L = - mean[ y log σ(ψ) + (1-y) log(1-σ(ψ)) ]
    """
    return F.binary_cross_entropy_with_logits(scores, labels)


def generated_loss(
    action: str,
    L_dec: torch.Tensor,
    L_aug: torch.Tensor,
) -> torch.Tensor:
    """L_gen = I[a ≠ Skip] (L_dec + L_aug)."""
    if action == "Skip":
        return L_aug.new_tensor(0.0)
    return L_dec + L_aug


def total_loss(
    L_kge: torch.Tensor,
    L_gen: torch.Tensor,
    L_pi: torch.Tensor,
    lambda_g: float = 0.1,
    lambda_pi: float = 1.0,
) -> torch.Tensor:
    """L = L_KGE + λ_g L_gen + λ_π L_π"""
    return L_kge + lambda_g * L_gen + lambda_pi * L_pi


def fuse_static_embedding(
    e: torch.Tensor,
    h_bar: torch.Tensor,
    W_fuse: torch.nn.Linear,
) -> torch.Tensor:
    """
    Test-time static fusion:
    e+ = W_fuse [e; sg(h̄)] + b
    """
    x = torch.cat([e, h_bar.detach()], dim=-1)
    return W_fuse(x)
