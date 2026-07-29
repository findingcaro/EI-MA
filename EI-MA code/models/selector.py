"""Shared EI-driven operator selector and ranking loss L_π."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ACTIONS
from training.ei_label import ranking_pairs


class OperatorSelector(nn.Module):
    """
    Û = MLP_π(sg(u_ij))
    π(a|i,j) = softmax(Û / τ)_a
    """

    def __init__(self, u_dim: int, hidden: int = 256, n_actions: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(u_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.action_to_idx = {a: i for i, a in enumerate(ACTIONS)}

    def utilities(self, u_ij: torch.Tensor) -> torch.Tensor:
        return self.mlp(u_ij.detach())  # stop-gradient on routing features

    def policy(self, u_ij: torch.Tensor, tau: float) -> torch.Tensor:
        return F.softmax(self.utilities(u_ij) / max(tau, 1e-6), dim=-1)

    def sample(self, u_ij: torch.Tensor, tau: float) -> str:
        probs = self.policy(u_ij, tau)
        idx = torch.multinomial(probs, num_samples=1).item()
        return ACTIONS[idx]


def ranking_loss(
    utilities: torch.Tensor,
    ei: Dict[str, float],
    margin: float = 0.1,
) -> torch.Tensor:
    """
    L_π = sum_{a<b, EI_a ≠ EI_b} max(0, μ - s_ab (U_a - U_b))
    s_ab = sign(EI_a - EI_b)
    """
    act_index = {a: i for i, a in enumerate(ACTIONS)}
    loss = utilities.new_tensor(0.0)
    n = 0
    for a, b, s_ab in ranking_pairs(ei):
        ua = utilities[act_index[a]]
        ub = utilities[act_index[b]]
        loss = loss + F.relu(margin - s_ab * (ua - ub))
        n += 1
    if n == 0:
        return utilities.new_tensor(0.0)
    return loss / n
