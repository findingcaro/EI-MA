"""Graph-bounded gated communication and pair interaction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCommunicator(nn.Module):
    """
    rho_{i,k} = sigma(W_g [h_i; h_k] + b_g)
    alpha_{i->j,k} = softmax_k (w_alpha^T rho)
    m_{i->j} = sum_k alpha * W_up (rho ⊙ W_down h_i)
    """

    def __init__(self, dim: int, d_lr: int = 32):
        super().__init__()
        self.W_g = nn.Linear(2 * dim, d_lr)
        self.w_alpha = nn.Parameter(torch.randn(d_lr))
        self.W_down = nn.Linear(dim, d_lr, bias=False)
        self.W_up = nn.Linear(d_lr, dim, bias=False)
        self.W_dir = nn.Linear(dim, dim, bias=False)  # direct-edge fallback

    def gate(self, h_i: torch.Tensor, h_k: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.W_g(torch.cat([h_i, h_k], dim=-1)))

    def message(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        neighbor_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        neighbor_states: [K, d] common neighbors; if K=0 use direct fallback.
        """
        if neighbor_states.numel() == 0:
            return self.W_dir(h_i)

        # rho for each neighbor k
        hi = h_i.unsqueeze(0).expand(neighbor_states.size(0), -1)
        rho = self.gate(hi, neighbor_states)  # [K, d_lr]
        logits = rho @ self.w_alpha
        alpha = F.softmax(logits, dim=0)  # [K]
        down = self.W_down(h_i)  # [d_lr]
        mixed = rho * down.unsqueeze(0)
        m = (alpha.unsqueeze(-1) * self.W_up(mixed)).sum(dim=0)
        return m


class PairInteraction(nn.Module):
    """z_ij = MLP_z [u; s; d; p] with s=m_ij+m_ji, d=|m_ij-m_ji|, p=m_ij⊙m_ji."""

    def __init__(self, u_dim: int, dim: int, z_dim: int = 256):
        super().__init__()
        self.mlp_z = nn.Sequential(
            nn.Linear(u_dim + 3 * dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(
        self,
        u_ij: torch.Tensor,
        m_ij: torch.Tensor,
        m_ji: torch.Tensor,
    ) -> torch.Tensor:
        s = m_ij + m_ji
        d = (m_ij - m_ji).abs()
        p = m_ij * m_ji
        return self.mlp_z(torch.cat([u_ij, s, d, p], dim=-1))


class RelationDecoder(nn.Module):
    """Auxiliary L_dec = -log p(r | z_ij)."""

    def __init__(self, z_dim: int, num_rel: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, num_rel)

    def loss(self, z_ij: torch.Tensor, rel_id: torch.Tensor) -> torch.Tensor:
        logits = self.fc(z_ij)
        return F.cross_entropy(logits.unsqueeze(0), rel_id.view(1))
