"""Local evidence retrieval (Eq. local-basics / memory summary)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactEncoder(nn.Module):
    """f_ω: encode a hyper-relational fact into R^d (placeholder)."""

    def __init__(self, num_ent: int, num_rel: int, dim: int):
        super().__init__()
        self.ent = nn.Embedding(num_ent, dim)
        self.rel = nn.Embedding(num_rel, dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        # Qualifiers can be pooled separately in a full StarE-style encoder.
        x = torch.cat([self.ent(s), self.rel(r), self.ent(o)], dim=-1)
        return self.mlp(x)


class LocalRetriever(nn.Module):
    """
    g_{i|j} = MLP_r [h_i; h_j; phi(d)]
    alpha soft-attn over top-M facts → r_i
    """

    def __init__(self, dim: int, d_phi: int = 16, M: int = 16):
        super().__init__()
        self.M = M
        self.mlp_r = nn.Sequential(
            nn.Linear(2 * dim + d_phi, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.dist_emb = nn.Embedding(8, d_phi)  # distance embedding phi(d_A)

    def query(self, h_i: torch.Tensor, h_j: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        phi = self.dist_emb(dist.clamp(max=7))
        return self.mlp_r(torch.cat([h_i, h_j, phi], dim=-1))

    def summarize(
        self,
        g: torch.Tensor,
        fact_embs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        fact_embs: [N_facts, d]
        returns r: [d], alpha: [N_facts] (after top-M truncation)
        """
        if fact_embs.numel() == 0:
            return torch.zeros_like(g), g.new_zeros(0)

        scores = fact_embs @ g  # [N]
        k = min(self.M, fact_embs.size(0))
        topv, topi = torch.topk(scores, k=k)
        selected = fact_embs[topi]
        alpha = F.softmax(topv, dim=0)
        r = (alpha.unsqueeze(-1) * selected).sum(dim=0)
        return r, alpha


def build_selector_input(
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    r_i: torch.Tensor,
    r_j: torch.Tensor,
    phi_d: torch.Tensor,
) -> torch.Tensor:
    """u_ij = [h_i; h_j; r_i; r_j; r_i ⊙ r_j; phi(d)]."""
    return torch.cat([h_i, h_j, r_i, r_j, r_i * r_j, phi_d], dim=-1)
