"""Lightweight StarE-style scorer stub (replace with real StarE in full code)."""

from __future__ import annotations

import torch
import torch.nn as nn


class StarEBackboneStub(nn.Module):
    """
    Placeholder for StarE + Transformer scoring function Psi_theta.
    Shared dense parameters θ_sh are those used for EI probe gradients.
    """

    def __init__(self, num_ent: int, num_rel: int, dim: int = 256):
        super().__init__()
        self.ent = nn.Embedding(num_ent, dim)
        self.rel = nn.Embedding(num_rel, dim)
        self.scorer = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        # agent state / cache
        self.agent_state = nn.Embedding(num_ent, dim)
        self.fuse = nn.Linear(2 * dim, dim)

    def shared_parameters(self):
        # illustrative: treat scorer MLP as shared dense θ_sh
        return list(self.scorer.parameters())

    def score(self, s, r, o, qualifiers=None) -> torch.Tensor:
        x = torch.cat([self.ent(s), self.rel(r), self.ent(o)], dim=-1)
        return self.scorer(x).squeeze(-1)

    def entity_embedding(self, eids: torch.Tensor) -> torch.Tensor:
        return self.ent(eids)
