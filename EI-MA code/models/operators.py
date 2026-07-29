"""Four augmentation operators: Skip, Recon, Struct-Pos, Hard-Neg."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from config import ACTIONS


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


class AugmentationOperators:
    """
    Leakage-free candidates C_un use train-only observed objects T(s,r).
    Struct-Pos uses an AnyBURL-style offline cache (injected as callable).
    """

    def __init__(
        self,
        score_fn: Callable,
        hardneg_margin: float = 1.0,
        hardneg_beta: float = 1.0,
        anyburl_best: Optional[Callable] = None,
    ):
        self.score_fn = score_fn  # Psi_theta(s,r,o,A) -> scalar
        self.m = hardneg_margin
        self.beta = hardneg_beta
        # anyburl_best(s, r, C_un) -> o+ or None
        self.anyburl_best = anyburl_best

    def skip(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def recon(self, anchor) -> torch.Tensor:
        # L = -log sigma(Psi(q0))
        s = self.score_fn(*anchor)
        return -F.logsigmoid(s)

    def struct_pos(self, anchor, C_un: Sequence[int]) -> torch.Tensor:
        s, r, o, A = anchor
        if self.anyburl_best is None or len(C_un) == 0:
            return self.recon(anchor)
        o_plus = self.anyburl_best(s, r, list(C_un))
        if o_plus is None:
            return self.recon(anchor)
        sp = self.score_fn(s, r, o_plus, A)
        return -F.logsigmoid(sp)

    def hard_neg(self, anchor, C_un: Sequence[int]) -> torch.Tensor:
        """
        o- = argmax_{v in C_un} Psi(s,r,v,A)
        delta = Psi(q0) - Psi(s,r,o-,A)
        L = w(delta) * softplus(m - delta),  w = sg[sigmoid(beta * delta)]
        """
        s, r, o, A = anchor
        score_pos = self.score_fn(s, r, o, A)

        if len(C_un) == 0:
            # typed uniform corrupt placeholder: random negative score
            score_neg = score_pos - 1.0
        else:
            # score all candidates (schematic; real code batches this)
            neg_scores = torch.stack([self.score_fn(s, r, v, A) for v in C_un])
            score_neg = neg_scores.max()

        delta = score_pos - score_neg
        w = torch.sigmoid(self.beta * delta).detach()  # stop-gradient
        return w * softplus(self.m - delta)

    def loss(self, action: str, anchor, C_un: Sequence[int]) -> torch.Tensor:
        if action == "Skip":
            return self.skip()
        if action == "Recon":
            return self.recon(anchor)
        if action == "Struct-Pos":
            return self.struct_pos(anchor, C_un)
        if action == "Hard-Neg":
            return self.hard_neg(anchor, C_un)
        raise ValueError(f"Unknown action: {action}")


def unobserved_candidates(
    candidates: Sequence[int],
    observed_objects: Sequence[int],
) -> list:
    """C_un = C \\ T(s,r)"""
    obs = set(observed_objects)
    return [c for c in candidates if c not in obs]
