"""First-order Expected Improvement labels (Eq. ei)."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

import torch

from config import ACTIONS


def flatten_shared_params(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params if p.requires_grad])


def grad_shared(
    loss: torch.Tensor,
    shared_params: Sequence[torch.nn.Parameter],
    create_graph: bool = False,
) -> torch.Tensor:
    """g = ∇_{θ_sh} loss, flattened."""
    grads = torch.autograd.grad(
        loss,
        shared_params,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )
    pieces = []
    for p, g in zip(shared_params, grads):
        if g is None:
            pieces.append(torch.zeros_like(p).reshape(-1))
        else:
            pieces.append(g.reshape(-1))
    return torch.cat(pieces)


@torch.no_grad()
def first_order_ei(
    probe_grad: torch.Tensor,
    action_grad: torch.Tensor,
    eta: float,
) -> float:
    """
    ΔM̂(i,j,a) = η < v_H , -g_a >
    Skip has g=0 → EI=0.
    """
    if action_grad is None:
        return 0.0
    val = eta * torch.dot(probe_grad, -action_grad)
    return float(val.item())


def estimate_action_eis(
    probe_loss_fn: Callable[[], torch.Tensor],
    aug_loss_fn: Callable[[str], torch.Tensor],
    shared_params: Sequence[torch.nn.Parameter],
    eta: float,
) -> Dict[str, float]:
    """
    For one pair/anchor, compute EI for all four actions under the same backbone state.
    M = -L_KGE(H); v = ∇ M = -∇ L_KGE(H)
    """
    # probe utility gradient
    L_probe = probe_loss_fn()
    M = -L_probe
    v = grad_shared(M, shared_params)

    eis: Dict[str, float] = {}
    for a in ACTIONS:
        if a == "Skip":
            eis[a] = 0.0
            continue
        L_a = aug_loss_fn(a)
        g_a = grad_shared(L_a, shared_params)
        eis[a] = first_order_ei(v.detach(), g_a.detach(), eta)
    return eis


def ranking_pairs(ei: Dict[str, float]) -> List[tuple]:
    """Enumerate unordered action pairs (a,b) with a<b and different EI."""
    acts = list(ACTIONS)
    out = []
    for i, a in enumerate(acts):
        for b in acts[i + 1 :]:
            if ei[a] != ei[b]:
                out.append((a, b, 1.0 if ei[a] > ei[b] else -1.0))
    return out
