"""
Schematic training loop for EI-MA (not end-to-end runnable).

Illustrates:
  - curriculum stages
  - periodic EI label refresh
  - one mini-batch + one eligible-pair augmentation update
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from config import EIMAConfig
from training.curriculum import current_stage


def train_one_step(
    model,
    optimizer: torch.optim.Optimizer,
    batch_kge_scores: torch.Tensor,
    batch_kge_labels: torch.Tensor,
    pair_pack: dict,
    step: int,
    epoch: int,
    cfg: EIMAConfig,
    ei_cache: Optional[Dict[str, float]] = None,
    probe_loss_fn=None,
):
    """
    pair_pack keys (schematic):
      pair: (i, j, dist)
      anchor: (s, r, o, A)
      C_un: list[int]
      facts_i, facts_j
      nbr_i, nbr_j  (tensors [K,d])
    """
    from training.losses import kge_1n_loss

    stage = current_stage(epoch, cfg)
    model.train()
    optimizer.zero_grad()

    L_kge = kge_1n_loss(batch_kge_scores, batch_kge_labels)

    # Stage 1: backbone only
    if stage.name == "warmup_kge":
        L_kge.backward()
        optimizer.step()
        return {"loss": float(L_kge.item()), "stage": stage.name, "action": None}

    need_ei = (
        stage.enable_selector
        and (ei_cache is None or step % cfg.ei_refresh_every == 0)
    )
    out = model.forward_pair_update(
        pair=pair_pack["pair"],
        anchor=pair_pack["anchor"],
        C_un=pair_pack["C_un"],
        facts_i=pair_pack["facts_i"],
        facts_j=pair_pack["facts_j"],
        nbr_states_i=pair_pack["nbr_i"],
        nbr_states_j=pair_pack["nbr_j"],
        L_kge=L_kge,
        ei_cache=None if need_ei else ei_cache,
        step=step,
        force_recon=stage.force_recon,
        enable_selector=stage.enable_selector,
        probe_loss_fn=probe_loss_fn,
    )
    out["loss"].backward()
    optimizer.step()
    out["stage"] = stage.name
    out["loss"] = float(out["loss"].item())
    return out


@torch.no_grad()
def static_infer_entity(backbone, eid: int) -> torch.Tensor:
    """Test-time: fused static embedding, no selector / augmentation."""
    e = backbone.entity_embedding(torch.tensor(eid))
    h = backbone.agent_state.weight[eid]
    return backbone.fuse(torch.cat([e, h.detach()], dim=-1))
