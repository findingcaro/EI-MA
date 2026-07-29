"""
EI-MA core: one training update for an eligible subject pair.

Flow (Figure 2 / Method):
  pairing → retrieve → (optional) communicate → EI selector → operator loss → backbone
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from config import ACTIONS, EIMAConfig
from models.communication import GatedCommunicator, PairInteraction, RelationDecoder
from models.operators import AugmentationOperators, unobserved_candidates
from models.retrieval import FactEncoder, LocalRetriever, build_selector_input
from models.selector import OperatorSelector, ranking_loss
from training.curriculum import selector_temperature
from training.ei_label import estimate_action_eis
from training.losses import generated_loss, total_loss


class EIMA(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_ent: int,
        num_rel: int,
        cfg: EIMAConfig,
        anyburl_best=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        dim = cfg.embed_dim

        self.fact_enc = FactEncoder(num_ent, num_rel, dim)
        self.retriever = LocalRetriever(dim, M=cfg.retrieve_M)
        self.comm = GatedCommunicator(dim, d_lr=cfg.low_rank_dim)

        # u_ij dim = 5d + d_phi
        d_phi = 16
        u_dim = 5 * dim + d_phi
        self.interaction = PairInteraction(u_dim, dim, z_dim=dim)
        self.decoder = RelationDecoder(dim, num_rel)
        self.selector = OperatorSelector(u_dim)
        self.ops = AugmentationOperators(
            score_fn=self._score,
            hardneg_margin=cfg.hardneg_margin,
            hardneg_beta=cfg.hardneg_beta,
            anyburl_best=anyburl_best,
        )
        self.d_phi = d_phi

    def _score(self, s, r, o, A=None):
        if not torch.is_tensor(s):
            s = torch.tensor(s)
            r = torch.tensor(r)
            o = torch.tensor(o)
        return self.backbone.score(s, r, o, A)

    def pair_features(
        self,
        i: int,
        j: int,
        dist: int,
        facts_i: Sequence,
        facts_j: Sequence,
        neighbor_states_ij: torch.Tensor,
        neighbor_states_ji: torch.Tensor,
        device,
    ):
        """Build u_ij (and optionally communicate)."""
        h_i = self.backbone.agent_state.weight[i]
        h_j = self.backbone.agent_state.weight[j]
        dist_t = torch.tensor([dist], device=device)

        g_i = self.retriever.query(h_i, h_j, dist_t).squeeze(0)
        g_j = self.retriever.query(h_j, h_i, dist_t).squeeze(0)

        # Encode local facts (schematic: facts as (s,r,o))
        def encode_facts(facts):
            if len(facts) == 0:
                return torch.zeros(0, self.cfg.embed_dim, device=device)
            embs = []
            for s, r, o, *_ in facts:
                embs.append(
                    self.fact_enc(
                        torch.tensor(s, device=device),
                        torch.tensor(r, device=device),
                        torch.tensor(o, device=device),
                    )
                )
            return torch.stack(embs, dim=0)

        fi = encode_facts(facts_i)
        fj = encode_facts(facts_j)
        r_i, _ = self.retriever.summarize(g_i, fi)
        r_j, _ = self.retriever.summarize(g_j, fj)
        phi = self.retriever.dist_emb(dist_t.clamp(max=7)).squeeze(0)
        u_ij = build_selector_input(h_i, h_j, r_i, r_j, phi)
        return u_ij, h_i, h_j

    def communicate(self, h_i, h_j, u_ij, nbr_i, nbr_j):
        m_ij = self.comm.message(h_i, h_j, nbr_i)
        m_ji = self.comm.message(h_j, h_i, nbr_j)
        z_ij = self.interaction(u_ij, m_ij, m_ji)
        return z_ij

    def forward_pair_update(
        self,
        pair,
        anchor,
        C_un: List[int],
        facts_i,
        facts_j,
        nbr_states_i: torch.Tensor,
        nbr_states_j: torch.Tensor,
        L_kge: torch.Tensor,
        ei_cache: Optional[Dict[str, float]],
        step: int,
        force_recon: bool = False,
        enable_selector: bool = True,
        probe_loss_fn=None,
    ):
        """
        One EI-MA update for eligible pair (i,j) with shared anchor q0.
        Returns total loss and diagnostics.
        """
        i, j, dist = pair
        device = L_kge.device
        u_ij, h_i, h_j = self.pair_features(
            i, j, dist, facts_i, facts_j, nbr_states_i, nbr_states_j, device
        )

        # --- decide action ---
        tau = selector_temperature(step, self.cfg)
        if force_recon:
            action = "Recon"
            L_pi = L_kge.new_tensor(0.0)
            ei = ei_cache or {a: 0.0 for a in ACTIONS}
        elif not enable_selector:
            action = "Skip"
            L_pi = L_kge.new_tensor(0.0)
            ei = ei_cache or {a: 0.0 for a in ACTIONS}
        else:
            # refresh / use EI labels
            if ei_cache is None and probe_loss_fn is not None:
                ei = estimate_action_eis(
                    probe_loss_fn=probe_loss_fn,
                    aug_loss_fn=lambda a: self.ops.loss(a, anchor, C_un),
                    shared_params=self.backbone.shared_parameters(),
                    eta=self.cfg.ei_step_size,
                )
            else:
                ei = ei_cache or {a: 0.0 for a in ACTIONS}

            U = self.selector.utilities(u_ij)
            L_pi = ranking_loss(U, ei, margin=self.cfg.ranking_margin)
            # sample with stopped discrete route (no PG through action)
            with torch.no_grad():
                action = self.selector.sample(u_ij, tau)

        # --- communicate only if non-skip ---
        if action != "Skip":
            z_ij = self.communicate(h_i, h_j, u_ij, nbr_states_i, nbr_states_j)
            rel = torch.tensor(anchor[1], device=device)
            L_dec = self.decoder.loss(z_ij, rel)
        else:
            L_dec = L_kge.new_tensor(0.0)

        L_aug = self.ops.loss(action, anchor, C_un)
        L_gen = generated_loss(action, L_dec, L_aug)
        loss = total_loss(
            L_kge, L_gen, L_pi, self.cfg.lambda_g, self.cfg.lambda_pi
        )
        return {
            "loss": loss,
            "action": action,
            "L_kge": L_kge.detach(),
            "L_gen": L_gen.detach(),
            "L_pi": L_pi.detach(),
            "ei": ei,
        }
