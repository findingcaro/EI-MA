"""Three-stage curriculum for EI-MA."""

from __future__ import annotations

from dataclasses import dataclass

from config import EIMAConfig


@dataclass
class CurriculumStage:
    name: str
    force_recon: bool
    enable_selector: bool


def current_stage(epoch: int, cfg: EIMAConfig) -> CurriculumStage:
    """
    (1) Warm up StarE with L_KGE only
    (2) Force Recon on sampled pairs to init local modules
    (3) Enable all four actions under π with periodic EI refresh
    """
    if epoch < cfg.warmup_epochs:
        return CurriculumStage("warmup_kge", force_recon=False, enable_selector=False)
    if epoch < cfg.warmup_epochs + cfg.recon_force_epochs:
        return CurriculumStage("force_recon", force_recon=True, enable_selector=False)
    return CurriculumStage("full_ei", force_recon=False, enable_selector=True)


def selector_temperature(step: int, cfg: EIMAConfig) -> float:
    """τ annealed from start → end over ~200 steps (paper)."""
    t = min(1.0, step / max(cfg.selector_anneal_steps, 1))
    return cfg.selector_tau_start + t * (cfg.selector_tau_end - cfg.selector_tau_start)
