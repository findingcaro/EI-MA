"""EI-MA core hyperparameters (aligned with the paper settings table)."""

from dataclasses import dataclass


ACTIONS = ("Skip", "Recon", "Struct-Pos", "Hard-Neg")


@dataclass
class EIMAConfig:
    # Backbone / training (shared with StarE-Base protocol)
    embed_dim: int = 256
    batch_size: int = 32
    epochs: int = 300
    lr: float = 1e-3

    # Subject-graph constraints
    K_A: int = 20                 # mutual top-K neighbors on G_A
    max_distance: int = 2         # eligible if d_A(i,j) <= 2

    # Local retrieval / communication
    retrieve_M: int = 16          # top-M local facts
    K_nbr: int = 12               # common-neighbor cap
    low_rank_dim: int = 32        # d_lr communication bottleneck

    # EI supervision
    probe_frac: float = 0.01      # |H| ≈ 1% of train
    ei_refresh_every: int = 100   # refresh EI labels every N steps
    ei_step_size: float = 1e-3    # eta in first-order EI
    selector_tau_start: float = 0.4
    selector_tau_end: float = 0.7
    selector_anneal_steps: int = 200
    ranking_margin: float = 0.1   # mu in L_pi

    # Operator losses
    hardneg_margin: float = 1.0   # m in softplus(m - delta)
    hardneg_beta: float = 1.0     # beta in confidence weight
    lambda_g: float = 0.1
    lambda_pi: float = 1.0

    # Curriculum lengths (in epochs; illustrative)
    warmup_epochs: int = 30
    recon_force_epochs: int = 20
