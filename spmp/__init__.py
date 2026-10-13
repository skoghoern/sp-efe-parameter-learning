"""Sum-product message passing for EFE planning.

Every planner is ``sp_efe_planning`` under a ``Scheme`` (``paper_scheme``,
``PAPER_ROWS``). ``dispatch._plan`` adds the enumerated EFE.

    sweeps.py     factor graph and one sum-product sweep
    tables.py     Dirichlet expected-log tables
    info_gain.py  per-entry information gains Δ_A, Δ_B
    sp_efe.py     kernels, channel fixed point, entry point

The filter, count update, and episode loop live in ``agent.py``.
"""

from .sweeps import (
    LOG_ZERO,
    backward_pass,
    damp_log_channel,
    forward_pass,
    is_live,
    log_normalize,
    safe_log,
)
from .tables import dirichlet_log_tables
from .info_gain import dirichlet_posterior_kl, uniform_over_valid
from .sp_efe import (
    PAPER_ROWS,
    SP_EFE_IGC_PLANNERS,
    SP_EFE_PLANNERS,
    SP_PREF_PLANNERS,
    PlanningProblem,
    PlanResult,
    Scheme,
    SweepState,
    build_problem,
    converge_obs_channel,
    dynamics_kernel,
    information_gain_B,
    initial_state,
    objective_terms,
    observation_kernel,
    paper_scheme,
    refresh_channels,
    sp_efe_planning,
    sweep,
)

__all__ = [
    # planner rows
    "PAPER_ROWS", "SP_EFE_IGC_PLANNERS", "SP_EFE_PLANNERS", "SP_PREF_PLANNERS",
    "Scheme", "paper_scheme",
    # planner and its parts, in reading order
    "sp_efe_planning", "PlanResult", "PlanningProblem", "SweepState",
    "build_problem", "initial_state", "sweep",
    "observation_kernel", "dynamics_kernel", "converge_obs_channel",
    "information_gain_B", "refresh_channels", "objective_terms",
    # primitives
    "LOG_ZERO", "is_live", "log_normalize", "safe_log",
    "forward_pass", "backward_pass", "damp_log_channel",
    "dirichlet_log_tables", "dirichlet_posterior_kl", "uniform_over_valid",
]
