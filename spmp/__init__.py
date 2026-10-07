"""SP-MP: sum-product message passing for EFE planning and parameter learning.

The paper's family term. Every planner here is one SP-MP scheme on the same
factor graph; the rows of the paper's table differ only in which entropy
corrections they carry:

``sp_efe.sp_efe_planning``
    SP-EFE, its four ablations, and the preference-only baselines SP-Pref /
    SP-Pref-Plan. All are the *same* function under a different ``Scheme``;
    ``paper_scheme(label)`` gives the paper's rows (``PAPER_ROWS``). Returns a
    ``PlanResult`` with q(u_1), the planned beliefs, the final kernel and
    channels, and optionally the objective trace term by term.

``dispatch._plan`` adds the enumerated EFE. Where to read, in order:

    sweeps.py     the factor graph and one sum-product sweep over it
    tables.py     Dirichlet expected-log tables, the θ interface
    info_gain.py  Δ_A, Δ_B -- the per-entry e^Δ ≥ 1 information-gain weights
    sp_efe.py     the two kernels, the channel fixed point, entry point, flags

The filter, the conjugate count update and the episode loop are deliberately not
here: they are shared by every planner and live in ``agent.py``.

The planner strings on the command line are the paper's labels (``SP-EFE``,
``SP-Pref``, ...) and are the keys of the saved ``.npz`` arrays. See the
table in the README.
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
