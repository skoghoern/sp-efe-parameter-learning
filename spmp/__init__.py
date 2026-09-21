"""SP-MP: sum-product message passing for EFE planning and parameter learning.

The paper's family term. Both planners here are SP-MP schemes on the same factor
graph, differing only in which entropy corrections they carry:

``sp_efe.sp_efe_planning``
    SP-EFE and all four of its ablations. The variants are the *same* function
    under different static flags; ``coupled_planner_kwargs`` maps a
    ``coupled-*`` planner name onto ``dyn_channel`` / ``dyn_tilt_coef`` /
    ``obs_tilt_coef``.

``sp_pref.sp_pref_planning``
    SP-Pref and SP-Pref-Plan, separated by ``planning_correction``.

``dispatch._plan`` selects between them. Where to read, in order:

    sweeps.py     the factor graph and one sum-product sweep over it
    tables.py     Dirichlet expected-log tables, the θ interface
    info_gain.py  Δ_A, Δ_B -- the per-entry e^Δ ≥ 1 information-gain weights
    sp_efe.py     SP-EFE: the two kernels, the channel fixed point, entry point
    sp_pref.py    the preference-only baselines, same skeleton minus the tilts

The filter, the conjugate count update and the episode loop are deliberately not
here: they are shared by every planner and live in ``agent.py``.

The planner *strings* on the command line (``coupled-mp``, ``pref-sp``) predate
the paper's labels and are keyed into the saved ``.npz`` arrays, so they cannot
be renamed; see the mapping table in the README.
"""

from .sweeps import (
    LOG_ZERO,
    backward_pass,
    damp_log_channel,
    forward_pass,
    safe_log,
)
from .tables import dirichlet_log_tables
from .info_gain import dirichlet_posterior_kl
from .sp_efe import (
    COUPLED_IGC_PLANNERS,
    COUPLED_PLANNERS,
    coupled_planner_kwargs,
    evaluate_single_objective,
    single_objective_terms,
    sp_efe_planning,
)
from .sp_pref import sp_pref_planning

__all__ = [
    "COUPLED_IGC_PLANNERS",
    "COUPLED_PLANNERS",
    "LOG_ZERO",
    "backward_pass",
    "coupled_planner_kwargs",
    "damp_log_channel",
    "dirichlet_log_tables",
    "dirichlet_posterior_kl",
    "evaluate_single_objective",
    "forward_pass",
    "sp_pref_planning",
    "safe_log",
    "single_objective_terms",
    "sp_efe_planning",
]
