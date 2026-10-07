"""SP-EFE planning by sum-product message passing.

Each entropy correction of the expected free energy is a channel in a factor
kernel. Likelihood tables are posterior means, log(α / α₀):

    log g_obs(y, x)     = log m_A + log r(y|x) + c · Δ_A              observation_kernel
    log g_dyn(x', x, u) = log m_B + Δ_B + log r(u|x) − log r(x'|x, u)   dynamics_kernel

Δ_A and Δ_B are the per-entry information gains (``info_gain.py``). The
channels are the closed-form conditionals of q, refreshed after each sweep
(``refresh_channels``):

    r(y|x)      ambiguity
    r(u|x)      planning correction, Σ H[q(u|x)]
    r(x'|x, u)  state information gain

Uniform channels recover standard Bethe belief propagation, which is the
preference-only baseline SP-Pref; SP-Pref-Plan keeps only r(u|x). Every row
of the paper's table is ``sp_efe_planning`` under a ``Scheme``; ``PAPER_ROWS``
lists them. q(θ) stays frozen during a plan; the count update is in
``agent.py``. ``demo_sweep.py`` prints every quantity below on a five-cell
corridor and is the place to start.

Notation -- axis order is (T, x', x, u) everywhere, as in ``sweeps.py`` and
``info_gain.py``:

    x_t ∈ {1..S}   state; x = x_{t-1} conditions, x' = x_t is the successor
    u_t ∈ {1..U}   action
    y   ∈ {1..C}   outcome of observation factor f ∈ {1..F}
    T              planning horizon; t = 0 is the current state
    m_A, m_B       posterior-mean tables     log_m_A (F, C, S)   log_m_B (S, S, U)
    r(y|x)         ambiguity channel         log_r_yx (F, C, S)
    r(u|x)         planning channel          log_r_ux (T, S, U)
    r(x'|x, u)     state-IG channel          log_r_xxu (T, S, S, U)
    Δ_A, Δ_B       information gains         delta_A (F, C, S)   delta_B (T, S, S, U)
    n̄_B            fractional counts         nbar_B (T, S, S, U)
    μ→, μ←         forward/backward msgs     log_fwd, log_bwd (T+1, S)
    φ_t(x)         unary factor at node t    log_unary_* (T+1, S): obs→x message,
                                             plus the preference when per-step

Computation is in log-space; LOG_ZERO marks structural zeros (invariant in
``sweeps.py``).
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float

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


def weighted_log(q, log_v):
    """q · log_v elementwise with the 0·log 0 = 0 convention.

    Not scipy's ``xlogy``: the log is already taken. Beliefs are exact zeros
    wherever a kernel entry is LOG_ZERO, so masking on q > 0 drops the
    spurious q·(−1e12) products without touching any live term.
    """
    return jnp.where(q > 0.0, q * log_v, 0.0)


# -----------------------------------------------------------------------------
# Which corrections a planner keeps
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Scheme:
    """The corrections a planner keeps (paper Table 1). Defaults are SP-EFE.

    Each flag removes one correction from the kernels; ``paper_scheme`` maps
    a paper label onto a ``Scheme``. Hashable, so it is a static jit argument.

    ``goal_in_sweeps`` is not a correction. It exists to keep the
    preference-only baselines as published: with no θ-channel to carry the
    per-step preference into the chain, SP-Pref and SP-Pref-Plan put it in the
    forward/backward sweeps directly. SP-EFE keeps its sweeps observation-only
    and lets the preference enter through the channel refresh.
    """
    ambiguity: bool = True             # r(y|x) in g_obs
    state_info_gain: bool = True       # r(x'|x,u) in g_dyn
    planning_correction: bool = True   # r(u|x) in g_dyn
    novelty_B: bool = True             # Δ_B in g_dyn
    novelty_A_weight: int = 2          # c on Δ_A in g_obs; 0 drops it
    goal_in_sweeps: bool = False       # per-step preference also in the sweeps

    def __post_init__(self):
        if self.novelty_A_weight and not self.ambiguity:
            raise ValueError("novelty_A needs the ambiguity channel: Δ_A is "
                             "built from the r(y|x) fixed point")

    @property
    def refreshes_a_channel(self) -> bool:
        """Whether anything in g_dyn is refreshed from the chain beliefs."""
        return self.state_info_gain or self.planning_correction or self.novelty_B


# Paper label -> (scheme, description), in the paper's row order. The README
# table lists the same labels; a test keeps them in step.
PAPER_ROWS: dict[str, tuple[Scheme, str]] = {
    "SP-EFE-dyn": (Scheme(state_info_gain=False),
                   "no dynamics channel"),
    "SP-EFE": (Scheme(),
               "full scheme: obs + dyn channels, Δ_A, Δ_B"),
    "SP-EFE-DeltaB": (Scheme(novelty_B=False),
                      "dynamics channel on, Δ_B off"),
    "SP-EFE-DeltaA-DeltaB-dyn": (Scheme(state_info_gain=False, novelty_B=False,
                                        novelty_A_weight=0),
                                 "channels only, dynamics channel dropped too"),
    "SP-EFE-DeltaA-DeltaB": (Scheme(novelty_B=False, novelty_A_weight=0),
                             "channels only (ambiguity, no novelty)"),
    "SP-Pref-Plan": (Scheme(ambiguity=False, state_info_gain=False,
                            novelty_B=False, novelty_A_weight=0,
                            goal_in_sweeps=True),
                     "planning correction r(u|x) only"),
    "SP-Pref": (Scheme(ambiguity=False, state_info_gain=False,
                       planning_correction=False, novelty_B=False,
                       novelty_A_weight=0, goal_in_sweeps=True),
                "no corrections"),
}
SP_EFE_PLANNERS = ("SP-EFE-dyn", "SP-EFE", "SP-EFE-DeltaB")
SP_EFE_IGC_PLANNERS = ("SP-EFE-DeltaA-DeltaB-dyn", "SP-EFE-DeltaA-DeltaB")
SP_PREF_PLANNERS = ("SP-Pref", "SP-Pref-Plan")


def paper_scheme(label: str) -> Scheme:
    """The ``Scheme`` of a paper planner label."""
    try:
        return PAPER_ROWS[label][0]
    except KeyError:
        raise ValueError(f"unknown planner {label!r}; "
                         f"expected one of {list(PAPER_ROWS)}") from None


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


class PlanningProblem(NamedTuple):
    """Everything fixed for the duration of one plan. Built by ``build_problem``."""
    scheme: Scheme
    horizon: int
    per_step_goal: bool
    damping: float
    alpha_B: Float[Array, "S S U"]              # for Δ_B
    log_m_B: Float[Array, "S S U"]
    valid_B: Bool[Array, "S S U"]               # support of m_B
    log_action_prior: Float[Array, "U"]
    log_q0: Float[Array, "S"]
    log_goal: Float[Array, "S"]
    log_unary_sweep: Float[Array, "T+1 S"]      # φ_t seen by the sweeps
    log_unary_refresh: Float[Array, "T+1 S"]    # φ_t seen by refresh and objective
    obs_energy_x: Float[Array, "S"]             # Σ_{f,y} q(y|x) log m_A
    obs_info_x: Float[Array, "S"]               # Σ_{f,y} q(y|x) Δ_A


class SweepState(NamedTuple):
    """What one sweep refreshes."""
    log_r_xxu: Float[Array, "T S S U"]   # r(x'|x,u)
    log_r_ux: Float[Array, "T S U"]      # r(u|x)
    nbar_B: Float[Array, "T S S U"]      # n̄_B = q_t(x',u|x), count space
    log_fwd: Float[Array, "T+1 S"]       # messages of the last sweep, for damping
    log_bwd: Float[Array, "T+1 S"]


class PlanResult(NamedTuple):
    action_dist: Float[Array, "U"]                # q(u_1), the planner's output
    q_u: Float[Array, "T U"]                      # q(u_t) for every planned step
    q_x: Float[Array, "T+1 S"]                    # q(x_t) at the fixed point
    log_g_dyn: Float[Array, "T S S U"]            # kernel implied by the final channels
    state: SweepState
    objective_trace: dict[str, Float[Array, "n_iterations"]] | None  # per term + "total"


# -----------------------------------------------------------------------------
# The two kernels and the channel refresh
# -----------------------------------------------------------------------------


def observation_kernel(
    log_m_A: Float[Array, "F C S"],
    log_r_yx: Float[Array, "F C S"] | float,  # scalar broadcasts; 0 without r(y|x)
    delta_A: Float[Array, "F C S"] | float,   # scalar broadcasts; 0 without Δ_A
    novelty_A_weight: float,
    salience=0.0,
) -> Float[Array, "F C S"]:
    """log g_obs = log m_A + log r(y|x) + c·Δ_A (+ optional salience tilt)."""
    log_g = log_m_A + log_r_yx + novelty_A_weight * delta_A + salience
    return jnp.where(is_live(log_m_A), log_g, LOG_ZERO)


def dynamics_kernel(
    log_m_B: Float[Array, "S S U"],
    delta_B,
    log_r_ux: Float[Array, "T S U"],
    log_r_xxu: Float[Array, "T S S U"],
) -> Float[Array, "T S S U"]:
    """log g_dyn = log m_B + Δ_B + log r(u|x) − log r(x'|x,u).

    A collapsed channel entry (for example x' ≠ goal on the last step) is
    clipped out rather than amplifying the kernel through −log r → +∞.
    Structural zeros are already LOG_ZERO in log_m_B. ``delta_B`` may be the
    scalar 0.0 when Δ_B is dropped.
    """
    ok = is_live(log_r_xxu)
    log_g = log_m_B + delta_B + log_r_ux[:, None] - jnp.where(ok, log_r_xxu, 0.0)
    return jnp.where(ok, log_g, LOG_ZERO)


def information_gain_B(problem: PlanningProblem, nbar_B) -> Float[Array, "T S S U"]:
    """Δ_B from the current fractional counts; 0 on structural zeros."""
    delta = dirichlet_posterior_kl(problem.alpha_B[None], nbar_B,
                                   axis=1, cond_axes=(0, 2))
    return jnp.where(problem.valid_B, delta, 0.0)


def converge_obs_channel(alpha_A, log_m_A, scheme: Scheme, n_iterations, damping,
                         salience=0.0):
    """Observation block: μ_obs→x, the conditional q(y|x), and Δ_A.

    r(y|x) conditions on x only, so it does not depend on the chain and is
    converged once. At the fixed point r(y|x) = q(y|x), and exponentiated it
    is the fractional count n̄_A in Δ_A = KL(Dir(α+e_y) ‖ Dir(α+n̄)). Without
    the ambiguity channel (SP-Pref) the kernel is m_A·e^salience, q(y|x) is
    its normalization and Δ_A = 0. p̂(y) is uniform.
    """
    valid_A = is_live(log_m_A)

    if not scheme.ambiguity:
        log_g = observation_kernel(log_m_A, 0.0, 0.0, 0.0, salience)
        log_q_yx = jnp.where(valid_A, log_normalize(log_g, axis=1), LOG_ZERO)
        delta_A = jnp.zeros_like(log_m_A)
    else:
        def kernel(log_r_yx):
            nbar = jnp.where(valid_A, jnp.exp(log_r_yx), 0.0)
            delta = jnp.where(valid_A, dirichlet_posterior_kl(
                alpha_A, nbar, axis=1, cond_axes=(0, 2)), 0.0)
            return observation_kernel(log_m_A, log_r_yx, delta,
                                      scheme.novelty_A_weight, salience), delta

        def refresh(_, log_r_yx):
            log_g, _ = kernel(log_r_yx)
            return damp_log_channel(log_r_yx, log_normalize(log_g, axis=1),
                                    damping, cond_axis=1)

        log_q_yx = lax.fori_loop(0, n_iterations, refresh,
                                 safe_log(uniform_over_valid(valid_A, axes=1)))
        log_g, delta_A = kernel(log_q_yx)

    # μ_obs→x = Π_f Σ_y g_obs(y, x) with uniform p̂(y)
    log_obs_to_x = logsumexp(log_g, axis=1).sum(axis=0)
    return log_normalize(log_obs_to_x, axis=0), log_q_yx, delta_A


def refresh_channels(problem: PlanningProblem, state: SweepState,
                     log_g_dyn, log_bwd) -> SweepState:
    """Set each channel to its closed-form conditional of the block belief.

    The block belief conditioned on x is m(x', u | x) ∝ g_dyn · μ←(x') · φ(x')
    · p(u). All three refreshed quantities condition on x, so the forward
    message cancels and is left out -- a LOG_ZERO forward entry would swamp
    the slice in float32. Channels are damped geometrically, n̄_B linearly in
    count space. A dropped channel keeps its uniform initialisation.
    """
    scheme, damping = problem.scheme, problem.damping
    bwd_unary = log_bwd[1:] + problem.log_unary_refresh[1:]        # (T, x')
    log_block = log_g_dyn + bwd_unary[:, :, None, None] + problem.log_action_prior
    log_pair = logsumexp(log_block, axis=1)                          # (T, x, u)
    joint = jnp.where(problem.valid_B, jax.nn.softmax(log_block, axis=(1, 3)), 0.0)

    log_r_xxu = state.log_r_xxu
    if scheme.state_info_gain:        # r(x'|x,u) ← norm_{x'} m
        log_r_xxu = damp_log_channel(log_r_xxu, log_normalize(log_block, axis=1),
                                     damping, cond_axis=1)
    log_r_ux = state.log_r_ux
    if scheme.planning_correction:    # r(u|x) ← norm_u Σ_{x'} m
        log_r_ux = damp_log_channel(log_r_ux, log_normalize(log_pair, axis=2),
                                    damping, cond_axis=2)
    nbar_B = state.nbar_B
    if scheme.novelty_B:              # n̄_B ← q_t(x',u|x)
        nbar_B = (1.0 - damping) * nbar_B + damping * joint
    return SweepState(log_r_xxu, log_r_ux, nbar_B, state.log_fwd, state.log_bwd)


# -----------------------------------------------------------------------------
# The objective, term by term
# -----------------------------------------------------------------------------


def objective_terms(problem: PlanningProblem, log_g_dyn, delta_B,
                    log_fwd, log_bwd) -> dict[str, Float[Array, ""]]:
    """Named terms of the planning objective at the current messages.

    Their sum is the objective the scheme is built to minimize. Beliefs are
    rebuilt from the passed messages. ``x_entropy`` is the net +H[q(x_t)] left
    at interior chain states after the observation-block entropies cancel.

    The trace is a diagnostic, not a descent certificate: with damped
    messages and clipped channels the sum is not monotone in the sweep count,
    and its fixed point depends on the damping.
    """
    unary = problem.log_unary_refresh
    horizon = problem.horizon

    # State beliefs q(x_t) ∝ μ→·μ←·φ, one per chain node
    log_qx = log_normalize(log_fwd + log_bwd + unary, axis=1)
    qx = jnp.exp(log_qx)

    # Dyn block beliefs q_t(x', x, u) ∝ g_dyn·μ→(x)φ(x)·μ←(x')φ(x')·p(u)
    fwd_unary = log_fwd[:-1] + unary[:-1]  # (T, x)
    bwd_unary = log_bwd[1:] + unary[1:]    # (T, x')
    log_qt = log_normalize(
        log_g_dyn + bwd_unary[:, :, None, None] + fwd_unary[:, None, :, None]
        + problem.log_action_prior, axis=(1, 2, 3))
    qt = jnp.exp(log_qt)
    log_qpair = logsumexp(log_qt, axis=1)  # (T, x, u), normalized
    q_u = qt.sum(axis=(1, 2))              # (T, U)

    if problem.per_step_goal:
        goal_energy = -weighted_log(qx[1:], problem.log_goal).sum()   # x_1..x_T
    else:
        goal_energy = -weighted_log(qx[horizon], problem.log_goal).sum()

    return {
        # dyn blocks: −H[q_t] − E[log m_B] and the three corrections
        "dyn_neg_entropy": weighted_log(qt, log_qt).sum(),
        "dyn_energy": -weighted_log(qt, problem.log_m_B).sum(),
        # −H[q(x'|x,u)]: the state-information-gain channel, from q_t itself
        "dyn_state_entropy": weighted_log(qt, log_normalize(log_qt, axis=1)).sum(),
        # +H[q(u|x)]: the planning correction
        "dyn_plan_entropy": -weighted_log(jnp.exp(log_qpair),
                                          log_normalize(log_qpair, axis=2)).sum(),
        "dyn_info": -(qt * delta_B).sum(),                           # −E[Δ_B]
        "dyn_action_energy": -weighted_log(q_u, problem.log_action_prior).sum(),
        # obs blocks: −E[log m_A] − c·E[Δ_A]; their entropies → x_entropy
        "obs_energy": -(qx * problem.obs_energy_x).sum(),
        "obs_info": -problem.scheme.novelty_A_weight * (qx * problem.obs_info_x).sum(),
        # +H[q(x_t)] at interior chain states only
        "x_entropy": -weighted_log(qx[1:horizon], log_qx[1:horizon]).sum(),
        "goal_energy": goal_energy,
        "init_energy": -weighted_log(qx[0], problem.log_q0).sum(),
    }


# -----------------------------------------------------------------------------
# One sweep, and the planner
# -----------------------------------------------------------------------------


def sweep(problem: PlanningProblem, state: SweepState, track_objective: bool):
    """Assemble g_dyn, run forward/backward, refresh the channels.

    Returns the new state and the per-sweep outputs ``(q_u, terms)`` that
    ``lax.scan`` stacks; ``terms`` is None unless ``track_objective``.
    """
    delta_B = information_gain_B(problem, state.nbar_B) if problem.scheme.novelty_B else 0.0
    log_g_dyn = dynamics_kernel(problem.log_m_B, delta_B, state.log_r_ux, state.log_r_xxu)

    log_fwd = forward_pass(
        log_g_dyn, problem.log_q0, problem.log_action_prior,
        problem.log_unary_sweep, problem.horizon,
        log_prev_fwd=state.log_fwd, msg_damping=problem.damping)
    log_bwd, q_u = backward_pass(
        log_g_dyn, log_fwd, problem.log_goal, problem.log_action_prior,
        problem.log_unary_sweep, problem.horizon,
        log_prev_bwd=state.log_bwd, msg_damping=problem.damping)

    new_state = refresh_channels(problem, state, log_g_dyn, log_bwd)
    new_state = new_state._replace(log_fwd=log_fwd, log_bwd=log_bwd)
    terms = (objective_terms(problem, log_g_dyn, delta_B, log_fwd, log_bwd)
             if track_objective else None)
    return new_state, (q_u, terms)


def _check_inputs(q_current_state, alpha_A, alpha_B, goal, action_prior,
                  horizon, scheme, per_step_goal):
    """Trace-time validation; shapes are static so this costs nothing."""
    n_states = q_current_state.shape[0]
    n_actions = alpha_B.shape[-1]
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if per_step_goal and not (scheme.refreshes_a_channel or scheme.goal_in_sweeps):
        raise ValueError("per_step_goal has no channel to enter through; "
                         "set Scheme(goal_in_sweeps=True)")
    if alpha_B.shape[:2] != (n_states, n_states):
        raise ValueError(
            f"alpha_B must be (S, S, U) with S={n_states}, got {alpha_B.shape}")
    if alpha_A.ndim != 3 or alpha_A.shape[-1] != n_states:
        raise ValueError(
            f"alpha_A must be (F, C, S) with S={n_states}, got {alpha_A.shape}")
    if goal.shape != (n_states,):
        raise ValueError(f"goal must be (S,)={n_states}, got {goal.shape}")
    if action_prior is not None and action_prior.shape != (n_actions,):
        raise ValueError(
            f"action_prior must be (U,)={n_actions}, got {action_prior.shape}")


def build_problem(q_current_state, alpha_A, alpha_B, goal, *, horizon,
                  n_iterations, scheme, damping, action_prior, obs_salience,
                  per_step_goal) -> PlanningProblem:
    """Tables, the converged observation block, and the unary factors."""
    n_states, n_actions = alpha_B.shape[1:]
    _, log_m_A = dirichlet_log_tables(alpha_A, axis=1)
    _, log_m_B = dirichlet_log_tables(alpha_B, axis=0)
    log_goal = safe_log(goal)

    log_obs_to_x, log_q_yx, delta_A = converge_obs_channel(
        alpha_A, log_m_A, scheme, n_iterations, damping,
        salience=0.0 if obs_salience is None else obs_salience)
    q_yx = jnp.where(is_live(log_q_yx), jnp.exp(log_q_yx), 0.0)

    # φ_t: the obs→x message at every node; the per-step preference on
    # t ≥ 1 is seen by the refresh, and by the sweeps only if the scheme says so
    log_unary = jnp.broadcast_to(log_obs_to_x, (horizon + 1, n_states))
    log_unary_refresh = (log_unary.at[1:].add(log_goal) if per_step_goal
                         else log_unary)
    log_unary_sweep = log_unary_refresh if scheme.goal_in_sweeps else log_unary

    return PlanningProblem(
        scheme=scheme, horizon=horizon, per_step_goal=per_step_goal,
        damping=damping, alpha_B=alpha_B, log_m_B=log_m_B,
        valid_B=is_live(log_m_B),
        log_action_prior=safe_log(jnp.full(n_actions, 1.0 / n_actions)
                                  if action_prior is None else action_prior),
        log_q0=safe_log(q_current_state), log_goal=log_goal,
        log_unary_sweep=log_unary_sweep, log_unary_refresh=log_unary_refresh,
        obs_energy_x=weighted_log(q_yx, log_m_A).sum(axis=(0, 1)),
        obs_info_x=(q_yx * delta_A).sum(axis=(0, 1)),
    )


def initial_state(problem: PlanningProblem) -> SweepState:
    """Uniform channels (Bethe BP), admissible uniform n̄_B, zero messages."""
    T, (S, _, U) = problem.horizon, problem.log_m_B.shape
    return SweepState(
        log_r_xxu=jnp.full((T, S, S, U), -jnp.log(S)),
        log_r_ux=jnp.full((T, S, U), -jnp.log(U)),
        nbar_B=jnp.broadcast_to(uniform_over_valid(problem.valid_B, axes=(0, 2)),
                                (T, S, S, U)),
        log_fwd=jnp.zeros((T + 1, S)),
        log_bwd=jnp.zeros((T + 1, S)))


@partial(jax.jit, static_argnames=(
    "horizon", "n_iterations", "scheme", "per_step_goal", "track_objective"))
def sp_efe_planning(
    q_current_state: Float[Array, "S"],
    alpha_A: Float[Array, "F C S"],
    alpha_B: Float[Array, "S S U"],
    goal: Float[Array, "S"],
    *,
    horizon: int,
    n_iterations: int,
    scheme: Scheme = Scheme(),
    damping: float = 1.0,
    action_prior: Float[Array, "U"] | None = None,
    obs_salience: Float[Array, "F C S"] | None = None,
    per_step_goal: bool = False,
    track_objective: bool = False,
) -> PlanResult:
    """Plan on the POMDP chain; every row of the paper's table is a ``Scheme``.

    Args:
        q_current_state: current state belief.
        alpha_A, alpha_B: Dirichlet counts of the observation and transition
            models; the planner uses their posterior-mean tables.
        goal: state preference p̂(x); the terminal factor, and with
            ``per_step_goal`` also a unary factor on x_1..x_T.
        horizon: T, the number of planned transitions (static).
        n_iterations: sweeps; each is one forward/backward pass followed by a
            channel refresh (static).
        scheme: which corrections are kept; ``paper_scheme(label)`` for the
            paper's rows (static).
        damping: 1 = full update; 0.5 is used throughout the paper.
        action_prior: p(u); uniform if None.
        obs_salience: optional (F, C, S) log-space tilt on g_obs (IG_c / Δ_θ).
        per_step_goal: preference on every future state as well as the
            terminal one (static). See ``Scheme.goal_in_sweeps`` for how it
            enters the chain.
        track_objective: record ``objective_terms`` after every sweep (static;
            roughly doubles the cost).

    Example:
        >>> result = sp_efe_planning(q0, alpha_A, alpha_B, goal, horizon=5,
        ...                          n_iterations=50, damping=0.5,
        ...                          scheme=paper_scheme("SP-EFE"))
        >>> result.action_dist            # q(u_1)
        >>> result.q_x                    # (T+1, S) planned state beliefs
        >>> jnp.exp(result.state.log_r_ux)  # the policy channel r(u|x)
    """
    _check_inputs(q_current_state, alpha_A, alpha_B, goal, action_prior,
                  horizon, scheme, per_step_goal)
    problem = build_problem(
        q_current_state, alpha_A, alpha_B, goal, horizon=horizon,
        n_iterations=n_iterations, scheme=scheme, damping=damping,
        action_prior=action_prior, obs_salience=obs_salience,
        per_step_goal=per_step_goal)

    state, (q_u, terms) = lax.scan(
        lambda s, _: sweep(problem, s, track_objective),
        initial_state(problem), None, length=n_iterations)

    delta_B = information_gain_B(problem, state.nbar_B) if scheme.novelty_B else 0.0
    log_g_dyn = dynamics_kernel(problem.log_m_B, delta_B, state.log_r_ux, state.log_r_xxu)
    q_x = jnp.exp(log_normalize(state.log_fwd + state.log_bwd
                                + problem.log_unary_refresh, axis=1))
    trace = None if terms is None else {**terms, "total": sum(terms.values())}
    return PlanResult(q_u[-1, 0], q_u[-1], q_x, log_g_dyn, state, trace)
