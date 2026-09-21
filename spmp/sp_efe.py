"""SP-EFE: the paper's message passing scheme. Start here.

EFE-based planning runs as sum-product message passing on the POMDP chain, with
each entropy correction of the objective absorbed into a factor kernel as a
*channel reparameterization*. Two kernels carry the whole scheme:

    log g_obs(y, x)     = log Ã + log r(y|x)  + coef · Δ_A
    log g_dyn(x', x, u) = log B̃ + Δ_B + log r(u|x) − log r(x'|x, u)

where

    Ã, B̃    "+1" joint-average Dirichlet tables, ψ(α+1) − ψ(α_0+1)
    Δ_A, Δ_B per-entry information gain, KL(Dir(α+e) ‖ Dir(α+n̄)); as e^Δ ≥ 1
             these are the paper's information-gain weights (``info_gain.py``)
    r(y|x)   observation channel      -> the ambiguity correction
    r(u|x)   policy channel           -> the planning correction Σ H[q(u|x)]
    r(x'|x,u) dynamics channel        -> the state information gain

The channels are not free parameters: each is substituted at its closed-form
optimum, the corresponding conditional of q, so a sweep is plain sum-product and
the channels are refreshed from the beliefs it produces. Iterating
{sweep → refresh → damp} to a fixed point is the planner; uniform channels
recover standard Bethe BP.

Reading order in this file:

    single_objective_terms / evaluate_single_objective
        the scalar objective F, term by term (diagnostics only; planning never
        needs it, and the drivers leave ``track_objective=False``)
    _converge_obs_channels
        the obs channel r(y|x) fixed point, which is independent of the chain
        and so converges once up front
    _planning_core
        the main loop: forward_pass, backward_pass, then refresh r(u|x) and
        r(x'|x,u) from the dynamics region beliefs and damp
    sp_efe_planning
        public entry point; ``coupled_planner_kwargs`` maps a planner name onto
        the three static flags that select SP-EFE or one of its ablations

Two subtleties worth knowing. The likelihood tables are the "+1" digammas
ψ(α+1) − ψ(α_0+1) rather than the mean-field ψ(α) − ψ(α_0), because the energies
must be evaluated under the same Bayes-recoupled measure the Δ terms use; the
mean-field tables carry a per-column mass deficit ≈ (C−1)/α_0 that swamps the
tilts at low counts. And the obs tilt enters at first power because the energy
already contains the recoupled ambiguity entropy, so adding the coupled
correction on top would count −I_A twice. (The entry point passes
``obs_tilt_coef=2`` for the prior-faithful coupled objective; see there.)

Point-mass limit (counts → ∞): Δ → 0 pointwise, ψ(α+1) − ψ(α_0+1) → log θ*, and
the kernels reduce to g_obs = A·r(y|x), g_dyn = B·r(u|x)/r(x'|x,u) — the
fixed-parameter AIF-MP scheme.

Learning is outside planning and outside this file: q(θ) is frozen during a
plan, the centered epistemic scores integrate to zero against q(θ), so the
parameter update is the plain conjugate count increment (see ``agent.py``).

All computation is in log-space with LOG_ZERO guards for structural zeros.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from .sweeps import (
    LOG_ZERO,
    backward_pass,
    damp_log_channel,
    forward_pass,
    safe_log,
)
from .tables import dirichlet_log_tables
from .info_gain import _uniform_over_valid, dirichlet_posterior_kl


def _xlogy(q, log_v):
    """q·log_v with the 0·log 0 = 0 convention.

    Guards every objective contraction: beliefs are exact zeros wherever a
    kernel/prior entry is LOG_ZERO (exp underflows to 0.0 in float32), so
    masking on q > 0 suppresses the spurious q·(−1e10) products without
    changing any live term.
    """
    return jnp.where(q > 0.0, q * log_v, 0.0)


# =============================================================================
# Objective evaluation (Step 1) — the new capability of this variant
# =============================================================================


def single_objective_terms(base_dyn, log_lik_B_k, delta_B, log_fwd, log_bwd,
                           log_local, log_action_prior, log_q0, log_goal,
                           obs_energy_x, obs_kl_x, obs_tilt_coef=1.0,
                           per_step_goal=False):
    """Named scalar terms of the single objective F⁗ (D.O-T⁗) at the current sweep.

    Bethe convention: non-singleton blocks (obs, dyn) carry −H_block − E[log
    kernel table]; singleton priors (q0, goal, action prior; p̂_y uniform and
    dropped) enter as energies; variable counting numbers run over the
    non-singleton blocks. The obs-block negative entropy and the ambiguity
    correction +H[y_t|x_t] combine exactly into −H[q(x_t)] (chain rule), which
    the counting terms offset to a net coefficient of 0 at the chain ends and
    +1 at interior states — implemented as `x_entropy` below.

    All beliefs are rebuilt from the passed messages/kernels, so the function
    is a pure evaluator: it never touches the iteration state.

    Args:
        base_dyn: (T, x_old, x_new, u) current dyn kernel log ḡ_dyn (D.31b⁗),
            LOG_ZERO-masked
        log_lik_B_k: (x_old, x_new, u) likelihood table in kernel layout,
            LOG_ZERO-masked ("+1" for D2a; arithmetic mean for Option 2)
        delta_B: (T, x_old, x_new, u) dyn contrast tables Δ_B ≥ 0 (0 at
            structural zeros)
        log_fwd, log_bwd: (T+1, S) current forward/backward messages
        log_local: (T+1, S) obs→x messages
        log_action_prior: (U,) log p(u)
        log_q0: (S,) log initial state belief
        log_goal: (S,) log state preference
        obs_energy_x: (S,) per-state obs energy Σ_{f,y} r(y|x)·log table
        obs_kl_x: (S,) per-state obs contrast Σ_{f,y} r(y|x)·Δ_A(f,y,x)
        obs_tilt_coef: explicit obs IG coefficient in ΔF (1 for D2a
            AMENDMENT 4; 2 for Option 2 / D.9a⁗ full corrections)
        per_step_goal: if True, preference energy is Σ_{t=1..T} −E[log goal];
            else only the terminal −E_{q(x_T)}[log goal]

    Returns:
        dict of named scalar contributions; their sum is F⁗ (minimized).
    """
    horizon = base_dyn.shape[0]

    # State beliefs q(x_t) ∝ fwd·bwd·local, one per chain node
    log_qx_un = log_fwd + log_bwd + log_local
    log_qx = log_qx_un - logsumexp(log_qx_un, axis=1, keepdims=True)
    qx = jnp.exp(log_qx)

    # Dyn block beliefs q_trip,t(x_old, x_new, u) ∝ ḡ_dyn·fwd·bwd·p(u)
    fwd_local = log_fwd[:-1] + log_local[:-1]
    bwd_local = log_bwd[1:] + log_local[1:]
    log_qt_un = (base_dyn
                 + fwd_local[:, :, None, None]
                 + bwd_local[:, None, :, None]
                 + log_action_prior[None, None, None, :])
    log_qt = log_qt_un - logsumexp(log_qt_un, axis=(1, 2, 3), keepdims=True)
    qt = jnp.exp(log_qt)

    # --- dyn blocks: −H[q_trip] − E_b[log p] and the three corrections ------
    dyn_neg_entropy = _xlogy(qt, log_qt).sum()
    dyn_energy = -_xlogy(qt, jnp.broadcast_to(log_lik_B_k[None],
                                              qt.shape)).sum()
    # −H[q(x'|x,u)]: the denominator-channel correction, from q_trip itself
    log_cond_xnew = log_qt - logsumexp(log_qt, axis=2, keepdims=True)
    dyn_state_entropy = _xlogy(qt, log_cond_xnew).sum()
    # +H[q(u|x)]: the always-on planning correction
    log_qpair = logsumexp(log_qt, axis=2)  # (T, x_old, u), normalized
    log_cond_u = log_qpair - logsumexp(log_qpair, axis=2, keepdims=True)
    dyn_plan_entropy = -_xlogy(jnp.exp(log_qpair), log_cond_u).sum()
    # −E[Δ_B]: dyn parameter information gain (bonus, factor 1)
    dyn_info = -(qt * delta_B).sum()
    # −E[log p(u)]: action-prior energy
    q_u = qt.sum(axis=(1, 2))  # (T, U)
    dyn_action_energy = -_xlogy(
        q_u, jnp.broadcast_to(log_action_prior[None], q_u.shape)).sum()

    # --- obs blocks: −E[log table] − coef·E[Δ_A]; entropies → x_entropy ------
    obs_energy = -(qx * obs_energy_x[None, :]).sum()
    obs_info = -obs_tilt_coef * (qx * obs_kl_x[None, :]).sum()

    # Net state-entropy terms: +H[q(x_t)] at interior chain states only
    x_entropy = -_xlogy(qx[1:horizon], log_qx[1:horizon]).sum()

    # --- singleton energies -------------------------------------------------
    if per_step_goal:
        # Preference on every future state x_1..x_T (matches pymdp per-step C)
        goal_energy = -_xlogy(qx[1:], log_goal[None, :]).sum()
    else:
        goal_energy = -_xlogy(qx[horizon], log_goal).sum()
    init_energy = -_xlogy(qx[0], log_q0).sum()

    return {
        "dyn_neg_entropy": dyn_neg_entropy,
        "dyn_energy": dyn_energy,
        "dyn_state_entropy": dyn_state_entropy,
        "dyn_plan_entropy": dyn_plan_entropy,
        "dyn_info": dyn_info,
        "dyn_action_energy": dyn_action_energy,
        "obs_energy": obs_energy,
        "obs_info": obs_info,
        "x_entropy": x_entropy,
        "goal_energy": goal_energy,
        "init_energy": init_energy,
    }


def evaluate_single_objective(base_dyn, log_lik_B_k, delta_B, log_fwd, log_bwd,
                              log_local, log_action_prior, log_q0, log_goal,
                              obs_energy_x, obs_kl_x, obs_tilt_coef=1.0,
                              per_step_goal=False):
    """Scalar F⁗ = Σ of `single_objective_terms` (jit-safe)."""
    terms = single_objective_terms(
        base_dyn, log_lik_B_k, delta_B, log_fwd, log_bwd, log_local,
        log_action_prior, log_q0, log_goal, obs_energy_x, obs_kl_x,
        obs_tilt_coef=obs_tilt_coef, per_step_goal=per_step_goal)
    total = jnp.asarray(0.0)
    for v in terms.values():
        total = total + v
    return total


# =============================================================================
# Planning (Step 2)
# =============================================================================


def _converge_obs_channels(alpha_A, log_lik_A, n_iterations, damping,
                                  obs_tilt_coef=1, obs_salience=None):
    """Converge the θ-free obs conditional r(y|x) and emit obs->x.

    Single carry (plan "one refresh, two roles"): the block conditional
    r(y|x) is both the self-consistent ambiguity factor in the kernel and,
    exponentiated, the fractional counts n̄_A of the cavity Dir(α + n̄) that
    Δ_A contrasts against. The refresh conditions on x over the y axis, so
    the x-message cancels: obs channels are message-independent, identical at
    all timesteps, and converged once outside the planning loop (same
    argument as conjugate_aif._converge_obs_channels). p̂_y is uniform.

    Per sweep:
    - Δ_A(y,x) = KL(Dir(α+e_y) ‖ Dir(α+n̄)) from n̄_A = exp(log_r_yx)
    - log_r_yx <- norm_y[ log table + log_r_yx + coef·Δ_A ]
      coef=1 with "+1" tables (D2a); coef=2 with arithmetic tables (Option 2)

    Args:
        alpha_A:   (F, C, S) Dirichlet counts (for the Δ contrast)
        log_lik_A: (F, C, S) likelihood table (joint "+1" or arithmetic mean)
        n_iterations: convergence iterations
        damping: geometric channel damping
        obs_tilt_coef: integer power on Δ_A in the obs kernel (static)
        obs_salience: optional (F, C, S) additive log-space tilt (e.g.
            c·Δ_θ from `discrete_theta_salience`) applied alongside coef·Δ_A

    Returns:
        log_obs_to_x: (S,) normalized message from all obs factors to x
        log_r_yx: (F, C, S) converged block conditional r(y|x)
        delta_A: (F, C, S) converged contrast table Δ_A (0 at invalid entries)
    """
    valid_A = log_lik_A > LOG_ZERO / 2
    nbar0 = _uniform_over_valid(valid_A, axes=1)
    log_r_yx = jnp.where(valid_A, safe_log(nbar0), LOG_ZERO)
    coef = jnp.asarray(obs_tilt_coef, dtype=log_lik_A.dtype)
    salience = (jnp.zeros_like(log_lik_A) if obs_salience is None
                else obs_salience)

    def kernel(log_r_yx):
        nbar = jnp.where(valid_A, jnp.exp(log_r_yx), 0.0)
        delta = jnp.where(
            valid_A,
            dirichlet_posterior_kl(alpha_A, nbar, axis=1, cond_axes=(0, 2)),
            0.0)
        log_k = jnp.where(
            valid_A, log_lik_A + log_r_yx + coef * delta + salience,
            LOG_ZERO)
        return log_k, delta

    def body(i, log_r_yx):
        log_k, _ = kernel(log_r_yx)
        raw = log_k - logsumexp(log_k, axis=1, keepdims=True)
        return damp_log_channel(log_r_yx, raw, damping, cond_axis=1)

    log_r_yx = lax.fori_loop(0, n_iterations, body, log_r_yx)

    # μ_obs→x = Σ_y ḡ_obs(y,x) with uniform p̂_y, product over obs factors
    log_k, delta_A = kernel(log_r_yx)
    log_obs_to_x = logsumexp(log_k, axis=1).sum(axis=0)  # (S,)
    return log_obs_to_x - logsumexp(log_obs_to_x), log_r_yx, delta_A


def _planning_core(q_current_state, alpha_A, log_lik_A, alpha_B, log_lik_B,
                   goal, horizon, n_iterations, damping, action_prior,
                   track_objective, obs_tilt_coef=1, per_step_goal=False,
                   obs_salience=None, dyn_channel=True, dyn_tilt_coef=1):
    """Shared planning loop; raw alpha tensors are needed for the Δ contrasts.

    ``dyn_channel=False`` ablates the conditional dyn channel r(x'|x,u) (the
    −Σ H[q(x_t|x_{t-1},u_t)] state-info-gain correction): the channel stays
    at its uniform initialisation, so the kernel is B̃·exp(Δ_B)·r(u|x).

    ``dyn_tilt_coef=0`` drops the Dirichlet Δ_B tilt (parameter IG), leaving
    the posterior-mean table B̃ in the dynamics numerator. Combined with
    ``dyn_channel=True`` this is the MP analogue of analytic-efe-noamb-noig's
    E[log B̂] term without −IG_B.
    """
    n_states = q_current_state.shape[0]
    n_actions = log_lik_B.shape[2]

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)

    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # (x_old, x_new, u) kernel convention, as in the other planners
    log_lik_B_k = log_lik_B.transpose(1, 0, 2)
    valid_B_k = log_lik_B_k > LOG_ZERO / 2  # static structural mask

    # Time-independent obs channel -> single obs->x message for all t
    log_obs_to_x, log_r_yx, delta_A = _converge_obs_channels(
        alpha_A, log_lik_A, n_iterations, damping, obs_tilt_coef,
        obs_salience)
    _log_local_obs = jnp.broadcast_to(
        log_obs_to_x[None, :], (horizon + 1, n_states))
    # Per-step goal modes (paper note P1):
    # - True ("channel"): fold preference into the *channel refresh* locals
    #   for t=1..T, obs-only fwd/bwd locals, terminal singleton kept.
    #   Empirical hybrid — avoids the trap-path pathology of unary insertion
    #   but has no rollout-semantics guarantee (P1.5).
    # - "additive": rollout (expected-utility) semantics per P1. NO
    #   preference in any graph potential (obs-only locals everywhere, flat
    #   terminal singleton); preference enters only the r(u|x) refresh via
    #   the additive Bellman sweep (P1.5a-b) computed below. Propagating
    #   preference with geometric/BP messages instead computes exponential
    #   utility and buys "fantasy" transitions at stochastic branch points
    #   (P1.4/P1.6) — the sink-trap-from-START failure.
    additive_goal = per_step_goal == "additive"
    if per_step_goal and not additive_goal:
        step_mask = (jnp.arange(horizon + 1) >= 1).astype(_log_local_obs.dtype)
        _log_local_chan = (_log_local_obs
                           + step_mask[:, None] * log_goal[None, :])
    else:
        _log_local_chan = _log_local_obs
    log_goal_bwd = jnp.zeros_like(log_goal) if additive_goal else log_goal
    # Beliefs / objective use the same local that enters the dyn blocks
    _log_local_bel = (_log_local_chan
                      if (per_step_goal and not additive_goal)
                      else _log_local_obs)

    if additive_goal:
        # Additive value sweep (P1.5a-b): risk-neutral cost-to-go under the
        # posterior-mean table B̂, soft-min over actions (the KL(r‖p(u))
        # regularizer). Message-independent -> computed once per call.
        # Zero-preference states get a large finite cost instead of +inf so
        # the expectation stays well-conditioned in float32.
        b_mean_k = (alpha_B / alpha_B.sum(axis=0, keepdims=True)
                    ).transpose(1, 0, 2)                      # (x, x', u)
        step_cost = jnp.where(log_goal > LOG_ZERO / 2, -log_goal, 1e3)
        w_val = jnp.zeros(n_states)
        q_vals = []
        for _ in range(horizon):  # t = T .. 1
            j_val = step_cost + w_val
            q_t = jnp.einsum("xyu,y->xu", b_mean_k, j_val)
            w_val = -logsumexp(log_action_prior[None, :] - q_t, axis=1)
            q_vals.append(q_t)
        neg_q_value = -jnp.stack(q_vals[::-1])                # (T, S, U)
    else:
        neg_q_value = jnp.zeros((horizon, n_states, n_actions))

    # Per-state obs contractions for the objective (time-independent):
    # Σ_{f,y} r(y|x)·log table and Σ_{f,y} r(y|x)·Δ_A
    r_yx_p = jnp.where(log_r_yx > LOG_ZERO / 2, jnp.exp(log_r_yx), 0.0)
    obs_energy_x = _xlogy(r_yx_p, log_lik_A).sum(axis=(0, 1))  # (S,)
    obs_kl_x = (r_yx_p * delta_A).sum(axis=(0, 1))             # (S,)
    obs_coef = jnp.asarray(obs_tilt_coef, dtype=log_lik_A.dtype)

    # Per-t contrast Δ_B from fractional counts: kernel layout
    # (x_old, x_new, u) -> dirichlet layout (x_new, x_old, u), axis=0,
    # cond x_old; the u-columns of a state x aggregate (cross terms)
    def delta_B_fn(nbar_B):
        d = jax.vmap(
            lambda nb: dirichlet_posterior_kl(
                alpha_B, nb, axis=0, cond_axes=(1,))
        )(nbar_B.transpose(0, 2, 1, 3))  # (T, x_new, x_old, u)
        return jnp.where(valid_B_k[None], d.transpose(0, 2, 1, 3), 0.0)

    q_u_init = jnp.zeros((horizon, n_actions))
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))
    # Conditional dyn channel r(x'|x,u): uniform over x'
    log_r_xxu_init = jnp.full((horizon, n_states, n_states, n_actions),
                              -jnp.log(n_states))
    # θ-free planning channel r(u|x): always-on +H[q(u|x)] term
    log_r_ux_init = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))
    # Fractional counts q_t(x',u|x), uniform over the valid (x',u) per x
    nbar_B_init = jnp.broadcast_to(
        _uniform_over_valid(valid_B_k, axes=(1, 2))[None],
        (horizon, n_states, n_states, n_actions))
    f_trace_init = jnp.zeros(n_iterations)

    def body_fn(i, carry):
        _, log_r_xxu, log_r_ux, nbar_B, log_fwd_prev, log_bwd_prev, f_trace = carry

        # ḡ_dyn = B̃ · exp(Δ_B) · r(u|x) / r(x'|x,u) (Proposition 13⁗). The
        # validity mask is CHANNEL-based, exactly as in
        # active_inference_planning: a denominator channel entry that
        # collapses toward zero (e.g. x' ≠ goal at the last step under a
        # one-hot goal) is clipped out of the graph one-way instead of
        # amplifying the kernel via −log r → +∞. Structural zeros are
        # already LOG_ZERO inside log_lik_B_k.
        delta_B = (delta_B_fn(nbar_B) if dyn_tilt_coef
                   else jnp.zeros((horizon, n_states, n_states, n_actions),
                                  log_lik_B_k.dtype))
        dyn_valid = log_r_xxu > LOG_ZERO / 2
        neg_r_xxu = jnp.where(dyn_valid, -log_r_xxu, 0.0)
        base_dyn = (log_lik_B_k[None] + delta_B
                    + log_r_ux[:, :, None, :] + neg_r_xxu)
        base_dyn = jnp.where(dyn_valid, base_dyn, LOG_ZERO)

        log_reduced_per_t = base_dyn.transpose(0, 2, 1, 3)  # (T, x_new, x_old, u)

        # Obs-only locals in fwd/bwd: preference enters via channels (+ terminal)
        log_fwd_msgs = forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, _log_local_obs, horizon,
            log_prev_fwd=log_fwd_prev, msg_damping=damping
        )
        log_bwd_msgs, q_u = backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal_bwd, log_action_prior,
            _log_local_obs, horizon,
            log_prev_bwd=log_bwd_prev, msg_damping=damping
        )

        # Channel refresh (D.5.2⁗): every target is a conditional of the
        # current block belief, so ONE contraction with the kernel table
        # (no arithmetic-table variant — block-marginal preservation). All
        # targets condition on x_old, so μ_{x_{t-1}→dyn} cancels exactly —
        # it is OMITTED rather than cancelled numerically: a LOG_ZERO
        # forward entry would shift the whole slice to ~LOG_ZERO magnitude
        # where float32 spacing destroys the O(1) within-slice structure.
        # Per-step preference (if any) lives in _log_local_chan here.
        bwd_local = log_bwd_msgs[1:] + _log_local_chan[1:]
        log_msgs = (bwd_local[:, None, :, None]
                    + log_action_prior[None, None, None, :])
        m = base_dyn + log_msgs

        # r(x'|x,u) <- norm_{x'}[m]  (preference-free m in additive mode:
        # the P1.2 hard constraint — preference must not tilt the state
        # conditionals or the learning counts)
        raw_r_xxu = m - logsumexp(m, axis=2, keepdims=True)
        # r(u|x) <- norm_u[ lse_{x'} m (- Q_t in additive mode, P1.5b) ]
        log_pair = logsumexp(m, axis=2) + neg_q_value  # (T, x_old, u)
        raw_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)
        # n̄_B <- q_t(x',u|x), damped linearly in count space; structural
        # zeros masked so the contrast never sees mass on them
        b_joint = jnp.exp(m - logsumexp(m, axis=(2, 3), keepdims=True))
        b_joint = jnp.where(valid_B_k[None], b_joint, 0.0)
        new_nbar = (1.0 - damping) * nbar_B + damping * b_joint

        new_r_xxu = (damp_log_channel(log_r_xxu, raw_r_xxu, damping, cond_axis=2)
                     if dyn_channel else log_r_xxu)
        new_r_ux = damp_log_channel(
            log_r_ux, raw_r_ux, damping, cond_axis=2)

        if track_objective:
            f = evaluate_single_objective(
                base_dyn, log_lik_B_k, delta_B, log_fwd_msgs, log_bwd_msgs,
                _log_local_bel, log_action_prior, log_q0, log_goal,
                obs_energy_x, obs_kl_x, obs_tilt_coef=obs_coef,
                per_step_goal=per_step_goal)
            f_trace = f_trace.at[i].set(f)

        return (q_u, new_r_xxu, new_r_ux, new_nbar,
                log_fwd_msgs, log_bwd_msgs, f_trace)

    result = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_r_xxu_init, log_r_ux_init, nbar_B_init,
         log_fwd_prev_init, log_bwd_prev_init, f_trace_init)
    )
    q_u, log_r_xxu, log_r_ux, nbar_B, _, _, f_trace = result

    if additive_goal:
        # Root policy (P1.4 item 4): q(u_1) ∝ Σ_x q0(x) r_1(u|x). The
        # backward-pass q_u is preference-free here, so it cannot rank
        # actions by value.
        log_root = logsumexp(log_q0[:, None] + log_r_ux[0], axis=0)
        action_dist = jnp.exp(log_root - logsumexp(log_root))
    else:
        action_dist = q_u[0]
        action_dist = action_dist / (action_dist.sum() + 1e-10)
    if track_objective:
        return action_dist, log_r_xxu, log_r_ux, nbar_B, f_trace
    return action_dist, log_r_xxu, log_r_ux, nbar_B


# =============================================================================
# Public entry point: SP-EFE and its ablations
# =============================================================================

COUPLED_PLANNERS = ("coupled-nodyn", "coupled-mp", "coupled-nodelta")
COUPLED_IGC_PLANNERS = ("coupled-nodyn-igc", "coupled-mp-igc",
                        "coupled-nodelta-igc")


def coupled_planner_kwargs(planner):
    """Channel / Dirichlet-tilt flags for a ``coupled-*`` planner name.

    ``coupled-*-igc`` keeps an optional ``obs_salience`` (IG_c / Δ_θ)
    and drops Dirichlet novelty: ``obs_tilt_coef=0`` (no IG_A) and
    ``dyn_tilt_coef=0`` (no IG_B). ``coupled-mp-igc`` and
    ``coupled-nodelta-igc`` are therefore identical.
    """
    igc = planner.endswith("-igc")
    base = planner[:-4] if igc else planner
    if base not in COUPLED_PLANNERS:
        raise ValueError(planner)
    return dict(
        dyn_channel=(base != "coupled-nodyn"),
        dyn_tilt_coef=0 if (igc or base == "coupled-nodelta") else 1,
        obs_tilt_coef=0 if igc else 2,
    )


@partial(jax.jit, static_argnums=(4, 5), static_argnames=(
    "track_objective", "per_step_goal", "dyn_channel", "dyn_tilt_coef",
    "obs_tilt_coef"))
def sp_efe_planning(
    q_current_state,
    alpha_A,
    alpha_B,
    goal,
    horizon,
    n_iterations,
    damping=1.0,
    action_prior=None,
    track_objective=False,
    per_step_goal=False,
    obs_salience=None,
    dyn_channel=True,     # bool (static): False ablates r(x'|x,u)
    dyn_tilt_coef=1,      # int (static): 0 drops Δ_B
    obs_tilt_coef=2,      # int (static): 0 drops Δ_A
) -> tuple:
    """Plan via the prior-faithful coupled objective (D.2b / Option 2).

    Arithmetic-mean likelihood tables + obs tilt 2·Δ_A + dyn tilt Δ_B.
    Same return signature as `conjugate_aif_single_planning`.

    ``per_step_goal=True`` folds the preference into dyn-channel refresh for
    every future state x_1..x_T (while keeping the terminal singleton). This
    breaks wall-self-loop indifference at long horizons without the trap-path
    pathology of injecting preference into forward/backward message locals.

    ``per_step_goal="additive"`` is the principled rollout-semantics variant
    (paper note P1): preference enters only the r(u|x) refresh through an
    additive (risk-neutral) Bellman value sweep under the posterior-mean B̂,
    while all graph potentials, beliefs, and learning counts stay
    preference-free.

    ``obs_salience``: optional (F, C, S) additive log-space obs tilt, e.g.
    c·Δ_θ from ``discrete_theta_salience`` (paper note P2) — count-free
    information gain about a discrete θ block shared by the obs factors.

    ``dyn_channel=False`` drops the conditional dyn channel r(x'|x,u), i.e.
    the −Σ H[q(x_t|x_{t-1},u_t)] state-info-gain correction (the notebook's
    "Exploration" prior), keeping Δ_B, the obs channels and r(u|x). Under a
    smeared/learned B̂ the channel makes the kernel support-uniform and wipes
    out the exploitation signal; with identity A it also rewards stochastic
    transitions ("noisy TV").

    ``dyn_tilt_coef=0`` drops the Dirichlet Δ_B tilt (parameter IG). With
    the dyn channel still on this is the MP analogue of analytic-efe-noamb-noig
    (E[log B̂] without −IG_B); both off recovers a KL-control-like kernel
    B̃·r(u|x).

    ``obs_tilt_coef=0`` drops the Dirichlet Δ_A tilt (parameter IG_A).
    Combined with ``dyn_tilt_coef=0`` this is the IG_c-only ablation
    (``obs_salience`` / Δ_θ remains): T-maze ``coupled-*-igc``.
    """
    _, log_lik_A = dirichlet_log_tables(alpha_A, axis=1)
    _, log_lik_B = dirichlet_log_tables(alpha_B, axis=0)
    return _planning_core(
        q_current_state, alpha_A, log_lik_A, alpha_B, log_lik_B, goal,
        horizon, n_iterations, damping, action_prior, track_objective,
        obs_tilt_coef=obs_tilt_coef, per_step_goal=per_step_goal,
        obs_salience=obs_salience, dyn_channel=dyn_channel,
        dyn_tilt_coef=dyn_tilt_coef)
