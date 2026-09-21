"""Preference-augmented sum-product (CLI: pref-sp / pref-sp-plan).

Baseline for `conjugate_aif_coupled`: same generative factors (obs A, dyn B,
state preference ˆp(x), action prior) and the same arithmetic-mean Dirichlet
tables log m = log(α/α₀), but **no entropy corrections** — no channel
reparameterizations r(·|·), no epistemic tilts Δ_A / Δ_B.

This is unmodified sum-product / plain preference-augmented VFE on the
temporal chain (= KL control in the paper's F_marginal sense). CLI
``pref-sp`` is this baseline; ``pref-sp-plan`` sets
``planning_correction=True``. Uniform channels + zero tilts recover the
same kernels; here we skip the channel machinery entirely.

``planning_correction=True`` adds back exactly one correction, the planning
term Σₜ H[q(uₜ|xₜ₋₁)]: the policy channel r(u|x) enters the dyn kernel
numerator (κ = B · r(u|x), refresh r(u|x) ← q(x,u)/q(x)) and, in the
``"additive"`` mode, the value sweep uses the hard-min instead of the soft-min
(the soft-min ``−log Σ_u p(u) e^{−Q}`` = ``min_r E_r[Q] − H[r] + log U`` carries
the same action-entropy bonus the channel removes from the graph). No obs
channels, no dyn channel, no Δ tilts — this is the cross-entropy-planning
(VBP-channel) counterpart of the KL-control baseline.

Factor graph (θ collapsed into mean tables):

    p(x₀)─x₀─[dyn: m_B]─x₁─[dyn]─···─x_T─ˆp(x)
              |             |
             u₀            u₁
              ↑             ↑
           [obs: m_A]    [obs: m_A]
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from .sweeps import LOG_ZERO, safe_log
from .tables import dirichlet_log_tables
from .sweeps import (
    backward_pass, damp_log_channel, forward_pass)


def _obs_to_x_from_mean(log_lik_A, obs_salience=None):
    """μ_obs→x from arithmetic-mean obs tables with uniform ˆp_y.

    log_lik_A: (F, C, S). For each factor, Σ_y m_A(y|x) exp(Δ), product over
    factors. ``obs_salience`` is an optional (F, C, S) additive log tilt
    (e.g. c·Δ_θ from ``discrete_theta_salience``) — the discrete-hypothesis
    information gain that mean-field would otherwise drop.
    """
    valid = log_lik_A > LOG_ZERO / 2
    salience = (jnp.zeros_like(log_lik_A) if obs_salience is None
                else obs_salience)
    log_k = jnp.where(valid, log_lik_A + salience, LOG_ZERO)
    # Σ_y m_A(y|x) e^{Δ} per factor → (F, S); product over F
    log_obs_to_x = logsumexp(log_k, axis=1).sum(axis=0)
    return log_obs_to_x - logsumexp(log_obs_to_x)


@partial(jax.jit, static_argnums=(4, 5),
         static_argnames=("per_step_goal", "planning_correction"))
def sp_pref_planning(
    q_current_state,      # (n_states,)
    alpha_A,              # (F, C, S) Dirichlet counts
    alpha_B,              # (S', S, U) Dirichlet counts
    goal,                 # (n_states,) state preference
    horizon,              # int (static)
    n_iterations,         # int (static)
    damping=1.0,
    action_prior=None,    # (n_actions,); uniform if None
    per_step_goal=False,  # False | True | "additive"
    planning_correction=False,  # bool (static): add Σ H[q(u|x)] via r(u|x)
    obs_salience=None,    # (F, C, S) additive log tilt (e.g. c·Δ_θ, note P2)
) -> tuple:
    """Plan via preference-augmented sum-product (pref-sp / pref-sp-plan).

    Arithmetic-mean likelihood tables, unmodified kernels, iterated
    forward/backward on the (x,u) chain. Same return shape convention as
    the conjugate planners for drop-in use in grid runners: ``(action_dist,)``
    plus placeholders so ``ad, *_ = ...`` works.

    Args:
        q_current_state: (S,) current state belief
        alpha_A: (F, C, S) obs Dirichlet counts
        alpha_B: (S', S, U) transition Dirichlet counts
        goal: (S,) state preference
        horizon: planning horizon T
        n_iterations: VMP / message-damping sweeps
        damping: inertial message damping (1 = full update)
        action_prior: (U,) action prior / mask
        per_step_goal: False = terminal-only preference; True = unary
            preference factors on x_1..x_T (exponential-utility / Variant A);
            ``"additive"`` = soft value iteration under mean B (rollout /
            expected-utility semantics, note P1) — no preference in the
            message graph; root policy from the soft Bellman Q.
        planning_correction: add the planning correction Σₜ H[q(uₜ|xₜ₋₁)]:
            policy channel r(u|x) in the dyn kernel numerator, refreshed as
            r(u|x) ← norm_u[lse_x' m(x',u|x) (− Q_t)], geometrically damped;
            in ``"additive"`` mode the value sweep is the hard-min and the
            root policy is Σ_x q0(x) r_1(u|x).
        obs_salience: optional (F, C, S) additive log-space obs tilt, e.g.
            c·Δ_θ from ``discrete_theta_salience`` (paper note P2). This is
            the count-free I(y; θ_c | x) drive; it does not add obs/dyn
            channels or Dirichlet Δ_A / Δ_B.

    Returns:
        action_dist: (U,) first-action marginal
        log_reduced: (T, S', S, U) dyn kernels used (for diagnostics)
        log_fwd: (T+1, S) final forward messages
        log_bwd: (T+1, S) final backward messages
    """
    n_states = q_current_state.shape[0]
    _, log_lik_A = dirichlet_log_tables(alpha_A, axis=1)
    _, log_lik_B = dirichlet_log_tables(alpha_B, axis=0)
    n_actions = log_lik_B.shape[2]

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    log_action_prior = safe_log(action_prior)
    log_q0 = safe_log(q_current_state)
    log_goal = safe_log(goal)

    # Dyn kernel = arithmetic mean only; reduced layout (T, x_new, x_old, u)
    log_reduced = jnp.broadcast_to(
        log_lik_B[None], (horizon, n_states, n_states, n_actions))

    # Obs → x from mean A (constant across time), plus optional Δ_θ
    log_obs_to_x = _obs_to_x_from_mean(log_lik_A, obs_salience)
    log_local_obs = jnp.broadcast_to(
        log_obs_to_x[None, :], (horizon + 1, n_states))

    additive_goal = per_step_goal == "additive"
    if additive_goal:
        # Value iteration under B̂ (P1.5): risk-neutral per-step cost. No
        # preference in the VMP graph — beliefs stay predictive. Soft-min
        # (KL-control: entropy-regularised, −log Σ_u p(u) e^{−Q}) by default;
        # hard-min with the planning correction (the soft-min's action-
        # entropy bonus is exactly the Σ H[q(u|x)] term the channel removes).
        b_mean = alpha_B / alpha_B.sum(axis=0, keepdims=True)  # (x', x, u)
        step_cost = jnp.where(log_goal > LOG_ZERO / 2, -log_goal, 1e3)
        w_val = jnp.zeros(n_states)
        q_vals = []
        for _ in range(horizon):  # t = T .. 1
            j_val = step_cost + w_val
            q_t = jnp.einsum("yxu,y->xu", b_mean, j_val)
            if planning_correction:
                w_val = q_t.min(axis=1)
            else:
                w_val = -logsumexp(log_action_prior[None, :] - q_t, axis=1)
            q_vals.append(q_t)
        neg_q_value = -jnp.stack(q_vals[::-1])                # (T, S, U)
        if not planning_correction:
            # Root action = soft Bellman policy contracted with q0
            log_root = logsumexp(log_q0[:, None] + log_action_prior[None, :]
                                 + neg_q_value[0], axis=0)
            action_dist = jnp.exp(log_root - logsumexp(log_root))
            # Preference-free VMP messages for diagnostic returns
            log_fwd = forward_pass(
                log_reduced, log_q0, log_action_prior, log_local_obs, horizon)
            log_bwd, _ = backward_pass(
                log_reduced, log_fwd, jnp.zeros_like(log_goal),
                log_action_prior, log_local_obs, horizon)
            return action_dist, log_reduced, log_fwd, log_bwd
        log_local = log_local_obs
        log_goal_bwd = jnp.zeros_like(log_goal)
    else:
        neg_q_value = jnp.zeros((horizon, n_states, n_actions))
        if per_step_goal:
            # Preference energy Σ_{t=1..T} −E[log ˆp]: inject into locals at
            # t≥1; flat terminal avoids double-counting x_T. (Variant A /
            # exponential utility — trap-seeking on the sink at START.)
            step_mask = (jnp.arange(horizon + 1) >= 1).astype(
                log_local_obs.dtype)
            log_local = log_local_obs + step_mask[:, None] * log_goal[None, :]
            log_goal_bwd = jnp.zeros_like(log_goal)
        else:
            log_local = log_local_obs
            log_goal_bwd = log_goal

    if not planning_correction:
        q_u_init = jnp.zeros((horizon, n_actions))
        log_fwd_prev = jnp.zeros((horizon + 1, n_states))
        log_bwd_prev = jnp.zeros((horizon + 1, n_states))

        def body_fn(_, carry):
            _, log_fwd_prev, log_bwd_prev = carry
            log_fwd = forward_pass(
                log_reduced, log_q0, log_action_prior, log_local, horizon,
                log_prev_fwd=log_fwd_prev, msg_damping=damping,
            )
            log_bwd, q_u = backward_pass(
                log_reduced, log_fwd, log_goal_bwd, log_action_prior,
                log_local, horizon,
                log_prev_bwd=log_bwd_prev, msg_damping=damping,
            )
            return q_u, log_fwd, log_bwd

        q_u, log_fwd, log_bwd = lax.fori_loop(
            0, n_iterations, body_fn, (q_u_init, log_fwd_prev, log_bwd_prev))

        action_dist = q_u[0]
        action_dist = action_dist / (action_dist.sum() + 1e-10)
        return action_dist, log_reduced, log_fwd, log_bwd

    # --- planning correction: policy channel r(u|x) in the dyn numerator ---
    # Kernel layout (T, x_old, x_new, u) so r(u|x) broadcasts over x_new.
    log_lik_B_k = log_lik_B.transpose(1, 0, 2)                # (x, x', u)
    valid_B_k = log_lik_B_k > LOG_ZERO / 2
    log_r_ux_init = jnp.full((horizon, n_states, n_actions),
                             -jnp.log(n_actions))
    q_u_init = jnp.zeros((horizon, n_actions))
    log_fwd_prev_init = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev_init = jnp.zeros((horizon + 1, n_states))

    def body_fn(_, carry):
        _, log_r_ux, log_fwd_prev, log_bwd_prev = carry
        # ḡ_dyn = B̂ · r(u|x); the +H[q(u|x)] correction is the numerator
        base_dyn = jnp.where(valid_B_k[None],
                             log_lik_B_k[None] + log_r_ux[:, :, None, :],
                             LOG_ZERO)                        # (T, x, x', u)
        log_reduced_t = base_dyn.transpose(0, 2, 1, 3)         # (T, x', x, u)

        log_fwd = forward_pass(
            log_reduced_t, log_q0, log_action_prior, log_local, horizon,
            log_prev_fwd=log_fwd_prev, msg_damping=damping)
        log_bwd, q_u = backward_pass(
            log_reduced_t, log_fwd, log_goal_bwd, log_action_prior,
            log_local, horizon, log_prev_bwd=log_bwd_prev,
            msg_damping=damping)

        # r(u|x) <- norm_u[ lse_x' m(x',u|x) (− Q_t) ]; m conditions on
        # x_old, so the forward message cancels and is omitted (D.5.2).
        bwd_local = log_bwd[1:] + log_local[1:]               # (T, x')
        m = (base_dyn + bwd_local[:, None, :, None]
             + log_action_prior[None, None, None, :])
        log_pair = logsumexp(m, axis=2) + neg_q_value          # (T, x, u)
        raw_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)
        new_r_ux = damp_log_channel(log_r_ux, raw_r_ux, damping, cond_axis=2)
        return q_u, new_r_ux, log_fwd, log_bwd

    q_u, log_r_ux, log_fwd, log_bwd = lax.fori_loop(
        0, n_iterations, body_fn,
        (q_u_init, log_r_ux_init, log_fwd_prev_init, log_bwd_prev_init))

    if additive_goal:
        # Root policy q(u_1) ∝ Σ_x q0(x) r_1(u|x): the backward-pass q_u is
        # preference-free here and cannot rank actions by value.
        log_root = logsumexp(log_q0[:, None] + log_r_ux[0], axis=0)
        action_dist = jnp.exp(log_root - logsumexp(log_root))
    else:
        action_dist = q_u[0]
        action_dist = action_dist / (action_dist.sum() + 1e-10)
    log_reduced = jnp.where(
        valid_B_k[None], log_lik_B_k[None] + log_r_ux[:, :, None, :],
        LOG_ZERO).transpose(0, 2, 1, 3)
    return action_dist, log_reduced, log_fwd, log_bwd
