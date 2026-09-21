"""Planner dispatch shared by the two experiments, plus the enumerated EFE.

Extracted verbatim from ``run_deep_sea_coupled.py`` in the parent repo
(UAI-MP-AIF-JAX), which is where both experiment drivers used to import
``_plan`` from. Nothing here is DeepSea-specific: the analytic planners
enumerate policies on whatever ``(alpha_A, alpha_B, goal)`` they are given.

Paper names (see ``PAPER_NAME`` in ``src/make_experiment_figures.py``):

==========================  ====================================
code name                   paper label
==========================  ====================================
``coupled-mp``              SP-EFE
``coupled-nodyn``           SP-EFE^-dyn
``coupled-nodelta``         SP-EFE^-Delta_B
``coupled-mp-igc``          SP-EFE^-Delta_A-Delta_B
``coupled-nodyn-igc``       SP-EFE^-Delta_A-Delta_B-dyn
``pref-sp-plan``            SP-Pref-Plan
``pref-sp``                 SP-Pref
``analytic-efe``            EFE (enumerated)
==========================  ====================================
"""

from __future__ import annotations

import itertools

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import digamma as jax_digamma

from spmp.sp_efe import (
    COUPLED_IGC_PLANNERS, COUPLED_PLANNERS, sp_efe_planning,
    coupled_planner_kwargs,
)
from spmp.sp_pref import sp_pref_planning

EPS = 1e-12
DEFAULT_ITERS = 100
DEFAULT_DAMPING = 0.5
IDENTIFIED_MASS = 0.8
PLANNERS = ("coupled-nodyn", "coupled-mp", "coupled-nodelta",
            "pref-sp", "pref-sp-plan")
ANALYTIC_PLANNERS = ("analytic-efe", "analytic-efe-noamb-noig")
ALL_PLANNERS = PLANNERS + ANALYTIC_PLANNERS
PLANNER_CHOICES = ALL_PLANNERS + COUPLED_IGC_PLANNERS

# Plot labels / styles, previously imported from ``run_grid3x3_learn_B``.
PRETTY = {
    "coupled-nodyn": "coupled-nodyn",
    "coupled-mp": "coupled-mp",
    "coupled-nodelta": "coupled-nodelta",
    "coupled-nodyn-igc": "coupled-nodyn-igc",
    "coupled-mp-igc": "coupled-mp-igc",
    "coupled-nodelta-igc": "coupled-nodelta-igc",
    "pref-sp": "pref-sp",
    "pref-sp-plan": "pref-sp-plan",
    "analytic-efe": "analytic-efe",
    "analytic-efe-noamb-noig": "analytic-efe-noamb-noig",
}
STYLE = {
    "coupled-nodyn": dict(color="#1f77b4", ls="-", marker="o"),
    "coupled-mp": dict(color="#ff7f0e", ls="-", marker="*"),
    "coupled-nodelta": dict(color="#9467bd", ls="-", marker="P"),
    "coupled-nodyn-igc": dict(color="#1f77b4", ls=":", marker="o"),
    "coupled-mp-igc": dict(color="#ff7f0e", ls=":", marker="*"),
    "coupled-nodelta-igc": dict(color="#9467bd", ls=":", marker="P"),
    "pref-sp": dict(color="#2ca02c", ls="-.", marker="s"),
    "pref-sp-plan": dict(color="#006400", ls="-.", marker="D"),
    "analytic-efe": dict(color="#d62728", ls="--", marker="^"),
    "analytic-efe-noamb-noig": dict(color="#8c564b", ls="--", marker="v"),
}


def _mean_A(alpha_A):
    a = np.asarray(alpha_A)
    return a / np.maximum(a.sum(axis=1, keepdims=True), EPS)


def _mean_B(alpha_B):
    a = np.asarray(alpha_B)
    return a / np.maximum(a.sum(axis=0, keepdims=True), EPS)


def _dg(x):
    """Digamma on numpy arrays via JAX (avoids a hard scipy import)."""
    return np.asarray(jax_digamma(jnp.asarray(x)))


def analytic_efe_action_posterior(alpha_A, alpha_B, q0, goal, horizon,
                                  include_ig=True):
    """Enumerated state-space EFE; first-action marginal of q(π)∝exp(−G).

    θ-free dynamics numerator (``B̂`` point estimate). When ``include_ig``
    is True, keep ambiguity and −IG_A / −IG_B (``analytic-efe``). When
    False, those three terms are zeroed, leaving E[log B̂] = −H[q(x'|x,u)]
    plus terminal CE (``analytic-efe-noamb-noig``). Preference is
    **terminal-only**: −E[log p̂(x_H)] on the remaining horizon, matching
    ``per_step_goal=False`` in the MP planners. Counts are frozen over the
    rollout (standard EFE).
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    alpha_A = np.asarray(alpha_A, dtype=np.float64)
    alpha_B = np.asarray(alpha_B, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    log_goal = np.log(np.maximum(goal / goal.sum(), EPS))

    safe_A = np.where(alpha_A > 0, alpha_A, 1.0)
    a0_A = alpha_A.sum(axis=1, keepdims=True)
    A_hat = np.where(alpha_A > 0, alpha_A / np.maximum(a0_A, EPS), 0.0)
    elog_A = np.where(alpha_A > 0, _dg(safe_A + 1) - _dg(a0_A + 1), 0.0)
    ambig = -(A_hat * elog_A).sum(axis=(0, 1))
    kl_A = np.where(alpha_A > 0,
                    np.log(np.maximum(a0_A, EPS)) - np.log(safe_A)
                    + _dg(safe_A + 1) - _dg(a0_A + 1), 0.0)
    ig_A = (A_hat * kl_A).sum(axis=(0, 1))
    state_term = ambig - ig_A if include_ig else np.zeros_like(ambig)

    safe_B = np.where(alpha_B > 0, alpha_B, 1.0)
    a0_B = alpha_B.sum(axis=0, keepdims=True)
    B_hat = np.where(alpha_B > 0, alpha_B / np.maximum(a0_B, EPS), 0.0)
    log_B = np.where(B_hat > 0, np.log(np.where(B_hat > 0, B_hat, 1.0)), 0.0)
    kl_B = np.where(alpha_B > 0,
                    np.log(np.maximum(a0_B, EPS)) - np.log(safe_B)
                    + _dg(safe_B + 1) - _dg(a0_B + 1), 0.0)
    ig_B = (B_hat * kl_B).sum(axis=0) if include_ig else np.zeros(
        alpha_B.shape[1:])

    n_actions = alpha_B.shape[2]
    policies = list(itertools.product(range(n_actions), repeat=horizon))
    G = np.zeros(len(policies))
    for i, pi in enumerate(policies):
        q = q0.copy()
        g = 0.0
        for u in pi:
            joint = B_hat[:, :, u] * q[None, :]
            q_next = joint.sum(axis=1)
            g += (joint * log_B[:, :, u]).sum()
            g += (q_next * state_term).sum()
            g -= (q * ig_B[:, u]).sum()
            q = q_next
        g -= (q * log_goal).sum()
        G[i] = g
    w = np.exp(-(G - G.min()))
    w /= w.sum()
    post = np.zeros(n_actions)
    for i, pi in enumerate(policies):
        post[pi[0]] += w[i]
    return post






def _plan(planner, q, alpha_A, alpha_B, goal, h, n_iterations, damping,
          per_step_goal):
    if planner in ANALYTIC_PLANNERS:
        return analytic_efe_action_posterior(
            alpha_A, alpha_B, q, goal, h,
            include_ig=(planner == "analytic-efe"))
    if planner in COUPLED_PLANNERS + COUPLED_IGC_PLANNERS:
        ad, *_ = sp_efe_planning(
            q, alpha_A, alpha_B, goal, h, n_iterations,
            damping=damping, per_step_goal=per_step_goal,
            **coupled_planner_kwargs(planner))
    else:
        ad, *_ = sp_pref_planning(
            q, alpha_A, alpha_B, goal, h, n_iterations,
            damping=damping, per_step_goal=per_step_goal,
            planning_correction=(planner == "pref-sp-plan"))
    return ad
