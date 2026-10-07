"""Planner dispatch shared by the two experiments, plus the enumerated EFE.

The analytic planners enumerate policies on whatever ``(alpha_A, alpha_B,
goal)`` they are given.

Planner ids are the paper's labels (``SP-EFE``, ``SP-EFE-dyn``,
``SP-Pref``, ...). The same strings are the ``--planner`` values and the
``.npz`` keys. ``PRETTY`` is that label typeset for matplotlib; superscripts
such as SP-EFE^{-dyn} are display only.
"""

from __future__ import annotations

import itertools

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import digamma as jax_digamma

from spmp.sp_efe import (
    SP_EFE_IGC_PLANNERS, SP_EFE_PLANNERS, SP_PREF_PLANNERS, paper_scheme,
    sp_efe_planning,
)

EPS = 1e-12
DEFAULT_ITERS = 100
DEFAULT_DAMPING = 0.5
IDENTIFIED_MASS = 0.8
PLANNERS = SP_EFE_PLANNERS + SP_PREF_PLANNERS
ANALYTIC_PLANNERS = ("EFE", "EFE-noamb-noig")
ALL_PLANNERS = PLANNERS + ANALYTIC_PLANNERS
PLANNER_CHOICES = ALL_PLANNERS + SP_EFE_IGC_PLANNERS

# Matplotlib mathtext for the same ids. Plain ids stay the npz / CLI keys.
PRETTY = {
    "SP-EFE": "SP-EFE",
    "SP-EFE-dyn": r"SP-EFE$^{-\mathrm{dyn}}$",
    "SP-EFE-DeltaB": r"SP-EFE$^{-\Delta_B}$",
    "SP-EFE-DeltaA-DeltaB": r"SP-EFE$^{-\Delta_A-\Delta_B}$",
    "SP-EFE-DeltaA-DeltaB-dyn": r"SP-EFE$^{-\Delta_A-\Delta_B-\mathrm{dyn}}$",
    "SP-Pref": "SP-Pref",
    "SP-Pref-Plan": "SP-Pref-Plan",
    "EFE": "EFE",
    "EFE-noamb-noig": "EFE-noamb-noig",
    "Random": "Random",
}
STYLE = {
    "SP-EFE-dyn": dict(color="#1f77b4", ls="-", marker="o"),
    "SP-EFE": dict(color="#ff7f0e", ls="-", marker="*"),
    "SP-EFE-DeltaB": dict(color="#9467bd", ls="-", marker="P"),
    "SP-EFE-DeltaA-DeltaB-dyn": dict(color="#1f77b4", ls=":", marker="o"),
    "SP-EFE-DeltaA-DeltaB": dict(color="#ff7f0e", ls=":", marker="*"),
    "SP-Pref": dict(color="#2ca02c", ls="-.", marker="s"),
    "SP-Pref-Plan": dict(color="#006400", ls="-.", marker="D"),
    "EFE": dict(color="#d62728", ls="--", marker="^"),
    "EFE-noamb-noig": dict(color="#8c564b", ls="--", marker="v"),
    "Random": dict(color="0.5", ls=":", marker=""),
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
    is True, keep ambiguity and −IG_A / −IG_B (``EFE``). When
    False, those three terms are zeroed, leaving E[log B̂] = −H[q(x'|x,u)]
    plus terminal CE (``EFE-noamb-noig``). Preference is
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
            include_ig=(planner == "EFE"))
    return sp_efe_planning(
        q, alpha_A, alpha_B, goal, horizon=h, n_iterations=n_iterations,
        damping=damping, per_step_goal=per_step_goal,
        scheme=paper_scheme(planner)).action_dist
