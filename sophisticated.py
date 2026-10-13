"""Sophisticated-inference EFE: the one-step terms of eq. (7), recursed over outcomes.

The enumerated ``EFE`` in ``dispatch.py`` scores open-loop action sequences
with the counts frozen over the rollout. Here the same one-step terms g(b, u)
are recursed over hypothetical observations (Friston et al., 2021):

    G(b, u) = g(b, u) + Σ_y q(y | b, u) Σ_u' σ(−γ G(b_y, ·))_u' G(b_y, u')

with b = (q(x), α_A, α_B) and b_y the belief after observing y: the state
posterior as in ``agent.filter_step`` and, with ``learn=True``, the counts as in
``agent.update_counts_*`` ("sophisticated learning"). Learning inside the tree
gives novelty diminishing returns: a cell already observed on a branch is
worth less the second time, which frozen-count EFE cannot express.

Outcomes with q(y) < ``obs_threshold`` are merged into one "other" branch whose
state belief is the exact posterior given y ∉ kept and whose counts get the
expected (fractional) increment over the merged outcomes. With nothing kept
this is the open-loop prediction plus fractional counts, so a flat column
(many equally likely outcomes) still loses novelty on revisits instead of the
tree being cut off there. Subtrees are memoized on (depth, belief, tables),
so a +1 on a clamped column does not split the memo. At H = 1 this equals
``EFE``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma

from dispatch import EPS, dyn_terms, obs_terms

DEFAULT_OBS_THRESHOLD = 1.0 / 16   # SPM's outcome-pruning threshold
MEMO_DECIMALS = 10                 # tables equal to this precision share a subtree


class _Counts:
    """Dirichlet counts and the per-column one-step EFE tables they imply."""

    __slots__ = ("alpha_A", "alpha_B", "A_hat", "state_term", "B_hat",
                 "dyn_term", "key")

    def __init__(self, alpha_A, alpha_B, A_hat, state_term, B_hat, dyn_term):
        self.alpha_A, self.alpha_B = alpha_A, alpha_B
        self.A_hat, self.state_term = A_hat, state_term
        self.B_hat, self.dyn_term = B_hat, dyn_term
        self.key = hash(tuple(np.round(t, MEMO_DECIMALS).tobytes()
                              for t in (A_hat, state_term, B_hat, dyn_term)))

    @classmethod
    def build(cls, alpha_A, alpha_B):
        A_hat, state_term = obs_terms(alpha_A, digamma=digamma)
        B_hat, dyn_term = dyn_terms(alpha_B, digamma=digamma)
        return cls(alpha_A, alpha_B, A_hat, state_term, B_hat, dyn_term)

    def learn(self, obs_counts, q_prev, action):
        """Counts after one step; only the touched columns are recomputed.

        ``obs_counts`` (C, S) is the increment to α_A: e_y ⊗ q(x_t | y) for an
        observed outcome, its expectation over the outcomes of a merged
        branch. Its column sums are q(x_t), which drives the α_B increment.
        """
        q_post = obs_counts.sum(axis=0)
        post_cols = np.flatnonzero(q_post > 0)
        prev_cols = np.flatnonzero(q_prev > 0)

        alpha_A = self.alpha_A.copy()
        alpha_A[0][:, post_cols] += obs_counts[:, post_cols]
        A_hat, state_term = self.A_hat.copy(), self.state_term.copy()
        A_hat[:, :, post_cols], state_term[post_cols] = obs_terms(
            alpha_A[:, :, post_cols], digamma=digamma)

        alpha_B = self.alpha_B.copy()
        alpha_B[np.ix_(post_cols, prev_cols, [action])] += np.outer(
            q_post[post_cols], q_prev[prev_cols])[:, :, None]
        B_hat, dyn_term = self.B_hat.copy(), self.dyn_term.copy()
        B_hat[:, prev_cols, action], dyn_term[prev_cols, action] = dyn_terms(
            alpha_B[:, prev_cols, action], digamma=digamma)
        return _Counts(alpha_A, alpha_B, A_hat, state_term, B_hat, dyn_term)


def _softmax(x):
    w = np.exp(x - x.max())
    return w / w.sum()


def si_efe_action_posterior(alpha_A, alpha_B, q0, goal, horizon, *, learn=True,
                            per_step_goal=False, gamma=1.0,
                            obs_threshold=DEFAULT_OBS_THRESHOLD, stats=None):
    """σ(−γ G(b_0, ·)) for the current belief, by depth-``horizon`` recursion.

    Args mirror ``analytic_efe_action_posterior``; the preference is
    terminal-only unless ``per_step_goal``. ``learn`` updates the counts
    along each branch. ``stats``, if a dict, receives the number of
    expanded nodes and memo hits.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    alpha_A = np.asarray(alpha_A, dtype=np.float64)
    alpha_B = np.asarray(alpha_B, dtype=np.float64)
    if alpha_A.shape[0] != 1:
        raise ValueError("one observation factor expected, as in agent.filter_step")
    goal = np.asarray(goal, dtype=np.float64)
    log_goal = np.log(np.maximum(goal / goal.sum(), EPS))
    n_actions = alpha_B.shape[2]
    memo, counter = {}, {"nodes": 0, "memo_hits": 0}

    def next_step_value(q, counts, depth):
        """E_{q(u'|b)}[G(b, u')] with q(u'|b) = σ(−γ G(b, ·))."""
        key = (depth, np.round(q, 9).tobytes(), counts.key)
        if key in memo:
            counter["memo_hits"] += 1
        else:
            G = action_values(q, counts, depth)
            memo[key] = float(_softmax(-gamma * G) @ G)
        return memo[key]

    def action_values(q, counts, depth):
        counter["nodes"] += 1
        G = np.empty(n_actions)
        for u in range(n_actions):
            q_pred = counts.B_hat[:, :, u] @ q
            # One step of eq. (7): E[log B̂] − IG_B on x_{t-1}; ambiguity − IG_A on x_t
            g = q @ counts.dyn_term[:, u] + q_pred @ counts.state_term
            if per_step_goal or depth == 1:
                g -= q_pred @ log_goal
            if depth > 1:
                joint = counts.A_hat[0] * q_pred[None, :]          # q(y, x_t)
                p_obs = joint.sum(axis=1)
                kept = np.flatnonzero(p_obs >= obs_threshold)
                for obs in kept:
                    obs_counts = np.zeros_like(joint)
                    obs_counts[obs] = joint[obs] / p_obs[obs]
                    child = counts.learn(obs_counts, q, u) if learn else counts
                    g += p_obs[obs] * next_step_value(obs_counts[obs], child, depth - 1)
                p_rest = p_obs.sum() - p_obs[kept].sum()
                if p_rest > EPS:   # every pruned outcome as one branch
                    rest_counts = joint.copy()
                    rest_counts[kept] = 0.0
                    rest_counts /= p_rest
                    child = counts.learn(rest_counts, q, u) if learn else counts
                    g += p_rest * next_step_value(rest_counts.sum(axis=0), child, depth - 1)
            G[u] = g
        return G

    G_root = action_values(np.asarray(q0, dtype=np.float64),
                           _Counts.build(alpha_A, alpha_B), horizon)
    if stats is not None:
        stats.update(counter)
    return _softmax(-gamma * G_root)
