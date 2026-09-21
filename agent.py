"""The protocol shared by every planner: filtering, learning, action selection.

This is Algorithm 1 of the paper, and it is deliberately the *only* copy. The
experiments compare objectives, so everything outside the objective has to be
held fixed: each planner sees the same belief from the same exact filter under
the same posterior-mean matrices, updates the same Dirichlet counts the same
way, and picks its action from q(u_1) by the same rule. Keeping one
implementation is what makes that claim checkable rather than a promise.

The per-step cycle, for step s of a T-step episode:

    1. filter_step      q(x_s) from the observation and the last action
    2. update_counts_*  fold the transition / observation into the Dirichlet
    3. horizon_at       H_s = min(H, T - s + 1), the shrinking plan depth
    4. <planner>        q(u_1) over the remaining horizon
    5. choose_action    argmax of q(u_1)

Counts persist across episodes; pose and belief reset. Planning never touches
the counts -- q(θ) is frozen for the duration of a plan -- so learning is the
plain conjugate increment below and nothing in ``spmp/`` needs to know about
it.

What is *not* here is the episode bookkeeping: what counts as an outcome, which
cells are absorbing, and which metrics get recorded are environment-specific,
and the two experiments legitimately differ. See ``experiments/``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import nn

# Floor on Dirichlet column sums and probabilities before a division or log.
EPS = 1e-12
# Concentration used to pin a known block to its true matrix: large enough that
# the posterior mean is the truth and the information gain is numerically zero.
CLAMP = 1.0e6
# Per-entry mass of a flat Dirichlet prior on the unknown block.
FLAT = 0.1


# ---------------------------------------------------------------------------
# Posterior-mean matrices: the only view of θ that filtering and learning use
# ---------------------------------------------------------------------------
def posterior_mean_A(alpha_A):
    """m_A = E[θ_A], normalizing each observation column over outcomes."""
    a = np.asarray(alpha_A)
    return a / np.maximum(a.sum(axis=1, keepdims=True), EPS)


def posterior_mean_B(alpha_B):
    """m_B = E[θ_B], normalizing each (state, action) column over successors."""
    a = np.asarray(alpha_B)
    return a / np.maximum(a.sum(axis=0, keepdims=True), EPS)


def matrix_error(alpha, true, axis, mask=None):
    """Mean |m − true|, over ``mask`` if given, else over every entry.

    Accumulated in float64 regardless of the Dirichlet's own dtype: this is a
    reported metric, not part of the inference path, so there is no reason to
    carry float32 rounding into it.
    """
    a = np.asarray(alpha, dtype=np.float64)
    mean = a / np.maximum(a.sum(axis=axis, keepdims=True), EPS)
    diff = np.abs(mean - np.asarray(true, dtype=np.float64))
    if mask is None:
        return float(diff.mean())
    mask = np.asarray(mask)
    return float(diff[mask].mean()) if mask.any() else 0.0


# ---------------------------------------------------------------------------
# Filtering and conjugate learning
# ---------------------------------------------------------------------------
def filter_step(q, obs, alpha_A, alpha_B, last_action):
    """One exact-filtering step under the posterior-mean matrices.

    ``obs`` is the observed index; ``last_action = -1`` means no prediction
    (the observation at episode start). Returns ``(q_pred, q_post)``: the
    one-step prediction m_B q and the posterior after the likelihood.
    """
    A_mean = jnp.asarray(posterior_mean_A(alpha_A))
    if last_action >= 0:
        B_mean = jnp.asarray(posterior_mean_B(alpha_B))
        pred = B_mean[:, :, last_action] @ q
        pred = pred / (pred.sum() + EPS)
    else:
        pred = q
    log_lik = jnp.log(A_mean[0, obs, :] + EPS)
    return pred, nn.softmax(log_lik + jnp.log(pred + EPS))


def update_counts_A(alpha_A, obs, q_post):
    """Conjugate increment on θ_A: add q(x) to the observed outcome's row."""
    return alpha_A.at[0, obs].add(q_post)


def update_counts_B(alpha_B, q_prev, q_post, action):
    """Conjugate increment on θ_B: add the pair marginal q(x')q(x) at ``action``."""
    return alpha_B.at[:, :, action].add(jnp.outer(q_post, q_prev))


# ---------------------------------------------------------------------------
# Plan depth and action selection
# ---------------------------------------------------------------------------
def horizon_at(step, max_steps, horizon, shrinking=True):
    """Shrinking plan depth H_s = min(H, T − s + 1); ``step`` is 0-based."""
    if not shrinking:
        return int(horizon)
    return max(1, min(int(horizon), int(max_steps) - int(step)))


def choose_action(action_dist, n_actions=None):
    """Argmax of q(u_1), ties broken by lowest action index."""
    ad = np.asarray(action_dist, dtype=np.float64)
    if not np.all(np.isfinite(ad)) or (
            n_actions is not None and ad.shape != (int(n_actions),)):
        raise RuntimeError(f"bad action_dist {ad}")
    peak = float(np.max(ad))
    return int(np.flatnonzero(ad >= peak - 1e-10)[0])


def flat_alpha(shape, prior_count=FLAT, dtype=jnp.float32):
    """Flat Dirichlet prior on an unknown block."""
    return jnp.full(shape, float(prior_count), dtype)


def clamped_alpha(true_matrix, clamp=CLAMP):
    """Pin a known block to its true matrix with ``clamp`` counts per entry."""
    return jnp.asarray(clamp * np.asarray(true_matrix))
