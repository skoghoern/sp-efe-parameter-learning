"""The batched enumerated EFE against a per-policy reference rollout."""

import itertools

import numpy as np

from dispatch import EPS, _dg, analytic_efe_action_posterior


def _reference_efe(alpha_A, alpha_B, q0, goal, horizon):
    """One policy at a time, term by term, as the EFE is written down."""
    a0_A = alpha_A.sum(axis=1, keepdims=True)
    A_hat = alpha_A / a0_A
    elog_A = _dg(alpha_A + 1) - _dg(a0_A + 1)
    ig_A = (A_hat * (np.log(a0_A) - np.log(alpha_A) + elog_A)).sum(axis=(0, 1))
    state_term = -(A_hat * elog_A).sum(axis=(0, 1)) - ig_A
    a0_B = alpha_B.sum(axis=0, keepdims=True)
    B_hat = alpha_B / a0_B
    ig_B = (B_hat * (np.log(a0_B) - np.log(alpha_B)
                     + _dg(alpha_B + 1) - _dg(a0_B + 1))).sum(axis=0)
    log_goal = np.log(np.maximum(goal / goal.sum(), EPS))

    n_actions = alpha_B.shape[2]
    policies = list(itertools.product(range(n_actions), repeat=horizon))
    G = np.zeros(len(policies))
    for i, pi in enumerate(policies):
        q = q0.copy()
        for u in pi:
            joint = B_hat[:, :, u] * q[None, :]
            G[i] += (joint * np.log(B_hat[:, :, u])).sum() - q @ ig_B[:, u]
            q = joint.sum(axis=1)
            G[i] += q @ state_term
        G[i] -= q @ log_goal
    w = np.exp(-(G - G.min()))
    w /= w.sum()
    return np.array([w[[p[0] == a for p in policies]].sum()
                     for a in range(n_actions)])


def test_batched_efe_matches_reference():
    rng = np.random.default_rng(0)
    n_states, n_actions, n_obs = 7, 4, 3
    for horizon in (1, 2, 4):
        alpha_A = rng.gamma(0.5, 1.0, (1, n_obs, n_states)) + 0.05
        alpha_B = rng.gamma(0.5, 1.0, (n_states, n_states, n_actions)) + 0.01
        q0 = rng.dirichlet(np.ones(n_states))
        goal = rng.dirichlet(np.ones(n_states))
        np.testing.assert_allclose(
            analytic_efe_action_posterior(alpha_A, alpha_B, q0, goal, horizon),
            _reference_efe(alpha_A, alpha_B, q0, goal, horizon),
            rtol=1e-6, atol=1e-9)   # _dg is float32 digamma via JAX
