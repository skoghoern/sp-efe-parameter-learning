"""Sophisticated-inference EFE against the enumerated EFE it extends."""

import numpy as np

from dispatch import analytic_efe_action_posterior
from sophisticated import si_efe_action_posterior


def _dwell_or_chain(novelty_dwell, novelty_chain, n_obs=2):
    """From state 0: action 0 enters an absorbing novel cell (1); action 1 walks
    a chain of three novel cells (2 → 3 → 4, then stays). B known, A flat on 1..4."""
    n_states, n_actions = 5, 2
    B = np.zeros((n_states, n_states, n_actions))
    B[1, 0, 0] = B[2, 0, 1] = 1
    B[1, 1, :] = B[3, 2, :] = B[4, 3, :] = B[4, 4, :] = 1
    alpha_A = np.full((1, n_obs, n_states), novelty_chain)
    alpha_A[0, :, 1] = novelty_dwell
    alpha_A[0, :, 0] = 1.0
    alpha_A[0, 0, 0] = 1e6
    return alpha_A, 1e6 * B, np.eye(n_states)[0], np.ones(n_states)


def test_one_step_equals_enumerated_efe():
    rng = np.random.default_rng(0)
    for _ in range(3):
        alpha_A = rng.gamma(0.5, 1.0, (1, 3, 6)) + 0.05
        alpha_B = rng.gamma(0.5, 1.0, (6, 6, 4)) + 0.01
        q0, goal = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        np.testing.assert_allclose(   # float64 vs float32 digamma
            si_efe_action_posterior(alpha_A, alpha_B, q0, goal, 1),
            analytic_efe_action_posterior(alpha_A, alpha_B, q0, goal, 1),
            atol=1e-6)


def test_frozen_counts_match_enumerated_on_deterministic_dynamics():
    args = _dwell_or_chain(0.1, 0.15)
    np.testing.assert_allclose(
        si_efe_action_posterior(*args, 3, learn=False),
        analytic_efe_action_posterior(*args, 3), atol=1e-6)


def test_in_tree_learning_prefers_distinct_novel_cells():
    """Frozen counts pay the dwell cell's novelty on every step and stay;
    learning inside the tree discounts the revisits and takes the chain."""
    args = _dwell_or_chain(0.1, 0.15)
    assert np.argmax(analytic_efe_action_posterior(*args, 3)) == 0
    assert np.argmax(si_efe_action_posterior(*args, 3, learn=False)) == 0
    assert np.argmax(si_efe_action_posterior(*args, 3, learn=True)) == 1


def test_merged_outcomes_still_discount_revisits():
    """With 20 equally likely outcomes every q(y) < 1/16, so all of them go to
    the merged branch; its fractional counts still change the plan."""
    args = _dwell_or_chain(0.1, 0.15, n_obs=20)
    stats = {}
    frozen = si_efe_action_posterior(*args, 3, learn=False)
    learned = si_efe_action_posterior(*args, 3, learn=True, stats=stats)
    assert learned[1] > frozen[1] + 0.05
    assert stats["nodes"] < 50                       # no per-outcome branching


def test_pruned_outcomes_keep_the_lookahead():
    """With every outcome pruned the recursion continues open-loop, so the
    terminal preference two steps away is still seen."""
    alpha_A, alpha_B, q0, _ = _dwell_or_chain(0.5, 0.5)
    goal = np.full(5, 1e-3)
    goal[3] = 1.0                                   # reachable only via the chain
    stats = {}
    dist = si_efe_action_posterior(alpha_A, alpha_B, q0, goal, 2, learn=False,
                                   obs_threshold=2.0, stats=stats)
    assert np.argmax(dist) == 1
    assert stats["nodes"] >= 2


def test_runners_accept_si_planners():
    from run_fog_walls import run_scenario
    from run_unmapped_detour import run_episodes
    from environments.unmapped_detour import UnmappedDetourGrid
    run = run_scenario("learn-A", n_episodes=1, planner="SI-EFE",
                       maze="fog-walls", horizon=3, quiet=True,
                       collect_frames=False)
    assert len(run["results"]) == 1
    run = run_episodes("SI-EFE-frozen", grid=UnmappedDetourGrid(unmapped_mode="doors"),
                       scenario="learn-B", n_episodes=1, horizon=3)
    assert run["n_episodes"] == 1
