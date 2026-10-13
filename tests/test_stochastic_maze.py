"""Tests for the fog-walls maze and its SP-EFE scenarios."""

import numpy as np


# ---------------------------------------------------------------------------
# Tensor generation tests
# ---------------------------------------------------------------------------

class TestStochasticMazeTensors:
    """Validate the fog-walls tensors."""

    def setup_method(self):
        from environments.stochastic_maze import (
            build_A, build_A_matrix, build_B, build_D, build_goal,
            maze_tensors,
        )
        self.A = build_A_matrix()
        self.A_planner = build_A()
        self.B = build_B()
        self.D = build_D()
        self.goal = build_goal()
        self.A_m, self.B_m, self.meta = maze_tensors("fog-walls")

    def test_layout_meta(self):
        from environments.stochastic_maze import FOG_CELLS, FOG_WALLS
        assert self.meta["name"] == "fog-walls"
        assert self.meta["sink_states"] == ()
        assert self.meta["wall_states"] == FOG_WALLS
        assert self.meta["fog_cells"] == FOG_CELLS
        assert np.array_equal(self.A_m, self.A_planner)
        assert np.array_equal(self.B_m, self.B)

    def test_unknown_maze_rejected(self):
        from environments.stochastic_maze import maze_tensors
        try:
            maze_tensors("fog")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "fog-walls" in str(exc)

    def test_transitions_normal_cells(self):
        from environments.stochastic_maze import xy_to_state
        B = self.B
        assert np.allclose(B.sum(axis=0), 1.0)
        s = xy_to_state(0, 0)
        assert B[xy_to_state(0, 1), s, 0] == 1.0   # NORTH
        assert B[xy_to_state(1, 0), s, 1] == 1.0   # EAST
        assert B[s, s, 2] == 1.0                   # SOUTH wall -> stay
        assert B[s, s, 3] == 1.0                   # WEST wall -> stay

    def test_walls_bump_and_are_unreachable(self):
        """A move into a wall stays put, and nothing else enters a wall."""
        from environments.stochastic_maze import FOG_WALLS, xy_to_state
        B = self.B
        for (x, y) in FOG_WALLS:
            w = xy_to_state(x, y)
            assert np.all(B[w, w, :] == 1.0)
            others = np.setdiff1d(np.arange(B.shape[1]), [w])
            assert np.all(B[w][others] == 0.0)
        E, N = 1, 0
        # EAST from the south fog landing (1,1) into wall (2,1): stay
        s = xy_to_state(1, 1)
        assert B[s, s, E] == 1.0
        # NORTH from (1,1) is the drift cell (1,2)
        assert B[xy_to_state(1, 2), s, N] == 1.0
        # the open middle row still leads to the goal
        s = xy_to_state(2, 2)
        assert B[xy_to_state(3, 2), s, E] == 1.0
        assert B[xy_to_state(4, 2), xy_to_state(3, 2), E] == 1.0

    def test_drift_at_fog_cell(self):
        """(1, 2) drifts north/south with p=0.6; the landing cells are open."""
        from environments.stochastic_maze import xy_to_state
        s = xy_to_state(1, 2)
        up, down, east = xy_to_state(1, 3), xy_to_state(1, 1), xy_to_state(2, 2)
        # intended EAST 0.4, drift N/S 0.3 each
        assert np.isclose(self.B[east, s, 1], 0.4, atol=1e-6)
        assert np.isclose(self.B[up, s, 1], 0.3, atol=1e-6)
        assert np.isclose(self.B[down, s, 1], 0.3, atol=1e-6)

    def test_observation_matrix(self):
        from environments.stochastic_maze import (
            FOG_CELLS, FOG_NOISY_OBSERVATIONS, N_STATES, xy_to_state,
        )
        A = self.A
        assert np.allclose(A.sum(axis=0), 1.0)
        fog = [xy_to_state(*c) for c in FOG_CELLS]
        noisy = {xy_to_state(*c) for c in FOG_NOISY_OBSERVATIONS}
        for s in range(N_STATES):
            if s not in fog and s not in noisy:
                assert A[s, s] == 1.0
        for s in fog:
            col = A[:, s]
            for o in fog:
                assert np.isclose(col[o], 0.2)
            assert np.allclose(col[np.setdiff1d(np.arange(len(col)), fog)], 0.0)
        # (2, 0): 0.2 exact, 0.8 over 5 neighbours
        s = xy_to_state(2, 0)
        assert np.isclose(A[s, s], 0.2, atol=1e-6)
        assert np.isclose(A[xy_to_state(1, 0), s], 0.8 / 5, atol=1e-6)
        assert self.A_planner.shape[0] == 1
        assert np.array_equal(self.A_planner[0], A)

    def test_goal_and_start(self):
        from environments.stochastic_maze import (
            GOAL_STATE, START_STATE, build_C, state_to_xy,
        )
        assert state_to_xy(GOAL_STATE) == (4, 2)
        assert state_to_xy(START_STATE) == (0, 2)
        C = build_C(preference=4.0)
        assert C[GOAL_STATE] == 4.0 and np.count_nonzero(C) == 1
        assert np.isclose(self.goal.sum(), 1.0)
        assert self.goal.argmax() == GOAL_STATE

    def test_start_prior(self):
        from environments.stochastic_maze import START_STATE
        assert self.D.sum() == 1.0
        assert self.D[START_STATE] == 1.0

    def test_safe_top_row_path(self):
        """The 8-step path along the top row is deterministic and its
        observations are exact."""
        from environments.stochastic_maze import (
            GOAL_XY, START_XY, state_to_xy, xy_to_state,
        )
        N, E, S = 0, 1, 2
        x, y = START_XY
        for a in (N, N, E, E, E, E, S, S):
            s = xy_to_state(x, y)
            col = self.B[:, s, a]
            assert col.max() == 1.0            # deterministic
            assert self.A[s, s] == 1.0         # exact observation
            x, y = state_to_xy(int(col.argmax()))
        assert (x, y) == GOAL_XY


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestStochasticMazeEnv:

    def test_step_semantics(self):
        from environments.stochastic_maze import (
            START_STATE, StochasticMazeEnv, xy_to_state,
        )
        env = StochasticMazeEnv(rng=np.random.default_rng(0))
        obs = env.reset()
        assert env.state == START_STATE and obs == START_STATE  # clean cell
        obs, reward = env.step(0)           # NORTH -> (0, 3), deterministic
        assert env.state == xy_to_state(0, 3) and reward == 0.0
        assert obs == env.state             # clean cell: obs = state
        obs, reward = env.step(2)           # SOUTH -> back to start
        assert env.state == START_STATE

    def test_wall_bump(self):
        from environments.stochastic_maze import StochasticMazeEnv, xy_to_state
        env = StochasticMazeEnv(rng=np.random.default_rng(0))
        start = xy_to_state(1, 1)           # fog cell south of the drift
        env.reset(start_state=start)
        env.step(1)                          # EAST into wall (2, 1)
        assert env.state == start

    def test_fog_observation_support(self):
        from environments.stochastic_maze import (
            FOG_CELLS, StochasticMazeEnv, xy_to_state,
        )
        env = StochasticMazeEnv(rng=np.random.default_rng(0))
        env.reset(start_state=xy_to_state(2, 2))
        fog = {xy_to_state(*c) for c in FOG_CELLS}
        obs = [env.observe() for _ in range(2000)]
        assert set(obs) <= fog
        hits = np.mean([o == env.state for o in obs])
        assert abs(hits - 0.2) < 0.05

    def test_goal_reward_via_safe_path(self):
        from environments.stochastic_maze import GOAL_STATE, StochasticMazeEnv
        env = StochasticMazeEnv(rng=np.random.default_rng(0))
        env.reset()
        N, E, S = 0, 1, 2
        for a in (N, N, E, E, E, E, S):
            _, reward = env.step(a)
            assert reward == 0.0
        _, reward = env.step(S)
        assert env.state == GOAL_STATE and reward == 1.0


# ---------------------------------------------------------------------------
# SP-EFE integration tests
# ---------------------------------------------------------------------------

class TestStochasticMazeCoupledIntegration:

    def test_known_scenario_reaches_goal(self):
        from run_fog_walls import run_scenario
        result = run_scenario("known", n_episodes=2, seed=0)   # T = H = 11
        assert [r["outcome"] for r in result["results"]] == ["GOAL", "GOAL"]
        for r in result["results"]:
            assert r["steps"] <= 20   # 8 = safe path

    def test_known_model_first_actions(self):
        """Known A and B at H=8: SP-EFE steps north onto the unambiguous
        top row; the preference planners step east into the fog."""
        import jax.numpy as jnp
        from agent import CLAMP
        from environments.stochastic_maze import build_A, build_B, build_D, build_goal
        from spmp.sp_efe import paper_scheme, sp_efe_planning
        aA = jnp.asarray(CLAMP * build_A())
        aB = jnp.asarray(CLAMP * build_B())
        goal, q0 = jnp.asarray(build_goal(4.0)), jnp.asarray(build_D())

        def plan_with(planner):
            return sp_efe_planning(
                q0, aA, aB, goal, horizon=8, n_iterations=100,
                damping=0.5, scheme=paper_scheme(planner)).action_dist

        pref, plan, efe = (plan_with(p) for p in
                           ("SP-Pref", "SP-Pref-Plan", "SP-EFE"))
        assert int(np.argmax(efe)) == 0     # NORTH
        assert int(np.argmax(pref)) == 1    # EAST
        assert int(np.argmax(plan)) == 1    # EAST

    def test_analytic_rejected_at_maze_horizon(self):
        from run_fog_walls import run_scenario
        try:
            run_scenario("known", n_episodes=1, planner="EFE",
                         quiet=True, collect_frames=False)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "enumerates" in str(exc)

    def test_learning_errors_decrease(self):
        from run_fog_walls import run_scenario
        res_a = run_scenario("learn-A", n_episodes=5, max_steps=10, seed=0)
        errs_a = [r["err"] for r in res_a["results"]]
        assert errs_a[-1] < errs_a[0]
        res_b = run_scenario("learn-B", n_episodes=5, max_steps=10, seed=0)
        errs_b = [r["err"] for r in res_b["results"]]
        assert errs_b[-1] < errs_b[0]
