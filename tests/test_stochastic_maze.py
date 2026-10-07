"""Tests for the stochastic maze environment and its SP-EFE scenarios.

The maze is ported from the RxInferExamples.jl notebook ``EFE Minimization
via Message Passing.ipynb`` (5x5 grid, drift cells, noisy observations).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Tensor generation tests
# ---------------------------------------------------------------------------

class TestStochasticMazeTensors:
    """Validate the maze tensors."""

    def setup_method(self):
        from environments.stochastic_maze import (
            build_A, build_A_matrix, build_B, build_D, build_goal,
        )
        self.A = build_A_matrix()
        self.A_planner = build_A()
        self.B = build_B()
        self.D = build_D()
        self.goal = build_goal()

    def test_transitions_normal_cells(self):
        from environments.stochastic_maze import xy_to_state
        B = self.B
        assert np.allclose(B.sum(axis=0), 1.0)
        s = xy_to_state(0, 0)
        assert B[xy_to_state(0, 1), s, 0] == 1.0   # NORTH
        assert B[xy_to_state(1, 0), s, 1] == 1.0   # EAST
        assert B[s, s, 2] == 1.0                   # SOUTH wall -> stay
        assert B[s, s, 3] == 1.0                   # WEST wall -> stay

    def test_sink_states_absorbing(self):
        from environments.stochastic_maze import SINK_STATES, xy_to_state
        for (x, y) in SINK_STATES:
            s = xy_to_state(x, y)
            assert np.all(self.B[s, s, :] == 1.0)

    def test_default_drift_conveyor_and_trap(self):
        """Default DRIFT_SPEC: (1,2),(2,2) drift East with p=0.5; (3,2)
        drifts up/down with p=0.7 into the two flanking sinks."""
        from environments.stochastic_maze import xy_to_state
        N, E, S, W = 0, 1, 2, 3
        # conveyor cell (1,2): intended 0.5, East drift 0.5
        s = xy_to_state(1, 2)
        assert np.isclose(self.B[xy_to_state(2, 2), s, E], 1.0, atol=1e-6)
        assert np.isclose(self.B[xy_to_state(1, 3), s, N], 0.5, atol=1e-6)
        assert np.isclose(self.B[xy_to_state(2, 2), s, N], 0.5, atol=1e-6)
        assert np.isclose(self.B[xy_to_state(0, 2), s, W], 0.5, atol=1e-6)
        assert np.isclose(self.B[xy_to_state(2, 2), s, W], 0.5, atol=1e-6)
        # trap cell (3,2): intended 0.3, 0.35 each into sinks (3,3), (3,1)
        s = xy_to_state(3, 2)
        up, down = xy_to_state(3, 3), xy_to_state(3, 1)
        assert np.isclose(self.B[xy_to_state(4, 2), s, E], 0.3, atol=1e-6)
        assert np.isclose(self.B[up, s, E], 0.35, atol=1e-6)
        assert np.isclose(self.B[down, s, E], 0.35, atol=1e-6)
        assert np.isclose(self.B[up, s, N], 0.65, atol=1e-6)
        assert np.isclose(self.B[down, s, S], 0.65, atol=1e-6)
        # P(sink | E from trap cell) = 0.7
        assert np.isclose(self.B[up, s, E] + self.B[down, s, E], 0.7)

    def test_notebook_drift_spec(self):
        """Notebook drift: 0.3 * move(intended) + 0.7 * (0.5 up + 0.5 down)."""
        from environments.stochastic_maze import (
            NOTEBOOK_DRIFT_SPEC, build_B, xy_to_state,
        )
        B = build_B(drift_spec=NOTEBOOK_DRIFT_SPEC)
        assert np.allclose(B.sum(axis=0), 1.0)
        s = xy_to_state(1, 2)
        up, down = xy_to_state(1, 3), xy_to_state(1, 1)
        assert np.isclose(B[up, s, 0], 0.65, atol=1e-6)      # NORTH
        assert np.isclose(B[down, s, 0], 0.35, atol=1e-6)
        assert np.isclose(B[up, s, 1], 0.35, atol=1e-6)      # EAST
        assert np.isclose(B[down, s, 1], 0.35, atol=1e-6)
        assert np.isclose(B[xy_to_state(2, 2), s, 1], 0.3, atol=1e-6)

    def test_observation_matrix(self):
        from environments.stochastic_maze import (
            NOISY_OBSERVATIONS, N_STATES, xy_to_state,
        )
        A = self.A
        assert np.allclose(A.sum(axis=0), 1.0)
        noisy = {xy_to_state(x, y) for (x, y) in NOISY_OBSERVATIONS}
        for s in range(N_STATES):
            if s not in noisy:
                assert A[s, s] == 1.0
        # interior noisy cell (2,2): 0.2 exact, 0.8 over 8 neighbours
        s = xy_to_state(2, 2)
        assert np.isclose(A[s, s], 0.2, atol=1e-6)
        assert np.isclose(A[xy_to_state(1, 1), s], 0.8 / 8, atol=1e-6)
        # edge noisy cell (2,0): 0.8 over 5 neighbours
        s = xy_to_state(2, 0)
        assert np.isclose(A[s, s], 0.2, atol=1e-6)
        assert np.isclose(A[xy_to_state(1, 0), s], 0.8 / 5, atol=1e-6)
        # planner layout: single padded modality
        assert self.A_planner.shape[0] == 1
        assert np.array_equal(self.A_planner[0], A)

    def test_goal_and_start(self):
        from environments.stochastic_maze import (
            GOAL_STATE, START_STATE, build_C, state_to_xy,
        )
        # notebook goal (3,5) -> (4,2); start (3,1) -> (0,2)
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
        observations are exact (the longer deterministic path)."""
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


class TestFogMazeTensors:
    def setup_method(self):
        from environments.stochastic_maze import maze_tensors
        self.A, self.B, self.meta = maze_tensors("fog")

    def test_layout_meta(self):
        from environments.stochastic_maze import FOG_CELLS, SINKS4
        assert self.meta["name"] == "fog"
        assert self.meta["sink_states"] == SINKS4
        assert self.meta["fog_cells"] == FOG_CELLS

    def test_four_sinks_absorbing(self):
        from environments.stochastic_maze import SINKS4, xy_to_state
        for (x, y) in SINKS4:
            s = xy_to_state(x, y)
            assert np.all(self.B[s, s, :] == 1.0)

    def test_fog_columns_uniform(self):
        from environments.stochastic_maze import FOG_CELLS, xy_to_state
        A = self.A[0]
        idxs = [xy_to_state(*c) for c in FOG_CELLS]
        for s in idxs:
            col = A[:, s]
            assert np.allclose(col.sum(), 1.0)
            for o in idxs:
                assert np.isclose(col[o], 0.2)
            assert np.allclose(col[np.setdiff1d(np.arange(len(col)), idxs)], 0.0)

    def test_drift_at_center_is_nonlethal(self):
        from environments.stochastic_maze import xy_to_state
        s = xy_to_state(1, 2)
        up, down, east = xy_to_state(1, 3), xy_to_state(1, 1), xy_to_state(2, 2)
        # intended EAST 0.4, drift N/S 0.3 each
        assert np.isclose(self.B[east, s, 1], 0.4, atol=1e-6)
        assert np.isclose(self.B[up, s, 1], 0.3, atol=1e-6)
        assert np.isclose(self.B[down, s, 1], 0.3, atol=1e-6)

    def test_fog_walls_layout(self):
        """fog-walls: same A and drift as fog, sinks replaced by walls that
        bump (stay) instead of absorbing; the wall cells are unreachable."""
        from environments.stochastic_maze import (
            FOG_WALLS, SINKS4, maze_tensors, xy_to_state,
        )
        A, B, meta = maze_tensors("fog-walls")
        assert meta["name"] == "fog-walls"
        assert meta["sink_states"] == ()
        assert meta["wall_states"] == FOG_WALLS == SINKS4
        assert np.array_equal(A, self.A)
        assert np.allclose(B.sum(axis=0), 1.0)
        E, S, N = 1, 2, 0
        # EAST from the south fog landing (1,1) into wall (2,1): stay
        s = xy_to_state(1, 1)
        assert B[s, s, E] == 1.0
        # NORTH from (1,1) is the drift cell (1,2), as in fog
        assert B[xy_to_state(1, 2), s, N] == 1.0
        # nothing ever enters a wall cell
        for (x, y) in FOG_WALLS:
            w = xy_to_state(x, y)
            others = np.setdiff1d(np.arange(B.shape[1]), [w])
            assert np.all(B[w][others] == 0.0)
        # the open middle row still leads to the goal
        s = xy_to_state(2, 2)
        assert B[xy_to_state(3, 2), s, E] == 1.0
        assert B[xy_to_state(4, 2), xy_to_state(3, 2), E] == 1.0

    def test_env_uses_fog_tables(self):
        from environments.stochastic_maze import FOG_CELLS, StochasticMazeEnv, xy_to_state
        env = StochasticMazeEnv(maze="fog", rng=np.random.default_rng(0))
        assert np.allclose(env.A, self.A[0])
        s = xy_to_state(*FOG_CELLS[0])
        assert np.isclose(env.A[xy_to_state(*FOG_CELLS[1]), s], 0.2)





# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestStochasticMazeEnv:

    def test_step_semantics(self):
        from environments.stochastic_maze import (
            START_STATE, StochasticMazeEnv, xy_to_state,
        )
        env = StochasticMazeEnv(rng=np.random.default_rng(0), maze="trap")
        obs = env.reset()
        assert env.state == START_STATE and obs == START_STATE  # clean cell
        obs, reward = env.step(0)           # NORTH -> (0, 3), deterministic
        assert env.state == xy_to_state(0, 3) and reward == 0.0
        assert obs == env.state             # clean cell: obs = state
        obs, reward = env.step(2)           # SOUTH -> back to start
        assert env.state == START_STATE

    def test_sink_traps(self):
        from environments.stochastic_maze import (
            StochasticMazeEnv, xy_to_state,
        )
        env = StochasticMazeEnv(rng=np.random.default_rng(0), maze="trap")
        sink = xy_to_state(3, 3)
        env.reset(start_state=sink)
        for a in range(4):
            env.step(a)
            assert env.state == sink

    def test_noisy_observation_frequencies(self):
        from environments.stochastic_maze import (
            StochasticMazeEnv, xy_to_state,
        )
        env = StochasticMazeEnv(rng=np.random.default_rng(0), maze="trap")
        env.reset(start_state=xy_to_state(2, 2))  # noisy: 20% correct
        hits = np.mean([env.observe() == env.state for _ in range(2000)])
        assert abs(hits - 0.2) < 0.05

    def test_goal_reward_via_safe_path(self):
        from environments.stochastic_maze import (
            GOAL_STATE, StochasticMazeEnv,
        )
        env = StochasticMazeEnv(rng=np.random.default_rng(0), maze="trap")
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
        result = run_scenario("known", n_episodes=2, max_steps=25, seed=0)
        assert [r["outcome"] for r in result["results"]] == ["GOAL", "GOAL"]
        for r in result["results"]:
            assert r["steps"] <= 20   # 8 = safe path; drift can be shorter

    def test_pref_sp_lured_into_risky_row(self):
        """The default drift layout is designed so the preference-driven
        planners enter the drift row from the start. Under the unary
        per-step preference (exponential utility) the conveyor's pull makes
        the risky row look cheap and both SP-Pref variants head East, while
        SP-EFE's ambiguity term still sends it up the safe top row."""
        import jax.numpy as jnp
        from environments.stochastic_maze import build_A, build_B, build_D, build_goal
        from agent import CLAMP
        from run_fog_walls import DEFAULT_HORIZON
        from spmp.sp_efe import sp_efe_planning, paper_scheme
        aA, aB = jnp.asarray(CLAMP * build_A()), jnp.asarray(CLAMP * build_B())
        goal, q0 = jnp.asarray(build_goal(4.0)), jnp.asarray(build_D())

        def plan_with(planner):
            return sp_efe_planning(
                q0, aA, aB, goal, horizon=DEFAULT_HORIZON, n_iterations=100,
                damping=0.5, per_step_goal=True,
                scheme=paper_scheme(planner)).action_dist

        pref, plan, cp = (plan_with(p) for p in ("SP-Pref", "SP-Pref-Plan", "SP-EFE"))
        assert int(np.argmax(pref)) == 1    # EAST into the conveyor
        assert int(np.argmax(plan)) == 1    # EAST, correction cannot fix it
        assert int(np.argmax(cp)) == 0    # NORTH toward the safe top row

    def test_pref_sp_planning_correction_terminal_only(self):
        """With the planning correction the terminal-only KL planner also
        leaves the start toward the top row (uncorrected it is ~uniform),
        and the corrected r(u|x) channel puts no mass on EAST."""
        import jax.numpy as jnp
        from environments.stochastic_maze import build_A, build_B, build_D, build_goal
        from agent import CLAMP
        from run_fog_walls import DEFAULT_HORIZON
        from spmp.sp_efe import sp_efe_planning, paper_scheme
        aA, aB = jnp.asarray(CLAMP * build_A()), jnp.asarray(CLAMP * build_B())
        goal, q0 = jnp.asarray(build_goal(4.0)), jnp.asarray(build_D())

        def plan_with(planner):
            return sp_efe_planning(
                q0, aA, aB, goal, horizon=DEFAULT_HORIZON, n_iterations=100,
                damping=0.5, scheme=paper_scheme(planner)).action_dist

        pref, plan = plan_with("SP-Pref"), plan_with("SP-Pref-Plan")
        assert np.ptp(np.asarray(pref)) < 0.05          # near-uniform
        assert int(np.argmax(plan)) in (0, 2)          # NORTH/SOUTH (symmetric)
        assert float(plan[1]) < 1e-3                   # no EAST


    def test_analytic_rejected_at_maze_horizon(self):
        from run_fog_walls import run_scenario
        try:
            run_scenario("known", n_episodes=1, planner="EFE",
                         maze="trap", quiet=True, collect_frames=False)
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
