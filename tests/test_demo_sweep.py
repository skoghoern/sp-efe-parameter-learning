"""The walkthrough demo doubles as a regression test on the scheme's behaviour.

If these break, the toy in ``demo_sweep.py`` no longer illustrates what its
prose claims, which is worse than a stale comment.
"""

import numpy as np
import jax.numpy as jnp


class TestDemoSweep:
    def _setup(self):
        import demo_sweep as d
        alpha_A, alpha_B, _ = d.build_tensors()
        goal = np.full(d.N_STATES, np.exp(-d.PREFERENCE), np.float32)
        goal[d.GOAL] = 1.0
        goal = jnp.asarray(goal / goal.sum())
        q0 = jnp.asarray(np.eye(d.N_STATES, dtype=np.float32)[d.START])
        return d, alpha_A, alpha_B, goal, q0

    def test_unknown_columns_are_flat_and_others_sharp(self):
        d, _, alpha_B, _, _ = self._setup()
        from spmp import dirichlet_log_tables
        _, log_mean_B = dirichlet_log_tables(alpha_B, axis=0)
        m = np.exp(np.asarray(log_mean_B))[:, :, d.RIGHT]
        for x in d.UNKNOWN_CELLS:
            assert np.allclose(m[:, x], 1.0 / d.N_STATES, atol=1e-5)
        # The goal column is absorbing, so it is sharp on itself.
        assert m[d.GOAL, d.GOAL] == 1.0

    def test_info_gain_weight_exceeds_one_only_where_unknown(self):
        d, _, alpha_B, _, _ = self._setup()
        from spmp import dirichlet_posterior_kl
        a = jnp.asarray(alpha_B)
        nbar = jnp.where(a > 0, a / jnp.maximum(a.sum(axis=0, keepdims=True),
                                                1e-12), 0.0)
        delta = np.asarray(dirichlet_posterior_kl(a, nbar, axis=0,
                                                  cond_axes=(1, 2)))
        w = np.exp(delta[:, :, d.RIGHT])
        for x in d.UNKNOWN_CELLS:
            assert w[:, x].max() > 10.0, "flat columns must carry novelty"
        known = [x for x in range(d.N_STATES) if x not in d.UNKNOWN_CELLS]
        assert np.all(w[:, known] <= 1.0 + 1e-3), "sharp columns must not"

    def test_sp_efe_explores_while_baselines_take_the_goal(self):
        """The point of the demo: novelty overrides a one-step goal."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import sp_efe_planning, paper_scheme
        ad = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, horizon=d.HORIZON, n_iterations=50,
            damping=0.5).action_dist)
        assert d.choose_action(ad, d.N_ACTIONS) == d.RIGHT
        for planner in ("SP-Pref-Plan", "SP-Pref"):
            ad = np.asarray(sp_efe_planning(
                q0, alpha_A, alpha_B, goal, horizon=d.HORIZON,
                n_iterations=50, damping=0.5,
                scheme=paper_scheme(planner)).action_dist)
            assert d.choose_action(ad, d.N_ACTIONS) == d.LEFT

    def test_stripping_all_corrections_recovers_pref_sp_plan(self):
        """SP-EFE with no novelty and no state-IG channel == SP-Pref-Plan
        (which additionally drops the ambiguity channel; identity A makes
        it inert)."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import Scheme, sp_efe_planning, paper_scheme
        stripped = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, horizon=d.HORIZON, n_iterations=50,
            damping=0.5, scheme=Scheme(state_info_gain=False, novelty_B=False,
                                       novelty_A_weight=0)).action_dist)
        pref = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, horizon=d.HORIZON, n_iterations=50,
            damping=0.5, scheme=paper_scheme("SP-Pref-Plan")).action_dist)
        assert np.allclose(stripped, pref, atol=1e-3)

    def test_dyn_channel_alone_still_follows_entropy(self):
        """The dagger caveat: the channel chases flat columns without novelty."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import sp_efe_planning, paper_scheme
        ad = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, horizon=d.HORIZON, n_iterations=50,
            damping=0.5, scheme=paper_scheme("SP-EFE-DeltaA-DeltaB")).action_dist)
        assert d.choose_action(ad, d.N_ACTIONS) == d.RIGHT

    def test_result_contents_and_input_errors(self):
        """PlanResult exposes beliefs, kernel and channels with the documented
        shapes; the objective trace is one finite scalar per sweep and term;
        bad inputs fail at trace time with a readable message."""
        import pytest
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import Scheme, sp_efe_planning, paper_scheme
        T, S, U = d.HORIZON, d.N_STATES, d.N_ACTIONS
        res = sp_efe_planning(
            q0, alpha_A, alpha_B, goal, horizon=T, n_iterations=10,
            damping=0.5, track_objective=True)
        assert res.action_dist.shape == (U,) and res.q_u.shape == (T, U)
        assert res.q_x.shape == (T + 1, S) and res.log_g_dyn.shape == (T, S, S, U)
        assert res.state.log_r_xxu.shape == (T, S, S, U)
        assert np.allclose(np.asarray(res.q_x).sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(res.q_x[0], q0, atol=1e-6)      # x_0 is known
        assert np.allclose(res.q_u[0], res.action_dist)
        trace = res.objective_trace
        assert "total" in trace and len(trace) == 12
        for v in trace.values():
            assert v.shape == (10,) and np.all(np.isfinite(np.asarray(v)))
        with pytest.raises(ValueError, match="goal"):
            sp_efe_planning(q0, alpha_A, alpha_B, goal[:-1], horizon=T,
                            n_iterations=1)
        with pytest.raises(ValueError, match="ambiguity"):
            Scheme(ambiguity=False)                         # Δ_A needs r(y|x)
        with pytest.raises(ValueError, match="goal_in_sweeps"):
            sp_efe_planning(q0, alpha_A, alpha_B, goal, horizon=T,
                            n_iterations=1, per_step_goal=True,
                            scheme=Scheme(ambiguity=False, state_info_gain=False,
                                          planning_correction=False,
                                          novelty_B=False, novelty_A_weight=0))
        with pytest.raises(ValueError, match="unknown planner"):
            paper_scheme("SP-Nope")

    def test_ablations_zero_their_objective_term(self):
        """Dropping Δ_B / Δ_A zeroes exactly the novelty terms of the objective."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import sp_efe_planning, paper_scheme

        def trace(label):
            return sp_efe_planning(
                q0, alpha_A, alpha_B, goal, horizon=d.HORIZON, n_iterations=5,
                damping=0.5, track_objective=True,
                scheme=paper_scheme(label)).objective_trace

        full = trace("SP-EFE")
        assert np.all(np.asarray(full["dyn_info"]) < 0)        # novelty is a bonus
        assert np.all(np.asarray(full["obs_info"]) <= 0)
        no_B = trace("SP-EFE-DeltaB")
        assert np.all(np.asarray(no_B["dyn_info"]) == 0)
        assert np.all(np.asarray(no_B["obs_info"]) <= 0)        # Δ_A still on
        none = trace("SP-EFE-DeltaA-DeltaB")
        assert np.all(np.asarray(none["dyn_info"]) == 0)
        assert np.all(np.asarray(none["obs_info"]) == 0)

    def test_sp_pref_is_plain_forward_backward(self):
        """With every correction off the planner is Bethe sum-product on the
        mean tables: q(u_1) must match an independent numpy forward/backward
        with the terminal preference as the only non-uniform unary."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import sp_efe_planning, paper_scheme
        T, U = d.HORIZON, d.N_ACTIONS
        B = np.asarray(alpha_B, dtype=np.float64)
        B = B / B.sum(axis=0, keepdims=True)                   # m_B (x', x, u)
        A = np.asarray(alpha_A, dtype=np.float64)
        A = A / A.sum(axis=1, keepdims=True)                   # m_A (f, y, x)
        obs_to_x = np.prod(A.sum(axis=1), axis=0)              # Σ_y m_A, uniform p̂(y)
        p_u = np.full(U, 1.0 / U)

        beta = np.asarray(goal, dtype=np.float64) * obs_to_x   # μ←(x_T)·φ(x_T)
        for _ in range(T - 1):                                 # back to x_1
            beta = np.einsum("yxu,y,u->x", B, beta, p_u) * obs_to_x
        q_u1 = np.einsum("x,yxu,y,u->u", np.asarray(q0, np.float64) * obs_to_x,
                         B, beta, p_u)
        q_u1 /= q_u1.sum()

        ad = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, horizon=T, n_iterations=50,
            damping=0.5, scheme=paper_scheme("SP-Pref")).action_dist)
        assert np.allclose(ad, q_u1, atol=1e-5)

    def test_paper_rows_match_readme_and_groups(self):
        """The README table and the planner groups list exactly PAPER_ROWS."""
        import pathlib
        from spmp import (PAPER_ROWS, SP_EFE_IGC_PLANNERS, SP_EFE_PLANNERS,
                          SP_PREF_PLANNERS)
        assert set(SP_EFE_PLANNERS + SP_EFE_IGC_PLANNERS + SP_PREF_PLANNERS) \
            == set(PAPER_ROWS)
        readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
        for label in PAPER_ROWS:
            assert f"| `{label}`" in readme, f"{label} missing from README table"

    def test_demo_runs_end_to_end(self):
        import demo_sweep
        demo_sweep.main()
