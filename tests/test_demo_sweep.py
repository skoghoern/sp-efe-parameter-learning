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
        from spmp import sp_pref_planning, sp_efe_planning
        ad = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, d.HORIZON, 50, damping=0.5)[0])
        assert d.choose_action(ad, d.N_ACTIONS) == d.RIGHT
        for corr in (True, False):
            ad = np.asarray(sp_pref_planning(
                q0, alpha_A, alpha_B, goal, d.HORIZON, 50, damping=0.5,
                planning_correction=corr)[0])
            assert d.choose_action(ad, d.N_ACTIONS) == d.LEFT

    def test_stripping_all_corrections_recovers_pref_sp_plan(self):
        """SP-EFE with no tilts and no dyn channel == SP-Pref-Plan."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import sp_pref_planning, sp_efe_planning
        stripped = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, d.HORIZON, 50, damping=0.5,
            dyn_channel=False, dyn_tilt_coef=0, obs_tilt_coef=0)[0])
        pref = np.asarray(sp_pref_planning(
            q0, alpha_A, alpha_B, goal, d.HORIZON, 50, damping=0.5,
            planning_correction=True)[0])
        assert np.allclose(stripped, pref, atol=1e-3)

    def test_dyn_channel_alone_still_follows_entropy(self):
        """The dagger caveat: the channel chases flat columns without novelty."""
        d, alpha_A, alpha_B, goal, q0 = self._setup()
        from spmp import sp_efe_planning
        ad = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, d.HORIZON, 50, damping=0.5,
            dyn_channel=True, dyn_tilt_coef=0, obs_tilt_coef=0)[0])
        assert d.choose_action(ad, d.N_ACTIONS) == d.RIGHT

    def test_demo_runs_end_to_end(self):
        import demo_sweep
        demo_sweep.main()
