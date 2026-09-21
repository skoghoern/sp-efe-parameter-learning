"""Tests for the unmapped-detour grid (known gate vs unmapped detour)."""

import numpy as np


class TestUnmappedDetourTensors:
    def setup_method(self):
        from environments.unmapped_detour import UnmappedDetourGrid
        self.grid = UnmappedDetourGrid()

    def test_layout_defaults(self):
        g = self.grid
        assert g.n_states == 20 and g.n_obs == 20
        assert g.start_xy == (0, 1) and g.goal_xy == (4, 2)
        assert g.gate_xy == (4, 1) and g.deadend_xy == (4, 0)
        assert g.detour_length == 7 and g.shortcut_length == 5
        assert len(g.unmapped_idx) == 6

    def test_identity_A(self):
        A = self.grid.build_A()
        assert A.shape == (1, 20, 20)
        assert np.allclose(A[0], np.eye(20))

    def test_B_is_stochastic_and_gate_splits(self):
        from environments.unmapped_detour import NORTH, SOUTH
        g = self.grid
        B = g.build_B()
        assert B.shape == (20, 20, 4)
        assert np.allclose(B.sum(axis=0), 1.0)
        # Gate: NORTH from X goes to G w.p. p and to D otherwise.
        assert np.isclose(B[g.goal_idx, g.gate_idx, NORTH], g.gate_p)
        assert np.isclose(B[g.deadend_idx, g.gate_idx, NORTH], 1 - g.gate_p)
        # Every other column is deterministic.
        det = np.delete(B.reshape(20, -1), g.gate_idx * 4 + NORTH, axis=1)
        assert np.all(det.max(axis=0) == 1.0)
        # Goal and dead end are absorbing.
        for s in (g.goal_idx, g.deadend_idx):
            assert np.all(B[s, s, :] == 1.0)
        # Walls bump: SOUTH from (1,3) stays.
        wall_top = g.xy_to_state(1, 3)
        assert B[wall_top, wall_top, SOUTH] == 1.0

    def test_detour_is_unmapped_only_and_safe(self):
        g = self.grid
        s = g.start_idx
        for a in g.detour_path:
            s = g.move(s, a)
            assert s == g.goal_idx or g.is_unmapped(s)
        assert s == g.goal_idx
        for u in g.unmapped_idx:
            for a in range(4):
                assert g.move(u, a) != g.deadend_idx
                assert not g.is_gate(u, a)

    def test_alpha_B_clamps_mapped_and_flattens_unmapped(self):
        g = self.grid
        alpha = g.build_alpha_B(prior_count=0.1, clamp=1e6)
        mask = g.unmapped_mask()
        assert np.all(alpha[mask] == np.float32(0.1))
        B = g.build_B()
        assert np.allclose(alpha[~mask], 1e6 * B[~mask])
        known = g.build_alpha_B(known=True)
        assert np.allclose(known, 1e6 * B)

    def test_gate_p_validation_and_optimism_bound(self):
        from environments.unmapped_detour import UnmappedDetourGrid
        for bad in (-0.1, 1.5):
            try:
                UnmappedDetourGrid(gate_p=bad)
            except ValueError:
                continue
            raise AssertionError(f"gate_p={bad} should fail")
        g = self.grid
        assert np.isclose(g.optimism_teleport_prob(7), 1 - (19 / 20) ** 6)
        assert g.gate_p > g.optimism_teleport_prob(8)

    def test_noisy_tv_variant(self):
        from environments.unmapped_detour import UnmappedDetourGrid
        g = UnmappedDetourGrid(noisy_tv=True)
        B = g.build_B()
        assert np.allclose(B.sum(axis=0), 1.0)
        tv = g.xy_to_state(2, 0)
        for a in range(4):
            col = B[:, tv, a]
            assert np.isclose(col[list(g.tv_targets)].sum(), 1.0)
            assert np.allclose(col[list(g.tv_targets)], 1.0 / 8)
        # The scrambler is mapped (clamped) and off the detour.
        alpha = g.build_alpha_B()
        assert np.allclose(alpha[:, tv, :], 1e6 * B[:, tv, :])
        assert g.detour_path == UnmappedDetourGrid().detour_path
        assert g.shortcut_path == UnmappedDetourGrid().shortcut_path

    def test_start_prior_and_goal(self):
        D = self.grid.build_D()
        assert D[self.grid.start_idx] == 1.0 and np.isclose(D.sum(), 1.0)
        goal = self.grid.build_goal(4.0)
        assert goal.argmax() == self.grid.goal_idx and np.isclose(goal.sum(), 1.0)

    def test_doors_mode_flattens_only_detour_columns(self):
        from environments.unmapped_detour import UnmappedDetourGrid
        g = UnmappedDetourGrid(unmapped_mode="doors")
        assert len(g.unmapped_cols) == 6            # one door per detour cell
        states = [g.start_idx]
        for a in g.detour_path:
            states.append(g.move(states[-1], a))
        assert set(g.unmapped_cols) == set(zip(states[1:-1], g.detour_path[1:]))
        alpha = g.build_alpha_B(prior_count=0.1)
        mask = g.unmapped_mask()
        assert mask.sum() == 6 * g.n_states
        assert np.all(alpha[mask] == np.float32(0.1))
        # Off-door columns of detour cells are clamped (known bumps / back).
        B = g.build_B()
        s, a = g.unmapped_cols[1]
        other = [b for b in range(4) if b != a]
        assert np.allclose(alpha[:, s, other], 1e6 * B[:, s, other])

    def test_mirror_layout_and_rest_cell(self):
        from environments.unmapped_detour import (
            NORTH, EAST, WEST, UnmappedDetourGrid)
        g = UnmappedDetourGrid(layout="mirror")
        assert g.n_states == 24 and g.goal_xy == (0, 2) and g.shortcut_length == 6
        assert g.detour_length == 8
        assert not g.is_unmapped(g.goal_idx)
        assert g.is_unmapped(g.xy_to_state(0, 3))   # vacated top-left
        assert (4, 0) not in [g.state_to_xy(s) for s in g.unmapped_idx]
        # (5,3): NORTH and EAST bump, only WEST (last index) progresses.
        corner = g.xy_to_state(5, 3)
        assert g.move(corner, NORTH) == corner and g.move(corner, EAST) == corner
        assert g.move(corner, WEST) == g.xy_to_state(4, 3)
        from environments.unmapped_detour import SOUTH
        assert g.detour_path == (NORTH, NORTH, WEST, WEST, WEST, WEST, WEST, SOUTH)
        assert np.allclose(g.build_B().sum(axis=0), 1.0)
        doors = UnmappedDetourGrid(layout="mirror", unmapped_mode="doors")
        assert len(doors.unmapped_cols) == 7
        # Rest cell: known, absorbing, and reported as its own outcome.
        from environments.unmapped_detour import UnmappedDetourEnv
        gr = UnmappedDetourGrid(layout="mirror-rest")
        rest = gr.xy_to_state(4, 0)
        assert rest in gr.rest_idx and gr.move(rest, NORTH) == rest
        assert np.allclose(gr.build_alpha_B()[:, rest, :], 1e6 * gr.build_B()[:, rest, :])
        env = UnmappedDetourEnv(gr, seed=0)
        env.state = gr.xy_to_state(4, 1)
        _, done, outcome = env.step(2)             # SOUTH into R
        assert done and outcome == "REST"

    def test_depict_ax_draws_both_layouts(self):
        import matplotlib
        matplotlib.use("Agg")
        from environments.unmapped_detour import UnmappedDetourGrid
        for name in ("default", "mirror"):
            ax = UnmappedDetourGrid(layout=name).depict_ax()
            assert ax is not None
            n_cells = {"default": 20, "mirror": 24}[name]
            rects = [p for p in ax.patches
                     if type(p).__name__ == "Rectangle" and p.get_width() == 1]
            assert len(rects) == n_cells

    def test_action_order_relabels_indices_only(self):
        from environments.unmapped_detour import UnmappedDetourGrid, parse_action_order
        assert parse_action_order("WSEN") == (3, 2, 1, 0)
        for bad in ("NNSW", "NES", "ABCD"):
            try:
                parse_action_order(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} should fail")
        base = UnmappedDetourGrid()
        perm = UnmappedDetourGrid(action_order="WSEN")
        # Same physics under the relabelling: B columns are permuted.
        assert np.allclose(perm.build_B()[:, :, 0], base.build_B()[:, :, 3])
        assert perm.gate_action == 3                # NORTH is index 3 in WSEN
        assert "".join(perm.action_symbol(a) for a in perm.detour_path) == "↑↑→→→→↓"
        assert perm.detour_path == (3, 3, 2, 2, 2, 2, 1)


class TestUnmappedDetourEnv:
    def test_detour_reaches_goal(self):
        from environments.unmapped_detour import UnmappedDetourEnv, UnmappedDetourGrid
        grid = UnmappedDetourGrid()
        env = UnmappedDetourEnv(grid, seed=0)
        assert env.reset() == grid.start_idx
        for a in grid.detour_path:
            obs, done, outcome = env.step(a)
        assert done and outcome == "GOAL" and env.state == grid.goal_idx

    def test_gate_frequency_matches_p(self):
        from environments.unmapped_detour import (
            NORTH, UnmappedDetourEnv, UnmappedDetourGrid)
        grid = UnmappedDetourGrid(gate_p=0.5)
        env = UnmappedDetourEnv(grid, seed=1)
        goals = 0
        n = 400
        for _ in range(n):
            env.reset()
            env.state = grid.gate_idx
            _, done, outcome = env.step(NORTH)
            assert done and outcome in ("GOAL", "DEADEND")
            goals += outcome == "GOAL"
        assert abs(goals / n - 0.5) < 0.08


class TestUnmappedDetourRunner:
    def test_known_detour_for_coupled_and_corr(self):
        from run_unmapped_detour import run_episodes
        for planner in ("coupled-nodyn", "pref-sp-plan"):
            r = run_episodes(planner, scenario="known", n_episodes=1,
                             n_iterations=20)
            rec = r["results"][0]
            assert rec["outcome"] == "GOAL" and rec["via_detour"]
            assert rec["steps"] == 7

    def test_frozen_corr_gambles(self):
        from run_unmapped_detour import run_episodes
        r = run_episodes("pref-sp-plan", scenario="frozen", n_episodes=2,
                         n_iterations=20)
        assert all(rec["via_gate"] for rec in r["results"])
        assert r["n_unmapped_cols"] == 0

    def test_doors_mode_known_detour_and_column_accounting(self):
        from environments.unmapped_detour import UnmappedDetourGrid
        from run_unmapped_detour import run_episodes
        grid = UnmappedDetourGrid(unmapped_mode="doors")
        r = run_episodes("coupled-nodyn", grid=grid, scenario="known",
                         n_episodes=1, n_iterations=20)
        rec = r["results"][0]
        assert rec["outcome"] == "GOAL" and rec["via_detour"]
        assert r["n_unmapped"] == 6 and rec["n_unmapped_cols"] == 6

    def test_learn_b_updates_unmapped_counts_only(self):
        from run_unmapped_detour import run_episodes
        r = run_episodes("random", scenario="learn-B", n_episodes=3,
                         n_iterations=5, seed=0)
        assert r["n_unmapped_cols"] >= 1
        assert r["results"][-1]["err"] < r["results"][0]["err"] + 1e-9
