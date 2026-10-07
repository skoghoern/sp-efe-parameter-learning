"""Learning the transition model θ_B: known risky shortcut vs unmapped detour.

The paper's experiment is ``--unmapped doors`` at H=7 (Figure 2d-f and the right
half of Table 1). A planner that never learns is capped at ``gate_p`` = 0.5, and
optimism is priced out by construction: a flat column spreads the successor over
all 20 cells, which reaches the goal within the horizon with probability 0.265,
below the gamble. So the safe route is only available to a planner that explores
its own model.

5×4 grid, identity A, B partially known. The mapped shortcut ``→→→→↑``
ends in a gate that reaches the goal w.p. ``gate_p`` and otherwise drops
into an absorbing dead end; the deterministic 7-step detour ``↑↑→→→→↓``
runs through six unmapped cells (24 flat Dirichlet columns). Terminal-only
preference, shrinking plan depth, argmax, stop at goal / dead end.

Scenarios
---------
- ``known``   : every B column clamped — the planning ceiling (1.00 detour).
- ``frozen``  : unmapped columns flat and never updated — the no-learning
  floor (everyone gambles: ≈ gate_p).
- ``learn-B`` : unmapped columns flat, Dirichlet counts persist across
  episodes. The benchmark.

Planners: the five MP planners, enumerated ``EFE`` / ``EFE-noamb-noig``
(4^H policies; H=7 default / H=8 mirror), and ``Random`` (uniform actions,
learning on).

Layouts (``--layout``): ``default`` (5×4 above) separates coupled from
``SP-Pref-Plan`` only — uncorrected ``SP-Pref`` walks the index-order detour.
``mirror`` (6×4: detour ``↑↑←←←←←↓``, no pocket; G at (0,2), top-left unmapped).
``--action-order`` relabels the action indices (robustness check for tie-break luck).
``--unmapped doors --layout mirror`` is the WEST-detour doors variant (no pocket).

The paper uses ``--unmapped doors`` / ``cells``, ``--layout mirror`` and
``--action-order`` (the Appendix C.2 robustness checks). ``--noisy-tv`` and
``--layout mirror-rest`` are inherited extras that no reported result uses; they
are off by default and can be ignored on a first read.

Examples
--------
    # paper: doors variant, learning θ_B (Figure 2d-f, Table 1 right)
    uv run python run_unmapped_detour.py --unmapped doors --compare \
        --episodes 300 --n-seeds 3
    uv run python run_unmapped_detour.py --scenario known
    uv run python run_unmapped_detour.py --scenario learn-B --episodes 200 --n-seeds 5
    uv run python run_unmapped_detour.py --compare
    uv run python run_unmapped_detour.py --compare --layout mirror --episodes 300 --n-seeds 3
    uv run python run_unmapped_detour.py --layout mirror --action-order WSEN --scenario learn-B
    uv run python run_unmapped_detour.py --scenario learn-B --planners SP-EFE-dyn SP-Pref-Plan Random
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

from environments.unmapped_detour import (
    ACTION_LETTERS, LAYOUTS, N_ACTIONS, DEFAULT_HORIZON, DEFAULT_LAYOUT,
    DEFAULT_STEPS, DEFAULT_GATE_P, DEFAULT_UNMAPPED_MODE, UNMAPPED_MODES,
    UnmappedDetourEnv, UnmappedDetourGrid,
)
from spmp.sp_efe import SP_EFE_IGC_PLANNERS
from agent import (
    CLAMP, FLAT, choose_action, filter_step, horizon_at, matrix_error,
    posterior_mean_B, update_counts_B,
)
from dispatch import (
    DEFAULT_DAMPING, DEFAULT_ITERS, IDENTIFIED_MASS, PRETTY, STYLE,
    PLANNERS as MP_PLANNERS, _plan,
)
from run_fog_walls import _trailing_mean

MAX_ANALYTIC_POLICIES = 4 ** 8  # mirror detour is H=8 (65536 policies)
STATE_ANALYTICS = ("EFE", "EFE-noamb-noig")
RANDOM = "Random"
# The ``-DeltaA-DeltaB`` labels (Δ_A / Δ_B dropped, channels kept) are ablations.
ALL_PLANNERS = MP_PLANNERS + SP_EFE_IGC_PLANNERS + STATE_ANALYTICS + (RANDOM,)
DEFAULT_PLANNERS = MP_PLANNERS + (RANDOM,)
SCENARIOS = ("known", "frozen", "learn-B")
DEFAULT_EPISODES = {"known": 20, "frozen": 20, "learn-B": 200}
DEFAULT_N_SEEDS = {"known": 1, "frozen": 1, "learn-B": 5}
DEFAULT_PREFERENCE = 4.0
N_BLOCKS = 10


# ---------------------------------------------------------------------------
# Metrics specific to this layout
# ---------------------------------------------------------------------------
def _unmapped_err(alpha_B, B_true, mask):
    """Mean |B̂ − B| over the unmapped columns only."""
    return matrix_error(alpha_B, B_true, axis=0, mask=mask)


def _unmapped_identified(alpha_B, B_true, grid, prior_count):
    """Unmapped columns whose MAP landing is right with mass ≥ 0.8."""
    hat = posterior_mean_B(alpha_B)
    a = np.asarray(alpha_B)
    n_id = 0
    for u, act in grid.unmapped_cols:
        if float(a[:, u, act].sum()) <= prior_count * grid.n_states + 0.5:
            continue
        pred = int(np.argmax(hat[:, u, act]))
        if (pred == int(np.argmax(B_true[:, u, act]))
                and hat[pred, u, act] >= IDENTIFIED_MASS):
            n_id += 1
    return n_id


def _cols_visited(col_visits, grid):
    """Number of flat columns tried at least once."""
    return int(sum(col_visits[s, a] > 0 for s, a in grid.unmapped_cols))






def _path_symbols(grid, path):
    """Arrows for agent action indices (respects the action relabelling)."""
    return "".join(grid.action_symbol(a) for a in path)


def _check_planner(planner, horizon):
    if planner not in ALL_PLANNERS:
        raise ValueError(f"unknown planner {planner!r}; choose from {ALL_PLANNERS}")
    if planner in STATE_ANALYTICS and 4 ** int(horizon) > MAX_ANALYTIC_POLICIES:
        raise ValueError(
            f"{planner} enumerates 4^{horizon} policies; "
            f"max {MAX_ANALYTIC_POLICIES}")




# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------
def run_episodes(planner, grid=None, scenario="learn-B", n_episodes=None,
                 max_steps=None, horizon=None,
                 n_iterations=DEFAULT_ITERS, damping=DEFAULT_DAMPING,
                 prior_count=FLAT, preference=DEFAULT_PREFERENCE,
                 shrinking_horizon=True, seed=0):
    """Multi-episode run; Dirichlet counts persist, pose and belief reset.

    Episodes end at the goal or the dead end (both absorbing); the last
    transition is folded into the counts so the goal-adjacent columns are
    learnable. Returns per-episode records plus summary statistics.
    ``horizon`` / ``max_steps`` default to the layout's detour length / +1.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    grid = grid or UnmappedDetourGrid()
    horizon = int(grid.detour_length if horizon is None else horizon)
    max_steps = int(grid.detour_length + 1 if max_steps is None else max_steps)
    _check_planner(planner, horizon)
    n_episodes = int(n_episodes or DEFAULT_EPISODES[scenario])
    if n_episodes < 1 or max_steps < 1 or horizon < 1:
        raise ValueError("n_episodes, max_steps and horizon must be >= 1")

    A_true = grid.build_A()
    B_true = grid.build_B()
    mask = grid.unmapped_mask()
    alpha_A = jnp.asarray(CLAMP * A_true)
    alpha_B = jnp.asarray(grid.build_alpha_B(
        prior_count=prior_count, clamp=CLAMP, known=(scenario == "known")))
    learn = scenario == "learn-B"
    goal = jnp.asarray(grid.build_goal(preference))
    D = jnp.asarray(grid.build_D())
    env = UnmappedDetourEnv(grid, seed=seed)
    rng_planner = np.random.default_rng(10_000 + seed)
    n_unmapped_cols = len(grid.unmapped_cols)

    col_visits = np.zeros((grid.n_states, N_ACTIONS), dtype=np.int64)
    results = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        q, last_action = D, -1
        path, states, q0 = [], [env.state], None
        outcome, steps_used = "timeout", max_steps

        def _observe_and_learn():
            """Fold the observation into the belief (and the B counts)."""
            nonlocal q, alpha_B
            q_prev = q
            _, q = filter_step(q, obs, alpha_A, alpha_B, last_action)
            if learn and last_action >= 0:
                alpha_B = update_counts_B(alpha_B, q_prev, q, last_action)

        for step in range(max_steps):
            _observe_and_learn()
            h = horizon_at(step, max_steps, horizon, shrinking_horizon)
            if planner == RANDOM:
                ad = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
                action = int(rng_planner.integers(N_ACTIONS))
            else:
                ad = np.asarray(_plan(
                    planner, q, alpha_A, alpha_B, goal, h,
                    n_iterations, damping, per_step_goal=False))
                action = choose_action(ad)
            if step == 0:
                q0 = ad
            col_visits[env.state, action] += 1
            path.append(action)
            obs, done, env_outcome = env.step(action)
            last_action = action
            states.append(env.state)
            if done:
                outcome, steps_used = env_outcome, step + 1
                break
        _observe_and_learn()   # terminal observation (goal / dead end / timeout)

        via_detour = (outcome == "GOAL"
                      and all(s in grid.detour_cells for s in states))
        results.append(dict(
            episode=ep, outcome=outcome, steps=steps_used, path=tuple(path),
            q0=None if q0 is None else np.asarray(q0),
            via_gate=grid.gate_idx in states,
            via_tv=(grid.noisy_tv and grid.tv_idx in states),
            via_detour=via_detour,
            optimal=(via_detour and steps_used == grid.detour_length),
            n_unmapped_cols=_cols_visited(col_visits, grid),
            err=_unmapped_err(alpha_B, B_true, mask),
            ident=_unmapped_identified(alpha_B, B_true, grid, prior_count),
        ))

    hits = np.array([r["outcome"] == "GOAL" for r in results], dtype=float)
    block = max(1, n_episodes // N_BLOCKS)
    blocks = [float(hits[i:i + block].mean()) for i in range(0, n_episodes, block)]
    hit_steps = [r["steps"] for r in results if r["outcome"] == "GOAL"]
    first = next((r["episode"] for r in results if r["outcome"] == "GOAL"), None)
    first_detour = next((r["episode"] for r in results if r["via_detour"]), None)
    last = results[-1]
    return dict(
        planner=planner, scenario=scenario, seed=seed, n_episodes=n_episodes,
        max_steps=max_steps, horizon=horizon, gate_p=grid.gate_p,
        results=results, hits=hits, blocks=blocks, block=block,
        first_hit=first, first_detour=first_detour,
        n_goal=int(hits.sum()), last_hits=int(hits[-block:].sum()),
        n_deadend=sum(r["outcome"] == "DEADEND" for r in results),
        n_rest=sum(r["outcome"] == "REST" for r in results),
        n_detour=sum(r["via_detour"] for r in results),
        n_gate=sum(r["via_gate"] for r in results),
        n_tv=sum(r["via_tv"] for r in results),
        last_detour=sum(r["via_detour"] for r in results[-block:]),
        mean_steps=(float(np.mean(hit_steps)) if hit_steps else None),
        err=last["err"], ident=last["ident"],
        n_unmapped_cols=last["n_unmapped_cols"], n_unmapped=n_unmapped_cols,
        col_visits=col_visits,
    )


# ---------------------------------------------------------------------------
# Compare / save / plot
# ---------------------------------------------------------------------------
def _fmt_ms(values, fmt="{:.2f}"):
    v = np.asarray(values, dtype=float)
    if v.size == 1:
        return fmt.format(v[0])
    return (fmt + "±" + fmt).format(v.mean(), v.std())


def compare(grid, planners, scenario, n_episodes, seeds,
            max_steps=None, horizon=None,
            n_iterations=DEFAULT_ITERS, damping=DEFAULT_DAMPING,
            prior_count=FLAT, preference=DEFAULT_PREFERENCE,
            shrinking_horizon=True, out=None):
    """Run every planner over the seeds and print a summary table."""
    planners = tuple(planners)
    seeds = tuple(int(s) for s in seeds)
    horizon = int(grid.detour_length if horizon is None else horizon)
    max_steps = int(grid.detour_length + 1 if max_steps is None else max_steps)
    for planner in planners:
        _check_planner(planner, horizon)
    hmark = "↓" if shrinking_horizon else ""
    print(f"unmapped-detour {grid.grid_x}x{grid.grid_y}  scenario={scenario}  "
          f"gate p={grid.gate_p}  prior={prior_count}  C={preference}@{grid.goal_xy}  "
          f"T={max_steps}  H={horizon}{hmark}  eps={n_episodes}  "
          f"seeds={list(seeds)}  iters={n_iterations}  damping={damping}")
    print(grid.depict())
    print(f"shortcut {_path_symbols(grid, grid.shortcut_path)} (p={grid.gate_p})  "
          f"detour {_path_symbols(grid, grid.detour_path)}  "
          f"optimism teleport value at H={horizon}: "
          f"{grid.optimism_teleport_prob(horizon):.3f}")
    print(f"planners: {', '.join(planners)}")
    print()

    runs = {p: [] for p in planners}
    for planner in planners:
        for seed in seeds:
            print(f"  running {planner} seed {seed} ...", flush=True)
            r = run_episodes(
                planner, grid=grid, scenario=scenario, n_episodes=n_episodes,
                max_steps=max_steps, horizon=horizon,
                n_iterations=n_iterations, damping=damping,
                prior_count=prior_count, preference=preference,
                shrinking_horizon=shrinking_horizon, seed=seed)
            runs[planner].append(r)
            first = "never" if r["first_hit"] is None else str(r["first_hit"])
            first_d = ("never" if r["first_detour"] is None
                       else str(r["first_detour"]))
            blocks = " ".join(f"{b:.2f}" for b in r["blocks"])
            q0 = r["results"][0]["q0"]
            q0s = ("—" if q0 is None
                   else "[" + ", ".join(f"{p:.2f}" for p in q0) + "]")
            tv = f"tv={r['n_tv']}  " if grid.noisy_tv else ""
            rest = f"rest={r['n_rest']}  " if grid.rest_idx else ""
            print(f"    goal={r['n_goal']}/{n_episodes}  detour={r['n_detour']}  "
                  f"gate={r['n_gate']}  {tv}deadend={r['n_deadend']}  {rest}"
                  f"first={first}  first-detour={first_d}  "
                  f"cols={r['n_unmapped_cols']}/{r['n_unmapped']}  "
                  f"ident={r['ident']}  errB={r['err']:.4f}  q0={q0s}")
            print(f"    blocks=[{blocks}]", flush=True)

    block = runs[planners[0]][0]["block"]
    print()
    tv_head = f"{'tv':12s} " if grid.noisy_tv else ""
    rest_head = f"{'rest':12s} " if grid.rest_idx else ""
    print(f"  {'planner':26s} {'goal':14s} {'last-'+str(block):10s} "
          f"{'detour':12s} {'last-det':10s} {'gate':12s} {tv_head}{'deadend':12s} "
          f"{rest_head}{'steps':10s} {'cols':10s} {'ident':10s} errB")
    for planner in planners:
        rs = runs[planner]
        steps = [r["mean_steps"] for r in rs if r["mean_steps"] is not None]
        tv_cell = (f"{_fmt_ms([r['n_tv'] for r in rs], '{:.1f}'):12s} "
                   if grid.noisy_tv else "")
        rest_cell = (f"{_fmt_ms([r['n_rest'] for r in rs], '{:.1f}'):12s} "
                     if grid.rest_idx else "")
        print(f"  {planner:26s} "
              f"{_fmt_ms([r['n_goal'] for r in rs], '{:.1f}'):14s} "
              f"{_fmt_ms([r['last_hits'] for r in rs], '{:.1f}'):10s} "
              f"{_fmt_ms([r['n_detour'] for r in rs], '{:.1f}'):12s} "
              f"{_fmt_ms([r['last_detour'] for r in rs], '{:.1f}'):10s} "
              f"{_fmt_ms([r['n_gate'] for r in rs], '{:.1f}'):12s} "
              f"{tv_cell}"
              f"{_fmt_ms([r['n_deadend'] for r in rs], '{:.1f}'):12s} "
              f"{rest_cell}"
              f"{(_fmt_ms(steps, '{:.1f}') if steps else '—'):10s} "
              f"{_fmt_ms([r['n_unmapped_cols'] for r in rs], '{:.1f}'):10s} "
              f"{_fmt_ms([r['ident'] for r in rs], '{:.1f}'):10s} "
              f"{_fmt_ms([r['err'] for r in rs], '{:.4f}')}")
    print(f"  gate ceiling for a pure gambler: {grid.gate_p:.2f}  "
          f"(= {grid.gate_p * n_episodes:.1f}/{n_episodes}); "
          f"unmapped columns {runs[planners[0]][0]['n_unmapped']}")
    print()
    print(f"  goal rate by {block}-ep block (mean over seeds):")
    for planner in planners:
        blocks = np.mean([r["blocks"] for r in runs[planner]], axis=0)
        print(f"  {planner:26s} " + " ".join(f"{b:.2f}" for b in blocks))
    print()

    if out:
        _save(runs, out, grid=grid, scenario=scenario, n_episodes=n_episodes)
    return runs


def _save(runs, out, grid, scenario, n_episodes):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = dict(scenario=scenario, n_episodes=n_episodes,
                   layout=grid.layout, action_order=grid.action_order,
                   gate_p=grid.gate_p, noisy_tv=grid.noisy_tv,
                   n_states=grid.n_states,
                   n_unmapped=len(grid.unmapped_cols),
                   unmapped_mode=grid.unmapped_mode,
                   detour_length=grid.detour_length,
                   shortcut_length=grid.shortcut_length)
    for planner, rs in runs.items():
        payload[f"{planner}|hit"] = np.stack([r["hits"] for r in rs])
        payload[f"{planner}|detour"] = np.array(
            [[x["via_detour"] for x in r["results"]] for r in rs], int)
        payload[f"{planner}|deadend"] = np.array(
            [[x["outcome"] == "DEADEND" for x in r["results"]] for r in rs], int)
        payload[f"{planner}|tv"] = np.array(
            [[x["via_tv"] for x in r["results"]] for r in rs], int)
        payload[f"{planner}|rest"] = np.array(
            [[x["outcome"] == "REST" for x in r["results"]] for r in rs], int)
        payload[f"{planner}|steps"] = np.array(
            [[x["steps"] for x in r["results"]] for r in rs], int)
        payload[f"{planner}|err"] = np.array(
            [[x["err"] for x in r["results"]] for r in rs], float)
        payload[f"{planner}|cols"] = np.array(
            [[x["n_unmapped_cols"] for x in r["results"]] for r in rs], int)
        payload[f"{planner}|ident"] = np.array(
            [[x["ident"] for x in r["results"]] for r in rs], int)
    np.savez(f"{out}.npz", **payload)
    _plot(runs, out, grid=grid, n_episodes=n_episodes)
    print(f"wrote {out}.png / {out}.pdf / {out}.npz")


def _plot(runs, out, grid, n_episodes):
    """Goal / detour rate, unmapped-column error and coverage (mean ± std)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_unmapped = len(grid.unmapped_cols)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.3))
    xs = np.arange(1, n_episodes + 1)
    win = max(1, n_episodes // N_BLOCKS)

    def _band(ax, series, st):
        arr = np.stack([_trailing_mean(s, win) for s in series])
        mean = arr.mean(axis=0)
        ax.plot(xs, mean, label=PRETTY.get(planner, planner), **st)
        if arr.shape[0] > 1:
            std = arr.std(axis=0)
            ax.fill_between(xs, mean - std, mean + std,
                            color=st.get("color", "0.5"), alpha=0.15, lw=0)

    for planner, rs in runs.items():
        st = dict(STYLE.get(planner, dict(color="black", ls="-")))
        st.pop("marker", None)
        _band(axes[0], [r["hits"] for r in rs], st)
        _band(axes[1], [np.array([x["via_detour"] for x in r["results"]], float)
                        for r in rs], st)
        err = np.array([[x["err"] for x in r["results"]] for r in rs])
        axes[2].plot(xs, err.mean(axis=0), **st)
        cols = np.array([[x["n_unmapped_cols"] for x in r["results"]] for r in rs])
        axes[3].plot(xs, cols.mean(axis=0), **st)
    axes[0].axhline(grid.gate_p, color="0.6", lw=0.8, ls="--")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel(f"goal rate ({win}-ep mean)")
    axes[0].set_title("return (dashed: gate p)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel(f"detour rate ({win}-ep mean)")
    axes[1].set_title("safe detour through unmapped cells")
    axes[2].set_ylabel(r"mean $|\hat B - B|$, unmapped columns")
    axes[2].set_title("table error where it matters")
    axes[3].axhline(n_unmapped, color="0.7", lw=0.6, ls=":")
    axes[3].set_ylim(0, n_unmapped + 1)
    axes[3].set_ylabel("unmapped (s,u) columns visited")
    axes[3].set_title("coverage")
    for ax in axes:
        ax.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="learn-B")
    ap.add_argument("--compare", action="store_true",
                    help="run known, frozen and learn-B in sequence")
    ap.add_argument("--layout", choices=tuple(LAYOUTS), default=DEFAULT_LAYOUT,
                    help="'default' (5x4, index-order detour) or 'mirror' "
                         "(6x4, WEST detour, G at (0,2), no pocket); "
                         "'mirror-nopocket' is an alias of mirror")
    ap.add_argument("--depict", action="store_true",
                    help="save layout PNGs (default and mirror) to figures/ "
                         "and exit")
    ap.add_argument("--unmapped", choices=UNMAPPED_MODES, default=DEFAULT_UNMAPPED_MODE,
                    help="'cells': every column of each unmapped cell is flat; "
                         "'doors': only the detour's own (cell, action) pairs "
                         "are flat (the other three actions are known bumps/"
                         "back). Doors starve SP-Pref's action-entropy bonus.")
    ap.add_argument("--action-order", default=ACTION_LETTERS,
                    help="relabel action indices, e.g. WSEN (default NESW); "
                         "robustness check for index-order tie-breaks")
    ap.add_argument("--gate-p", type=float, default=DEFAULT_GATE_P,
                    help="success probability of the known gate (default 0.5)")
    ap.add_argument("--noisy-tv", action="store_true",
                    help="add a known scrambler cell T=(2,0) (uniform over the "
                         "mapped free cells): separates parameter curiosity "
                         "from entropy seeking")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes per seed (default 20 known/frozen, 200 learn-B)")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="seeds (default 1 known/frozen, 5 learn-B)")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--steps", type=int, default=None,
                    help="episode length T (default: detour length + 1)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="plan-depth cap (default: detour length)")
    ap.add_argument("--fixed-horizon", action="store_true")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    ap.add_argument("--prior-count", type=float, default=FLAT)
    ap.add_argument("--preference", type=float, default=DEFAULT_PREFERENCE)
    ap.add_argument("--planners", nargs="+", choices=ALL_PLANNERS, default=None,
                    help=f"default: {' '.join(DEFAULT_PLANNERS)}")
    ap.add_argument("--out", default=None,
                    help="prefix for png/pdf/npz (empty string skips save)")
    args = ap.parse_args()
    if args.depict:
        from environments.unmapped_detour import save_layout_figures
        for path in save_layout_figures():
            print(f"wrote {path}")
        return

    grid = UnmappedDetourGrid(gate_p=args.gate_p, noisy_tv=args.noisy_tv,
                              layout=args.layout, action_order=args.action_order,
                              unmapped_mode=args.unmapped)
    planners = tuple(args.planners) if args.planners else DEFAULT_PLANNERS
    scenarios = SCENARIOS if args.compare else (args.scenario,)
    for scenario in scenarios:
        n_eps = args.episodes or DEFAULT_EPISODES[scenario]
        n_seeds = args.n_seeds or DEFAULT_N_SEEDS[scenario]
        seeds = tuple(range(args.seed, args.seed + max(1, n_seeds)))
        tag = (f"unmapped_detour_p{grid.gate_p:.2f}"
               f"{'' if grid.layout == DEFAULT_LAYOUT else '_' + grid.layout}"
               f"{'' if grid.unmapped_mode == DEFAULT_UNMAPPED_MODE else '_' + grid.unmapped_mode}"
               f"{'' if grid.action_order == ACTION_LETTERS else '_' + grid.action_order}"
               f"{'_tv' if grid.noisy_tv else ''}_{scenario}_H"
               f"{args.horizon if args.horizon is not None else grid.detour_length}")
        out = (os.path.join("figures", tag) if args.out is None
               else (f"{args.out}_{scenario}" if args.out else None))
        compare(grid, planners, scenario, n_eps, seeds,
                max_steps=args.steps, horizon=args.horizon,
                n_iterations=args.iters, damping=args.damping,
                prior_count=args.prior_count, preference=args.preference,
                shrinking_horizon=not args.fixed_horizon, out=out)


if __name__ == "__main__":
    main()
