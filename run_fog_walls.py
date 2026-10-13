"""Learning the observation model θ_A: SP-EFE vs the preference baselines.

The paper's experiment is fog-walls at H=8 (Figure 2a-c and the left half of
Table 1). The direct 4-step route crosses a drift cell and an ambiguous fog
block; the 8-step top row identifies every state. Only a planner that *knows*
the top row is unambiguous will pay the extra four steps for it, so learning θ_A
changes the route -- which is the point.

The maze is the 5×5 fog-walls layout (layout horizon 11; the paper uses H=8).
Four interior cells are impassable walls, so mislocalization costs bumps
rather than the episode, and a novelty-driven learner survives long enough
to learn which cells are foggy.

Planners come from ``dispatch.py``; the enumerated analytic ones need 4^H
policies, so they run up to H=8 (65536 policies).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from environments.stochastic_maze import (
    ACTIONS, MAZE_LAYOUTS, N_ACTIONS,
    StochasticMazeEnv, build_D, build_goal, maze_tensors,
    state_to_xy, xy_to_state,
)
from spmp.sp_efe import SP_EFE_IGC_PLANNERS
from agent import (
    FLAT, choose_action, clamped_alpha, filter_step, flat_alpha,
    horizon_at, matrix_error, posterior_mean_A, posterior_mean_B,
    update_counts_A, update_counts_B,
)
from dispatch import ALL_PLANNERS as BASE_PLANNERS, SI_PLANNERS, _plan
from figures.paths import split_stem
from viz import save_gif

# agent.FLAT (0.1) is the Dirichlet prior count for the reported learn-A runs.
DEFAULT_ITERS = 100
DEFAULT_DAMPING = 0.5
DEFAULT_HORIZON = 11         # fog-walls layout horizon; the paper uses H=8
DEFAULT_SHRINKING_HORIZON = True
DEFAULT_FIXED_STEPS = 25     # episode length under a fixed H; shrinking H uses T = H
DEFAULT_EPISODES = {"known": 3, "learn-A": 60, "learn-B": 20}
DEFAULT_PREFERENCE = 4.0
COMPARE_KNOWN_SEEDS = 10
COMPARE_LEARN_B_EPISODES = 20
COMPARE_LEARN_A_SEEDS = 3
LEARN_A_BLOCKS = ((1, 10), (11, 20), (21, 30), (31, 60))
MP_PLANNERS = ("SP-EFE-dyn", "SP-EFE", "SP-EFE-DeltaB",
               "SP-Pref", "SP-Pref-Plan")
# Ablations: the ``-DeltaA-DeltaB`` labels drop the Dirichlet tilts Δ_A / Δ_B
# (channels and ambiguity only); ``Random`` is the uniform-action floor
# (learning on).
RANDOM = "Random"
ABLATION_PLANNERS = SP_EFE_IGC_PLANNERS + (RANDOM,)
ANALYTIC_PLANNERS = ("EFE", "EFE-noamb-noig")
PLANNERS = MP_PLANNERS + ANALYTIC_PLANNERS + SI_PLANNERS + ABLATION_PLANNERS
MAX_ANALYTIC_POLICIES = 4 ** 8  # fog-walls is H=8 (65536 policies)


def _default_compare_planners(mazes, horizon=None):
    """Analytic EFE when every maze's H is tractable; else MP only."""
    hs = [maze_tensors(m)[2]["horizon"] if horizon is None else horizon
          for m in mazes]
    return (MP_PLANNERS + ANALYTIC_PLANNERS
            if all(4 ** int(h) <= MAX_ANALYTIC_POLICIES for h in hs)
            else MP_PLANNERS)


# Errors are reported over every entry of the matrix being learned.
def _a_err(alpha_A, A_true):
    return matrix_error(alpha_A, A_true, axis=1)


def _b_err(alpha_B, B_true):
    return matrix_error(alpha_B, B_true, axis=0)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------
def _is_top_row(path_xy, cells):
    allowed = set(cells)
    return bool(path_xy) and all(xy in allowed for xy in path_xy)


def _check_planner(planner, horizon):
    if planner not in BASE_PLANNERS + ABLATION_PLANNERS:
        raise ValueError(
            f"unknown planner {planner!r}; choose from {PLANNERS}")
    if planner in ANALYTIC_PLANNERS:
        n = 4 ** int(horizon)
        if n > MAX_ANALYTIC_POLICIES:
            raise ValueError(
                f"{planner} enumerates {n} policies at H={horizon}; "
                f"use an MP planner (max {MAX_ANALYTIC_POLICIES})")


def run_scenario(scenario, n_episodes=None, max_steps=None,
                 horizon=None, n_iterations=DEFAULT_ITERS,
                 damping=DEFAULT_DAMPING, seed=0, prior_count=FLAT,
                 preference=DEFAULT_PREFERENCE,
                 per_step_goal=None, stop_at_goal=None,
                 action_select=None,
                 shrinking_horizon=DEFAULT_SHRINKING_HORIZON,
                 planner="SP-EFE", maze="fog-walls", quiet=False,
                 collect_frames=True):
    """Run one scenario; returns dict with frames and metrics.

    ``horizon`` defaults to the layout's planning depth (11). Analytic EFE
    is allowed only when 4^H ≤ ``MAX_ANALYTIC_POLICIES``.
    ``max_steps`` is the episode length T; by default T = H under a
    shrinking horizon (the paper's protocol) and ``DEFAULT_FIXED_STEPS``
    otherwise. Passing it decouples T from H, e.g. for a horizon sweep.
    """
    if scenario not in ("known", "learn-A", "learn-B"):
        raise ValueError(scenario)
    if maze not in MAZE_LAYOUTS:
        raise ValueError(f"unknown maze {maze!r}; choose from {MAZE_LAYOUTS}")
    A_true, B_true, layout = maze_tensors(maze)
    n_states = int(layout["n_states"])
    gx = int(layout["grid_x"])
    goal_state = int(layout["goal_state"])
    horizon = int(layout["horizon"] if horizon is None else horizon)
    _check_planner(planner, horizon)
    n_episodes = n_episodes or DEFAULT_EPISODES[scenario]
    if stop_at_goal is None:
        stop_at_goal = scenario != "learn-B"
    if per_step_goal is None:
        per_step_goal = False
    if action_select is None:
        action_select = "argmax"
    if action_select not in ("argmax", "sample"):
        raise ValueError(action_select)

    sink_idx = {xy_to_state(*c, grid_x=gx) for c in layout["sink_states"]}

    if scenario == "learn-A":
        alpha_A = flat_alpha((1, n_states, n_states), prior_count)
    else:
        alpha_A = clamped_alpha(A_true)
    if scenario == "learn-B":
        alpha_B = flat_alpha((n_states, n_states, N_ACTIONS), prior_count)
    else:
        alpha_B = clamped_alpha(B_true)

    goal = jnp.asarray(build_goal(
        preference=preference, goal_state=goal_state, n_states=n_states))
    rng = np.random.default_rng(seed)
    env = StochasticMazeEnv(rng=rng, maze=maze, A=A_true[0], B=B_true)
    if max_steps is None:
        max_steps = horizon if shrinking_horizon else DEFAULT_FIXED_STEPS
    max_steps = int(max_steps)

    frames, results = [], []
    if not quiet:
        print(f"stochastic-maze/{maze}/{scenario} — {planner}  H={horizon}"
              f"{'↓' if shrinking_horizon else ''}  T={max_steps}"
              f"  iters={n_iterations}  damping={damping}"
              f"  per_step_goal={per_step_goal}  action_select={action_select}"
              f"  prior_count={prior_count}  episodes={n_episodes}")

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        q = jnp.asarray(build_D(
            start_state=layout["start_state"], n_states=n_states))
        last_action = -1
        outcome, steps_used = "timeout", max_steps
        path_xy = [state_to_xy(env.state, grid_x=gx)]
        if not quiet:
            print(f"-- ep{ep} --")

        for step in range(1, max_steps + 1):
            q_prev = q
            q_pred, q = filter_step(q, obs, alpha_A, alpha_B, last_action)
            if scenario == "learn-A":
                alpha_A = update_counts_A(alpha_A, obs, q)
            elif scenario == "learn-B" and last_action >= 0:
                alpha_B = update_counts_B(alpha_B, q_prev, q, last_action)

            # step is 1-based here; horizon_at takes the 0-based index.
            h = horizon_at(step - 1, max_steps, horizon, shrinking_horizon)
            if planner == RANDOM:
                # Uniform-action floor: no planning, learning still on.
                ad = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
                action = int(rng.integers(N_ACTIONS))
            else:
                ad = np.asarray(_plan(
                    planner, q, alpha_A, alpha_B, goal, h,
                    n_iterations, damping, per_step_goal))
                if action_select == "sample":
                    p = np.maximum(ad, 0)
                    action = int(rng.choice(N_ACTIONS, p=p / p.sum()))
                else:
                    action = choose_action(ad)

            err = (_a_err(alpha_A, A_true) if scenario == "learn-A"
                   else _b_err(alpha_B, B_true) if scenario == "learn-B"
                   else 0.0)
            if collect_frames:
                frames.append(dict(
                    episode=ep, step=step, state=env.state,
                    q=np.asarray(q), q_pred=np.asarray(q_pred),
                    action=action, action_dist=ad, err=err,
                    A=posterior_mean_A(alpha_A), B=posterior_mean_B(alpha_B), arrived=False))

            obs, reward = env.step(action)
            last_action = action
            path_xy.append(state_to_xy(env.state, grid_x=gx))
            if not quiet:
                print(f"  ep{ep} step {step:02d}: a={ACTIONS[action]:5s} "
                      f"-> {state_to_xy(env.state, grid_x=gx)} "
                      f"obs={state_to_xy(obs, grid_x=gx)}"
                      + (f"  err={err:.3f}" if scenario != "known" else ""))

            if env.state == goal_state and outcome == "timeout":
                steps_used = step
                outcome = "GOAL"
            if env.state in sink_idx and outcome == "timeout":
                steps_used = step
                outcome = "SINK"
            if env.state == goal_state and stop_at_goal:
                q_prev = q
                q_pred, q = filter_step(q, obs, alpha_A, alpha_B, last_action)
                if scenario == "learn-A":
                    alpha_A = update_counts_A(alpha_A, obs, q)
                elif scenario == "learn-B":
                    alpha_B = update_counts_B(alpha_B, q_prev, q,
                                              last_action)
                break
            if env.state in sink_idx and stop_at_goal:
                break

        err = (_a_err(alpha_A, A_true) if scenario == "learn-A"
               else _b_err(alpha_B, B_true) if scenario == "learn-B"
               else 0.0)
        if collect_frames:
            frames.append(dict(
                episode=ep, step=steps_used + 1, state=env.state,
                q=np.asarray(q), q_pred=np.asarray(q),
                action=None, action_dist=(frames[-1]["action_dist"]
                                          if frames else np.zeros(N_ACTIONS)),
                err=err, A=posterior_mean_A(alpha_A), B=posterior_mean_B(alpha_B),
                arrived=True, outcome=outcome))
        results.append(dict(
            episode=ep, outcome=outcome, steps=steps_used, err=err,
            sink_end=env.state in sink_idx,
            top_row=_is_top_row(path_xy, layout["top_row_cells"]),
            path=path_xy))
        if not quiet:
            print(f"  ep{ep}: {outcome} in {steps_used} steps"
                  + (f"  err={err:.3f}" if scenario != "known" else ""))

    goals = sum(r["outcome"] == "GOAL" for r in results)
    if not quiet:
        print(f"\n{maze}/{scenario}: {goals}/{n_episodes} reached goal  "
              f"avg steps={np.mean([r['steps'] for r in results]):.1f}"
              + (f"  final err={results[-1]['err']:.3f}"
                 if scenario != "known" else ""))
    return dict(scenario=scenario, planner=planner, maze=maze,
                frames=frames, results=results, horizon=horizon,
                per_step_goal=per_step_goal,
                shrinking_horizon=shrinking_horizon,
                layout=layout, final_err=results[-1]["err"])



def _agg_known(runs):
    n = sum(len(run["results"]) for run in runs)
    goals = sum(r["outcome"] == "GOAL" for run in runs for r in run["results"])
    sinks = sum(r["sink_end"] for run in runs for r in run["results"])
    top = sum(r["top_row"] for run in runs for r in run["results"])
    hit_steps = [r["steps"] for run in runs for r in run["results"]
                 if r["outcome"] == "GOAL"]
    return dict(goals=goals, sinks=sinks, top=top, n=n,
                steps=float(np.mean(hit_steps)) if hit_steps else None)


def _agg_learn(run):
    n = len(run["results"])
    goals = sum(r["outcome"] == "GOAL" for r in run["results"])
    sinks = sum(r["sink_end"] for r in run["results"])
    first = next((r["episode"] for r in run["results"]
                  if r["outcome"] == "GOAL"), None)
    last_block = max(5, n // 5)
    last_hits = sum(r["outcome"] == "GOAL" for r in run["results"][-last_block:])
    return dict(goals=goals, sinks=sinks, n=n, first=first,
                last_hits=last_hits, last_block=last_block,
                err=run["final_err"])


def _block_rate(results, start, end, key="GOAL"):
    """Inclusive 1-based episode window → hit rate."""
    chunk = results[start - 1:end]
    if not chunk:
        return float("nan")
    return sum(r["outcome"] == key for r in chunk) / len(chunk)


def _learn_a_blocks(n_episodes):
    """Fixed 60-episode windows, plus 20-episode windows beyond 60."""
    blocks = [b for b in LEARN_A_BLOCKS if b[0] <= n_episodes]
    lo = LEARN_A_BLOCKS[-1][1] + 1
    while lo <= n_episodes:
        blocks.append((lo, min(lo + 19, n_episodes)))
        lo += 20
    return tuple(blocks)


def _agg_learn_a(run):
    a = _agg_learn(run)
    res = run["results"]
    blocks = _learn_a_blocks(len(res))
    a["top"] = sum(r["top_row"] for r in res)
    a["blocks"] = {f"{lo}-{end}": _block_rate(res, lo, end)
                  for lo, end in blocks}
    a["sink_blocks"] = {f"{lo}-{end}": _block_rate(res, lo, end, "SINK")
                         for lo, end in blocks}
    return a


def compare_learn_a(mazes=("fog-walls",), **kwargs):
    """Multi-seed learn-A (flat A, B known). Analytics included when 4^H fits."""
    return compare_learn("learn-A", mazes=mazes, **kwargs)


def compare_learn(scenario, mazes=("fog-walls",), planners=None,
                   n_episodes=None, n_seeds=COMPARE_LEARN_A_SEEDS,
                   start_seed=0, n_iterations=DEFAULT_ITERS,
                   damping=DEFAULT_DAMPING, horizon=None,
                   preference=DEFAULT_PREFERENCE, prior_count=FLAT,
                   shrinking_horizon=True, max_steps=None, out=None):
    """Multi-seed learn-A (flat A, B known) or learn-B (A known, flat B)."""
    if scenario not in ("learn-A", "learn-B"):
        raise ValueError(f"scenario must be learn-A or learn-B, got {scenario}")
    err_lab = "errA" if scenario == "learn-A" else "errB"
    mazes = tuple(mazes)
    planners = tuple(planners or _default_compare_planners(mazes, horizon))
    n_episodes = n_episodes or DEFAULT_EPISODES[scenario]
    n_seeds = int(n_seeds)
    print(f"stochastic-maze compare {scenario}  mazes={','.join(mazes)}  "
          f"H={horizon if horizon is not None else 'layout'}"
          f"{'↓' if shrinking_horizon else ''}  "
          f"T={max_steps if max_steps is not None else 'default'}  "
          f"seeds={n_seeds}  eps={n_episodes}  prior={prior_count}")
    print(f"planners: {', '.join(planners)}")
    print()

    runs = {maze: {p: [] for p in planners} for maze in mazes}
    for maze in mazes:
        for planner in planners:
            for k in range(n_seeds):
                seed = start_seed + k
                print(f"  {maze}/{planner} {scenario} seed={seed} ...",
                      flush=True)
                run = run_scenario(
                    scenario, n_episodes=n_episodes, planner=planner,
                    maze=maze, seed=seed, quiet=True, collect_frames=False,
                    n_iterations=n_iterations, damping=damping,
                    horizon=horizon, preference=preference,
                    prior_count=prior_count, max_steps=max_steps,
                    shrinking_horizon=shrinking_horizon)
                runs[maze][planner].append(run)
                b = _agg_learn_a(run)
                first = "never" if b["first"] is None else str(b["first"])
                blocks = "  ".join(
                    f"{name}={rate:.2f}" for name, rate in b["blocks"].items())
                print(f"    goal={b['goals']}/{b['n']}  sink={b['sinks']}/{b['n']}  "
                      f"top={b['top']}/{b['n']}  first={first}  "
                      f"last{b['last_block']}={b['last_hits']}/{b['last_block']}  "
                      f"{err_lab}={b['err']:.4f}")
                print(f"    blocks: {blocks}", flush=True)
        print()

    blocks = _learn_a_blocks(n_episodes)
    for maze in mazes:
        print(f"{maze} maze, {scenario} ({n_episodes} eps × {n_seeds} seeds):")
        print(f"  {'planner':26s}  {'goal':8s}  {'sink':8s}  {'top':8s}  "
              f"{'last':10s}  {err_lab:6s} " +
              "  ".join(f"{lo}-{end:>2d}" for lo, end in blocks))
        for planner in planners:
            seed_runs = runs[maze][planner]
            aggs = [_agg_learn_a(r) for r in seed_runs]
            n = aggs[0]["n"]
            g = np.mean([a["goals"] for a in aggs])
            sk = np.mean([a["sinks"] for a in aggs])
            top = np.mean([a["top"] for a in aggs])
            last = np.mean([a["last_hits"] for a in aggs])
            err = np.mean([a["err"] for a in aggs])
            last_n = aggs[0]["last_block"]
            blk = [np.mean([a["blocks"][f"{lo}-{end}"] for a in aggs])
                    for lo, end in blocks]
            print(f"  {planner:26s}  {g:.1f}/{n:<4d}  {sk:.1f}/{n:<4d}  "
                  f"{top:.1f}/{n:<4d}  {last:.1f}/{last_n:<6d}  {err:.4f}  "
                  + "  ".join(f"{r:.2f}" for r in blk))
        print()

    if out:
        payload = dict(mazes=np.array(mazes), planners=np.array(planners),
                        n_episodes=n_episodes, n_seeds=n_seeds,
                        start_seed=start_seed, scenario=scenario)
        for maze in mazes:
            for planner in planners:
                for k, run in enumerate(runs[maze][planner]):
                    tag = f"{maze}|{planner}|s{start_seed + k}"
                    res = run["results"]
                    payload[f"{tag}|outcome"] = np.array(
                        [r["outcome"] for r in res])
                    payload[f"{tag}|steps"] = np.array(
                        [r["steps"] for r in res])
                    payload[f"{tag}|err"] = np.array(
                        [r["err"] for r in res])
                    payload[f"{tag}|top"] = np.array(
                        [r["top_row"] for r in res])
                    payload[f"{tag}|sink"] = np.array(
                        [r["sink_end"] for r in res])
        data_stem, fig_stem = split_stem(os.path.splitext(out)[0])
        npz_path = f"{data_stem}.npz"
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez(npz_path, **payload)
        print(f"saved {npz_path}")
        _plot_learn(runs, fig_stem, n_episodes, scenario)
    return runs


def _trailing_mean(x, win):
    """Trailing moving average with a growing window at the start (no edge
    artefacts, unlike ``np.convolve(..., mode="same")``)."""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - int(win), 0)
    return (c[idx] - c[lo]) / (idx - lo)


def _plot_learn_a(runs, out, n_episodes):
    """Backward-compatible alias."""
    _plot_learn(runs, out, n_episodes, scenario="learn-A")


def _plot_learn(runs, out, n_episodes, scenario="learn-A"):
    """Per-maze goal-rate and table error: mean ± 1 std across seeds.

    Goal rate is a trailing mean per seed first (so the band is seed
    variability, not Bernoulli noise), then mean±std of those curves.
    """
    from dispatch import PRETTY, STYLE

    win = max(1, n_episodes // 10)
    xs = np.arange(1, n_episodes + 1)
    err_lab = (r"mean $|\hat A - A|$ ± 1 std" if scenario == "learn-A"
               else r"mean $|\hat B - B|$ ± 1 std")
    for maze, per_planner in runs.items():
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        for planner, seed_runs in per_planner.items():
            st = dict(STYLE.get(planner, dict(color="black", ls="-")))
            st.pop("marker", None)
            hits = np.array(
                [[r["outcome"] == "GOAL" for r in run["results"]]
                 for run in seed_runs], dtype=float)
            err = np.array(
                [[r["err"] for r in run["results"]]
                 for run in seed_runs], dtype=float)
            smooth = np.stack([_trailing_mean(h, win) for h in hits])
            m, s = smooth.mean(axis=0), smooth.std(axis=0)
            em, es = err.mean(axis=0), err.std(axis=0)
            color = st.get("color", "black")
            axes[0].plot(xs, m, label=PRETTY.get(planner, planner), **st)
            axes[0].fill_between(
                xs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                color=color, alpha=0.18, lw=0)
            axes[1].plot(xs, em, **st)
            axes[1].fill_between(
                xs, em - es, em + es, color=color, alpha=0.18, lw=0)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel(f"goal rate ({win}-ep mean ± 1 std)")
        axes[0].set_title(f"{maze}: {scenario} return")
        axes[0].legend(fontsize=7, frameon=False)
        axes[1].set_xlabel("episode")
        axes[1].set_ylabel(err_lab)
        axes[1].set_title("table error")
        fig.tight_layout()
        stem = f"{out}_{maze}" if len(runs) > 1 else out
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        fig.savefig(f"{stem}.png")
        fig.savefig(f"{stem}.pdf")
        plt.close(fig)
        print(f"wrote {stem}.png / {stem}.pdf")


def compare_mazes(mazes=("fog-walls",), planners=None,
                  n_known_seeds=COMPARE_KNOWN_SEEDS,
                  n_learn_episodes=COMPARE_LEARN_B_EPISODES,
                  start_seed=0, n_iterations=DEFAULT_ITERS,
                  damping=DEFAULT_DAMPING, horizon=None,
                  preference=DEFAULT_PREFERENCE, prior_count=FLAT,
                  shrinking_horizon=True, max_steps=None):
    """Known (multi-seed) and learn-B (one seed, multi-ep) on each maze."""
    mazes = tuple(mazes)
    planners = tuple(planners or _default_compare_planners(mazes, horizon))
    print(f"stochastic-maze compare  mazes={','.join(mazes)}  "
          f"H={horizon if horizon is not None else 'layout'}"
          f"{'↓' if shrinking_horizon else ''}  "
          f"known seeds={n_known_seeds}  learn-B eps={n_learn_episodes}")
    print(f"planners: {', '.join(planners)}")
    print()

    known = {maze: {p: [] for p in planners} for maze in mazes}
    learn = {maze: {} for maze in mazes}
    for maze in mazes:
        for planner in planners:
            print(f"  {maze}/{planner} known ...", flush=True)
            for k in range(n_known_seeds):
                known[maze][planner].append(run_scenario(
                    "known", n_episodes=1, planner=planner, maze=maze,
                    seed=start_seed + k, quiet=True, collect_frames=False,
                    n_iterations=n_iterations, damping=damping,
                    horizon=horizon, preference=preference,
                    prior_count=prior_count, max_steps=max_steps,
                    shrinking_horizon=shrinking_horizon))
            a = _agg_known(known[maze][planner])
            steps = "—" if a["steps"] is None else f"{a['steps']:.1f}"
            print(f"    goal={a['goals']}/{a['n']}  sink={a['sinks']}/{a['n']}  "
                  f"top={a['top']}/{a['n']}  steps={steps}")

            print(f"  {maze}/{planner} learn-B ...", flush=True)
            learn[maze][planner] = run_scenario(
                "learn-B", n_episodes=n_learn_episodes, planner=planner,
                maze=maze, seed=start_seed, quiet=True, collect_frames=False,
                n_iterations=n_iterations, damping=damping,
                horizon=horizon, preference=preference,
                prior_count=prior_count, max_steps=max_steps,
                shrinking_horizon=shrinking_horizon)
            b = _agg_learn(learn[maze][planner])
            first = "never" if b["first"] is None else str(b["first"])
            print(f"    goal={b['goals']}/{b['n']}  sink={b['sinks']}/{b['n']}  "
                  f"first={first}  last{b['last_block']}="
                  f"{b['last_hits']}/{b['last_block']}  errB={b['err']:.4f}")
        print()

    for maze in mazes:
        print(f"{maze} maze, known ({n_known_seeds} seeds):")
        print(f"  {'planner':26s}  {'goal':8s}  {'sink':8s}  "
              f"{'top-row':8s}  steps")
        for planner in planners:
            a = _agg_known(known[maze][planner])
            steps = "—" if a["steps"] is None else f"{a['steps']:.1f}"
            print(f"  {planner:26s}  {a['goals']}/{a['n']:<5d}  "
                  f"{a['sinks']}/{a['n']:<5d}  {a['top']}/{a['n']:<5d}  {steps}")
        print()
        print(f"{maze} maze, learn-B ({n_learn_episodes} eps):")
        print(f"  {'planner':26s}  {'goal':8s}  {'sink':8s}  "
              f"{'first':6s}  {'last-block':10s}  errB")
        for planner in planners:
            b = _agg_learn(learn[maze][planner])
            first = "never" if b["first"] is None else str(b["first"])
            print(f"  {planner:26s}  {b['goals']}/{b['n']:<5d}  "
                  f"{b['sinks']}/{b['n']:<5d}  {first:6s}  "
                  f"{b['last_hits']}/{b['last_block']:<7d}  {b['err']:.4f}")
        print()
    return dict(known=known, learn=learn)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all",
                    choices=("known", "learn-A", "learn-B", "all"))
    ap.add_argument("--planner", default="SP-EFE", choices=PLANNERS,
                    help="MP planner (analytic EFE is 4^H; skipped at H=11)")
    ap.add_argument("--maze", default="fog-walls", choices=MAZE_LAYOUTS,
                    help="fog-walls, the paper's learn-A benchmark")
    ap.add_argument("--compare", action="store_true",
                    help="all planners that fit 4^H; known+learn-B, or "
                         "multi-seed learn-A / learn-B if --scenario is set; "
                         "no GIFs")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="seeds for --compare (default: 10 known, 3 learn-A)")
    ap.add_argument("--compare-planners", nargs="+", choices=PLANNERS,
                    default=None)
    ap.add_argument("--episodes", type=int, default=None,
                    help="default: 3 (known), 60 (learn-A), 200 "
                         "(--compare learn-B), 20 (learn-B GIF)")
    ap.add_argument("--steps", type=int, default=None,
                    help="episode length T (default: H under the shrinking "
                         f"horizon, {DEFAULT_FIXED_STEPS} with --fixed-horizon)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="planning depth (default: the layout horizon, 11)")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior-count", type=float, default=FLAT)
    ap.add_argument("--preference", type=float, default=DEFAULT_PREFERENCE)
    ap.add_argument("--per-step-goal", default="off",
                    choices=("on", "off"),
                    help="off: terminal-only x_T ~ goal (baseline); on: "
                         "preference in the channel-refresh locals every "
                         "step")
    ap.add_argument("--stop-at-goal", default="auto",
                    choices=("auto", "on", "off"),
                    help="end episodes on goal entry (auto: on except "
                         "learn-B, whose goal columns would stay unlearnable)")
    ap.add_argument("--action-select", default="auto",
                    choices=("auto", "argmax", "sample"),
                    help="action choice from q(u) (auto: argmax)")
    ap.add_argument("--fixed-horizon", action="store_true",
                    help="plan with a fixed H every step. The default shrinks "
                         "H to the time remaining")
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="GIF path (default figures/figures/"
                         "stochastic_maze_<maze>_<scenario>_<planner>.gif)")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    per_step = {"on": True, "off": False}[args.per_step_goal]
    planners = (tuple(args.compare_planners) if args.compare_planners
                else None)

    mazes = (args.maze,)

    if args.compare:
        if args.scenario in ("learn-A", "learn-B"):
            tag = "learnA" if args.scenario == "learn-A" else "learnB"
            if args.save:
                save = args.save
            else:
                save = (f"figures/data/stochastic_maze_"
                        f"{'+'.join(mazes)}_{tag}.npz")
            n_eps = args.episodes or (
                DEFAULT_EPISODES["learn-A"] if args.scenario == "learn-A"
                else 200)
            compare_learn(
                args.scenario, mazes=mazes, planners=planners,
                n_episodes=n_eps,
                n_seeds=(COMPARE_LEARN_A_SEEDS if args.n_seeds is None
                         else args.n_seeds),
                start_seed=args.seed, n_iterations=args.iters,
                damping=args.damping, horizon=args.horizon,
                preference=args.preference, prior_count=args.prior_count,
                shrinking_horizon=not args.fixed_horizon,
                max_steps=args.steps, out=save)
        else:
            compare_mazes(
                mazes=mazes, planners=planners,
                n_known_seeds=(COMPARE_KNOWN_SEEDS if args.n_seeds is None
                               else args.n_seeds),
                n_learn_episodes=args.episodes or COMPARE_LEARN_B_EPISODES,
                start_seed=args.seed, n_iterations=args.iters,
                damping=args.damping, horizon=args.horizon,
                preference=args.preference, prior_count=args.prior_count,
                shrinking_horizon=not args.fixed_horizon,
                max_steps=args.steps)
        return

    scenarios = (["known", "learn-A", "learn-B"]
                 if args.scenario == "all" else [args.scenario])

    for scenario in scenarios:
        result = run_scenario(
            scenario, n_episodes=args.episodes, max_steps=args.steps,
            horizon=args.horizon, n_iterations=args.iters,
            damping=args.damping, seed=args.seed,
            prior_count=args.prior_count, preference=args.preference,
            per_step_goal=per_step,
            stop_at_goal={"auto": None, "on": True,
                          "off": False}[args.stop_at_goal],
            action_select=(None if args.action_select == "auto"
                           else args.action_select),
            shrinking_horizon=not args.fixed_horizon,
            planner=args.planner, maze=args.maze)
        if not args.no_gif:
            path = args.save or (
                f"figures/figures/stochastic_maze_{args.maze}_{scenario}_"
                f"{args.planner}.gif")
            save_gif(result, path)
        print()


if __name__ == "__main__":
    main()
