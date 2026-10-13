"""Episode rendering for the fog-walls maze: belief overlays and GIFs.

Not used by the paper -- ``figures/paper_figure.py`` draws its own environment panels so
both share one visual vocabulary. This is here for inspecting a single episode:
``run_fog_walls.py --save-gif`` writes a frame per step showing the agent, the
belief q(x), the action posterior q(u), and the matrix being learned.

Frames come from ``run_scenario(collect_frames=True)``. Cell (x, y) is drawn at
plot coordinates (x+.5, y+.5), with y pointing up.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from environments.stochastic_maze import (
    ACTIONS, ACTION_SYMBOLS, ACTION_DXY, N_ACTIONS, depict, state_to_xy,
)


def draw_maze(ax, f, layout=None):
    kw = {}
    if layout:
        kw = dict(
            drift_spec=layout["drift_spec"],
            sink_states=layout["sink_states"],
            noisy_observations=layout["noisy_observations"],
            fog_cells=layout.get("fog_cells") or (),
            grid_x=layout["grid_x"], grid_y=layout["grid_y"],
            goal_xy=layout["goal_xy"],
            wall_states=layout.get("wall_states") or (),
        )
    gx = int(layout["grid_x"]) if layout else int(np.sqrt(len(f["q"])))
    n_states = len(f["q"])
    depict(ax=ax, **kw)

    for s in range(n_states):
        p = float(f["q"][s])
        if p < 0.02:
            continue
        x, y = state_to_xy(s, grid_x=gx)
        ax.add_patch(patches.Circle(
            (x + 0.5, y + 0.5), 0.30 * max(p, 0.18),
            facecolor="mediumseagreen", edgecolor="darkgreen", alpha=0.45,
            zorder=4))

    x, y = state_to_xy(f["state"], grid_x=gx)
    ax.add_patch(patches.Circle((x + 0.5, y + 0.5), 0.18, color="crimson",
                                zorder=5))
    if f.get("action") is not None:
        dx, dy = ACTION_DXY[f["action"]]
        ax.annotate("", xy=(x + 0.5 + 0.38 * dx, y + 0.5 + 0.38 * dy),
                    xytext=(x + 0.5, y + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
                    zorder=6)

    if f.get("arrived"):
        color = {"GOAL": "darkgoldenrod", "SINK": "firebrick",
                 "timeout": "gray"}[f["outcome"]]
        ax.set_title(f"ep{f['episode']} — {f['outcome']}", fontsize=10,
                     color=color, fontweight="bold")
    else:
        ax.set_title(f"ep{f['episode']} step {f['step']}  "
                     f"a={ACTIONS[f['action']]}", fontsize=10)


def draw_actions(ax, f):
    ax.clear()
    ad = f["action_dist"]
    colors = ["lightsteelblue"] * N_ACTIONS
    if f.get("action") is not None:
        colors[f["action"]] = "steelblue"
    ax.bar(range(N_ACTIONS), ad, color=colors, edgecolor="gray")
    ax.set_xticks(range(N_ACTIONS))
    ax.set_xticklabels([f"{s}\n{a[0]}" for a, s in
                        zip(ACTIONS, ACTION_SYMBOLS)])
    ax.set_ylim(0, 1.05)
    ax.set_title("action posterior q(u)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def draw_matrix(ax, f, scenario, layout=None):
    ax.clear()
    n_states = int(f["q"].shape[0])
    gx = int(layout["grid_x"]) if layout else int(np.sqrt(n_states))
    gy = int(layout["grid_y"]) if layout else n_states // gx
    if scenario == "learn-A":
        ax.imshow(f["A"][0], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("state", fontsize=8)
        ax.set_ylabel("observation", fontsize=8)
        ax.set_title(f"learned likelihood A   mean|A−A*| = {f['err']:.3f}",
                     fontsize=10)
        ax.tick_params(labelsize=6)
    elif scenario == "learn-B":
        B = f["B"]
        gap = np.full((n_states, 2), np.nan)
        blocks = []
        for u in range(N_ACTIONS):
            blocks.append(B[:, :, u])
            if u < N_ACTIONS - 1:
                blocks.append(gap)
        M = np.concatenate(blocks, axis=1)
        ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        centers = [u * (n_states + 2) + (n_states - 1) / 2
                   for u in range(N_ACTIONS)]
        ax.set_xticks(centers)
        ax.set_xticklabels(ACTION_SYMBOLS, fontsize=12)
        ax.set_xlabel("current state, grouped by action", fontsize=8)
        ax.set_ylabel("next state", fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"learned transitions B   mean|B−B*| = {f['err']:.3f}",
                     fontsize=10)
    else:
        M = f["q"].reshape(gy, gx)
        ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto",
                  origin="lower")
        ax.set_xticks(range(gx))
        ax.set_yticks(range(gy))
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_title("state posterior q(x, y)", fontsize=10)


def save_gif(result, path, interval_ms=None, hold_frames=3):
    scenario = result["scenario"]
    if interval_ms is None:  # learning runs have many frames — pace faster
        interval_ms = 700 if scenario == "known" else 300
    frames = []
    for f in result["frames"]:
        frames.append(f)
        if f.get("arrived"):
            frames.extend([f] * (hold_frames - 1))
    fig = plt.figure(figsize=(9, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1],
                          width_ratios=[1.15, 1])
    maze_ax = fig.add_subplot(gs[0, 0])
    act_ax = fig.add_subplot(gs[0, 1])
    mat_ax = fig.add_subplot(gs[1, :])
    fig.suptitle(f"Stochastic maze ({result.get('maze', 'fog-walls')}) — "
                 f"{result.get('planner', 'SP-EFE')}"
                 f" — {scenario}  (H={result['horizon']}"
                 f"{'↓' if result.get('shrinking_horizon') else ''})",
                 fontweight="bold")
    fig.subplots_adjust(hspace=0.45, top=0.9)

    def show(i):
        f = frames[i]
        maze_ax.clear()
        draw_maze(maze_ax, f, layout=result.get("layout"))
        draw_actions(act_ax, f)
        draw_matrix(mat_ax, f, scenario, layout=result.get("layout"))

    anim = FuncAnimation(fig, show, frames=len(frames),
                         interval=interval_ms, repeat=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    anim.save(path, writer="pillow", fps=max(1, round(1000 / interval_ms)))
    print(f"animation saved to {path}")
    plt.close(fig)
