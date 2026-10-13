"""Conference GIFs of the doors detour, one planner each.

SP-EFE stays on the safe route. SP-Pref walks that route while the six doors
are flat, then switches to the ice. Same hyperparameters as the paper run
(doors, learn-B, H=7, seed 0). Trajectories come from ``run_episodes``.

    uv run python figures/make_detour_gifs.py

Writes ``figures/figures/detour_sp_efe.gif`` and
``figures/figures/detour_sp_pref.gif``.
"""

from __future__ import annotations

import os
import sys

# This file lives in ``figures/``. Keep that directory importable so
# ``paper_figure`` resolves beside this file, and put the repo root first
# so ``dispatch`` and the experiment modules resolve there.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.append(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import paper_figure as pf  # noqa: E402
from dispatch import PRETTY  # noqa: E402
from environments.unmapped_detour import (  # noqa: E402
    ACTION_DXY,
    UnmappedDetourGrid,
)
from figures.paths import RENDER_DIR  # noqa: E402
from run_unmapped_detour import run_episodes  # noqa: E402

FIG_DIR = RENDER_DIR
N_EPISODES = 40
SEED = 0
# Milliseconds. The opening frame of an episode holds long enough to read
# the caption; the outcome holds long enough to see where the agent stopped.
MOVE_MS = 480
OPEN_MS = 900
OUTCOME_MS = 1700


def _clips(planner, results):
    """The episodes that carry each planner's story, with a one-line caption."""
    if planner == "SP-EFE":
        late = next(r for r in results if r["episode"] == 30)
        return [
            (results[0], "safe route"),
            (late, "still the safe route"),
        ]
    first_ice = next(r for r in results if r["via_gate"])
    other = next((r for r in results
                  if r["via_gate"] and r["outcome"] != first_ice["outcome"]),
                 None)

    clips = [(results[0], "safe route — doors still flat"),
             (first_ice, "ice,  p = 1/2")]
    if other is not None:
        clips.append((other, "ice,  p = 1/2"))
    return clips


def _draw_board(ax, grid):
    """The doors grid in the talk-figure colours, without the route arrows."""
    gx, gy = grid.grid_x, grid.grid_y
    walls = {grid.state_to_xy(s) for s in grid.wall_idx}
    doors = {grid.state_to_xy(s) for s in grid.unmapped_idx}
    goal, gate, dead = grid.goal_xy, grid.gate_xy, grid.deadend_xy
    for y in range(gy):
        for x in range(gx):
            cell = (x, y)
            face = (pf.MAZE["wall"] if cell in walls
                    else pf.MAZE["goal"] if cell == goal
                    else pf.MAZE["ice"] if cell == gate
                    else pf.MAZE["deadend"] if cell == dead
                    else pf.MAZE["door_face"] if cell in doors
                    else pf.MAZE["free"])
            pf._cell(ax, x, y, face)
    for (x, y) in doors:
        ax.add_patch(patches.Rectangle(
            (x + 0.08, y + 0.08), 0.84, 0.84, fill=False,
            edgecolor=pf.MAZE["door"], lw=0.7 * pf.POINT,
            ls=(0, (1.4, 1.1)), zorder=3))
    grid._draw_ice_crystal(ax, gate[0] + 0.5, gate[1] + 0.5)
    pf._frame(ax, gx, gy)


def _frame_image(grid, planner, rec, caption, index):
    """One GIF frame: the agent at ``states[index]``, trail behind it."""
    color = pf.STYLE[planner]["color"]
    states = [grid.state_to_xy(s) for s in rec["states"]]
    path = rec["path"]
    at_end = index == len(states) - 1
    x, y = states[index]

    fig = plt.figure(figsize=(7.2, 7.4), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.74])
    _draw_board(ax, grid)

    pts = [(c[0] + 0.5, c[1] + 0.5) for c in states[:index + 1]]
    if len(pts) >= 2:
        halo = [pe.withStroke(linewidth=5.2 * pf.POINT, foreground="white")]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=color, lw=2.4 * pf.POINT, solid_capstyle="round",
                solid_joinstyle="round", zorder=6, path_effects=halo)
    ax.add_patch(patches.Circle(
        (x + 0.5, y + 0.5), 0.16, facecolor=color, edgecolor="white",
        lw=1.6 * pf.POINT, zorder=8))
    # Corner marks stay visible once the agent is standing on the cell.
    ax.text(grid.start_xy[0] + 0.12, grid.start_xy[1] + 0.12, "S",
            ha="left", va="bottom", fontsize=8 * pf.POINT, color=pf.MAZE["ink"],
            zorder=10, path_effects=[pe.withStroke(linewidth=2.0 * pf.POINT,
                                                   foreground="white")])
    ax.text(grid.goal_xy[0] + 0.88, grid.goal_xy[1] + 0.88, "G",
            ha="right", va="top", fontsize=8 * pf.POINT, color=pf.MAZE["ink"],
            zorder=10, path_effects=[pe.withStroke(linewidth=2.0 * pf.POINT,
                                                   foreground="white")])

    if not at_end:
        dx, dy = ACTION_DXY[grid.true_action(path[index])]
        ax.annotate(
            "", xy=(x + 0.5 + 0.34 * dx, y + 0.5 + 0.34 * dy),
            xytext=(x + 0.5 + 0.18 * dx, y + 0.5 + 0.18 * dy),
            arrowprops=dict(arrowstyle="-|>", color=pf.MAZE["ink"],
                            lw=1.3 * pf.POINT, mutation_scale=14,
                            shrinkA=0, shrinkB=0),
            zorder=9)

    outcome = ""
    outcome_color = pf.MAZE["ink"]
    if at_end:
        if rec["outcome"] == "GOAL":
            outcome, outcome_color = "goal", "#1B7A3A"
        elif rec["outcome"] == "DEADEND":
            outcome, outcome_color = "dead end", "#A33B24"
        else:
            outcome = rec["outcome"].lower()
    fig.text(0.06, 0.93, PRETTY[planner], fontsize=22, color=color,
             ha="left", va="center")
    fig.text(0.06, 0.875, f"episode {rec['episode']}   ·   {caption}",
             fontsize=15, color="0.15", ha="left", va="center")
    if outcome:
        fig.text(0.50, 0.045, outcome, fontsize=18, color=outcome_color,
                 ha="center", va="center")

    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def _save_gif(grid, planner, results, path):
    frames, durations = [], []
    for rec, caption in _clips(planner, results):
        n = len(rec["states"])
        for index in range(n):
            frames.append(_frame_image(grid, planner, rec, caption, index))
            if index == 0:
                durations.append(OPEN_MS)
            elif index == n - 1:
                durations.append(OUTCOME_MS)
            else:
                durations.append(MOVE_MS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # One palette for every frame, so the loop does not flicker.
    palette = frames[0].quantize(colors=128)
    quantized = [palette]
    quantized.extend(im.quantize(palette=palette, dither=Image.Dither.NONE)
                     for im in frames[1:])
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                      duration=durations, loop=0, disposal=2)
    print(f"wrote {path}  ({len(frames)} frames)")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    grid = UnmappedDetourGrid(unmapped_mode="doors")
    # Match the slide figure's stroke weight on a larger canvas.
    pf.POINT = 1.7
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    for planner in ("SP-EFE", "SP-Pref"):
        print(f"running {planner} ...", flush=True)
        run = run_episodes(planner, grid=grid, scenario="learn-B",
                           n_episodes=N_EPISODES, seed=SEED)
        for rec, caption in _clips(planner, run["results"]):
            print(f"  ep {rec['episode']:3d}  {rec['outcome']:8s}  "
                  f"detour={rec['via_detour']}  {caption}")
        _save_gif(grid, planner, run["results"],
                  os.path.join(FIG_DIR, f"detour_{planner.lower().replace('-', '_')}.gif"))


if __name__ == "__main__":
    main()
