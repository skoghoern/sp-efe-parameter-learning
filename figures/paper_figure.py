"""Figures for the paper's experiments, drawn from the saved ``.npz`` arrays.

Re-renders fog-walls (learn θ_A) and the unmapped detour (learn θ_B) as one
2×3 page: environment, goal rate, and a second metric. No experiment is
re-run. The page is saved at a fixed size, without a tight bounding box, so
including it does not rescale the fonts.

    uv run python figures/paper_figure.py

Writes ``figures/figures/exp_learning.pdf``.

    uv run python figures/paper_figure.py --slides

Also writes slide PNGs for both scenarios: the grid with its glyph legend,
and the curve panels with the planner legend. Fog-walls panels are
(a) and (b)–(c); the detour panels are (d), (e)–(g), and (e)+(g).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# --- locations -------------------------------------------------------------
# This file lives in ``figures/``. ``REPO`` is the checkout that holds
# ``figures/data/*.npz``; ``SP_EFE_FIG_OUT`` redirects the PDF into a paper tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MP_AIF_REPO", os.path.dirname(_HERE))
DATA_DIR = os.path.join(REPO, "figures", "data")
FIG_OUT = os.environ.get("SP_EFE_FIG_OUT", os.path.join(REPO, "figures", "figures"))
LEARN_A_NPZ = os.path.join(DATA_DIR, "stochastic_maze_fog_walls_learnA_H8.npz")
LEARN_A_ABL_NPZ = os.path.join(
    DATA_DIR, "stochastic_maze_fog_walls_learnA_H8_ablation.npz")
LEARN_B_NPZ = os.path.join(DATA_DIR, "unmapped_detour_p0.50_doors_learn-B_H7.npz")

sys.path.insert(0, REPO)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dispatch import PRETTY as PAPER_NAME  # noqa: E402

# --- paper styling ----------------------------------------------------------
# Keys are the paper labels, which are also the ``.npz`` keys. ``PAPER_NAME``
# is the same label typeset with superscripts.
# Planner colours are reserved for curves only; the environment panels use
# purple / grey routes so no hue means two things in one figure.
_EFE = "#D55E00"
_NODYN = "#0072B2"
_PREF_PLAN = "#009E73"
_PREF = "#56B4E9"
STYLE = {
    "SP-EFE": dict(color=_EFE, ls="-", lw=1.5, zorder=6),
    "SP-EFE-dyn": dict(color=_NODYN, ls="-", lw=1.1, zorder=5),
    "SP-EFE-DeltaB": dict(color="#CC79A7", ls="-", lw=1.1, zorder=4),
    "SP-EFE-DeltaA-DeltaB": dict(color="#8B5A00", ls=(0, (3.0, 1.1, 0.8, 1.1)),
                           lw=1.2, zorder=4),
    "SP-EFE-DeltaA-DeltaB-dyn": dict(color="#8B5A00", ls=(0, (3.0, 1.1, 0.8, 1.1)),
                              lw=1.2, zorder=4),
    "SP-Pref-Plan": dict(color=_PREF_PLAN, ls="--", lw=1.3, zorder=5),
    "SP-Pref": dict(color=_PREF, ls="-.", lw=1.1, zorder=4),
    "Random": dict(color="0.45", ls=":", lw=1.0, zorder=3),
}
# Shared cell/route vocabulary for both environment panels.
MAZE = {
    "ink": "#1A1A1A",
    "edge": "#1A1A1A",
    "free": "white",
    "wall": "0.38",
    "fog": "0.82",
    "hatch": "0.55",
    "drift": "#C8A165",
    "noisy": "0.92",
    "goal": "#BBE8B4",
    "ice": "#CFE6F5",
    "ice_mark": "#3e6f8a",  # six-point crystal, same ink as the detour GIF
    "deadend": "#F2C9C0",
    "door": "#2C6FAF",      # dotted outline of a cell with unknown B columns
    "door_face": "#EDF4FB",
    "safe": "#6A3D9A",   # informative / deterministic route
    "risky": "0.42",     # short but ambiguous / stochastic route
}
RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": None,   # keep the exact figsize: no rescaling in LaTeX
}
TEXTWIDTH_IN = 12.2 / 2.54  # LNCS text width, 4.803 in
# Fraction of a 12.2 cm text width. Match this if the figure is included.
# the figure is saved at its final printed size so LaTeX never rescales it.
WIDTH_FRAC = 0.92
FIG_WIDTH_IN = WIDTH_FRAC * TEXTWIDTH_IN
FIG_HEIGHT_IN = 3.20
# Point sizes below are the journal figure. Slide PNGs set this above 1 so
# letters, strokes, and line widths stay in proportion on a larger page.
POINT = 1.0


# --- data helpers -----------------------------------------------------------
def load_npz(path):
    """Load a results archive, failing loudly with the regenerating hint."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing results archive {path!r}; regenerate it in the code repo "
            f"(see AGENTS.md) or point MP_AIF_REPO at the right checkout")
    return np.load(path, allow_pickle=True)


def trailing_mean(x, win):
    """Trailing moving average with a growing window at the start."""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - int(win), 0)
    return (c[idx] - c[lo]) / (idx - lo)


def band(ax, xs, per_seed, win, style, label, hi=None):
    """Seed-mean of the trailing means with a +-1 std ribbon, clipped to [0, hi]."""
    smooth = np.stack([trailing_mean(h, win) for h in per_seed])
    m, s = smooth.mean(axis=0), smooth.std(axis=0)
    ax.plot(xs, m, label=label, **style)
    ax.fill_between(xs, np.clip(m - s, 0.0, None),
                    np.clip(m + s, 0.0, hi if hi is not None else np.inf),
                    color=style["color"], alpha=0.16, lw=0)


# --- environment panels: one shared vocabulary ------------------------------
def _cell(ax, x, y, face, hatch=None):
    """Draw one cell: face (optionally hatched) plus a crisp border on top."""
    if hatch:
        ax.add_patch(patches.Rectangle(
            (x, y), 1, 1, facecolor=face, edgecolor=MAZE["hatch"],
            hatch=hatch, lw=0, zorder=1))
    else:
        ax.add_patch(patches.Rectangle(
            (x, y), 1, 1, facecolor=face, lw=0, zorder=1))
    ax.add_patch(patches.Rectangle(
        (x, y), 1, 1, fill=False, edgecolor=MAZE["edge"], lw=0.5 * POINT,
        zorder=2))


def _letter(ax, x, y, text):
    """Cell label with a white halo so routes never make it unreadable."""
    ax.text(x + 0.5, y + 0.5, text, ha="center", va="center",
            fontsize=6.5 * POINT, color=MAZE["ink"], zorder=7,
            path_effects=[pe.withStroke(linewidth=1.5 * POINT,
                                        foreground="white")])


def _route(ax, cells, color, ls, lw, tail=0.28, head=0.30):
    """One polyline through the cell centres with a single arrowhead."""
    pts = [(x + 0.5, y + 0.5) for x, y in cells]
    if len(pts) < 2:
        return

    def _unit(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    ux0, uy0 = _unit(pts[0], pts[1])
    ux1, uy1 = _unit(pts[-2], pts[-1])
    pts[0] = (pts[0][0] + tail * ux0, pts[0][1] + tail * uy0)
    stop = (pts[-1][0] - head * ux1, pts[-1][1] - head * uy1)
    xs = [p[0] for p in pts[:-1]] + [stop[0]]
    ys = [p[1] for p in pts[:-1]] + [stop[1]]
    halo = [pe.withStroke(linewidth=lw + 1.3 * POINT, foreground="white")]
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, solid_capstyle="round",
            dash_capstyle="round", zorder=4, path_effects=halo)
    ax.annotate("", xy=(stop[0] + 0.14 * ux1, stop[1] + 0.14 * uy1),
                xytext=stop,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=5 * POINT,
                                path_effects=halo),
                zorder=5)


def _frame(ax, gx, gy):
    """Square cells, no axes decoration, a hair of padding for the border."""
    ax.set_xlim(-0.06, gx + 0.06)
    ax.set_ylim(-0.06, gy + 0.06)
    ax.set_aspect("equal")
    ax.axis("off")


def _snowflake_segments(cx, cy, radius):
    """Three diameters at 60°, the six-point ice mark from the detour GIF."""
    segments = []
    for deg in (0.0, 60.0, 120.0):
        angle = math.radians(deg)
        dx, dy = radius * math.cos(angle), radius * math.sin(angle)
        segments.append(((cx - dx, cx + dx), (cy - dy, cy + dy)))
    return segments


def _draw_snowflake(ax, cx, cy, radius=0.20):
    """Ice crystal at a cell centre, the same six spokes as the detour GIF.

    A thin white edge keeps the spokes off the dashed route that crosses
    the cell.
    """
    lw = 0.9 * POINT
    for xs, ys in _snowflake_segments(cx, cy, radius):
        ax.plot(xs, ys, color=MAZE["ice_mark"], lw=lw, solid_capstyle="round",
                zorder=6, clip_on=True,
                path_effects=[pe.withStroke(linewidth=lw + 0.55 * POINT,
                                             foreground="white")])


def draw_fog_walls(ax):
    """Panel (a): fog-walls, with the direct and observation-exact routes."""
    from environments.stochastic_maze import maze_tensors

    _, _, layout = maze_tensors("fog-walls")
    gx, gy = int(layout["grid_x"]), int(layout["grid_y"])
    fog = set(layout.get("fog_cells") or ())
    wall = set(layout.get("wall_states") or ())
    noisy = set(layout.get("noisy_observations") or ())
    drift = dict(layout.get("drift_spec") or {})
    start, goal = tuple(layout["start_xy"]), tuple(layout["goal_xy"])

    for y in range(gy):
        for x in range(gx):
            cell = (x, y)
            if cell in wall:
                _cell(ax, x, y, MAZE["wall"])
            elif cell == goal:
                _cell(ax, x, y, MAZE["goal"])
            elif cell in fog:
                _cell(ax, x, y, MAZE["fog"], hatch="////")
            elif cell in noisy:
                _cell(ax, x, y, MAZE["noisy"])
            else:
                _cell(ax, x, y, MAZE["free"])
    # Drift bands sit on top of the (fog) face of their own cell; the offsets
    # straddle the cell centre so the direct route never hides all of them.
    for (x, y) in drift:
        for frac in (0.18, 0.37, 0.63, 0.82):
            ax.add_patch(patches.Rectangle(
                (x, y + frac - 0.055), 1, 0.11, facecolor=MAZE["drift"],
                edgecolor="none", zorder=2))

    direct = [(x, start[1]) for x in range(gx)]  # 4 steps through the fog
    _route(ax, direct, MAZE["risky"], (0, (2.6, 1.4)), 1.1 * POINT)
    _route(ax, [tuple(c) for c in layout["top_row_cells"]],
           MAZE["safe"], "-", 1.2 * POINT)
    _letter(ax, *start, "S")
    _letter(ax, *goal, "G")
    _frame(ax, gx, gy)


def draw_detour(ax):
    """Panel (d): unmapped detour (doors), risky ice vs six unknown doors."""
    from environments.unmapped_detour import UnmappedDetourGrid

    grid = UnmappedDetourGrid(unmapped_mode="doors")
    gx, gy = grid.grid_x, grid.grid_y
    walls = {grid.state_to_xy(s) for s in grid.wall_idx}
    doors = {grid.state_to_xy(s) for s in grid.unmapped_idx}
    start, goal = tuple(grid.start_xy), tuple(grid.goal_xy)
    gate, dead = tuple(grid.gate_xy), tuple(grid.deadend_xy)

    for y in range(gy):
        for x in range(gx):
            cell = (x, y)
            face = (MAZE["wall"] if cell in walls
                    else MAZE["goal"] if cell == goal
                    else MAZE["ice"] if cell == gate
                    else MAZE["deadend"] if cell == dead
                    else MAZE["door_face"] if cell in doors
                    else MAZE["free"])
            _cell(ax, x, y, face)
    for (x, y) in doors:  # cells whose outgoing B columns start flat
        ax.add_patch(patches.Rectangle(
            (x + 0.08, y + 0.08), 0.84, 0.84, fill=False,
            edgecolor=MAZE["door"], lw=0.6 * POINT, ls=(0, (1.4, 1.1)),
            zorder=3))

    detour = [grid.state_to_xy(s) for s in grid._path_states(grid.detour_path)]
    shortcut = [grid.state_to_xy(s) for s in grid._path_states(grid.shortcut_path)]
    _route(ax, shortcut, MAZE["risky"], (0, (2.6, 1.4)), 1.1 * POINT)
    _route(ax, [gate, dead], MAZE["risky"], (0, (0.9, 1.1)), 1.0 * POINT,
           tail=0.30)
    _route(ax, detour, MAZE["safe"], "-", 1.2 * POINT)
    _draw_snowflake(ax, gate[0] + 0.5, gate[1] + 0.5)
    for cell, text in ((start, "S"), (goal, "G")):
        _letter(ax, *cell, text)
    # The gate probabilities live in the glyph legend: at this cell size an
    # in-grid "p = 0.5" is half a cell tall and crosses the wall block.
    _frame(ax, gx, gy)


# --- Row 1: fog-walls learn-A ----------------------------------------------
def draw_learn_a_curves(goal_ax, err_ax,
                        planners=("SP-EFE", "SP-EFE-dyn", "SP-Pref-Plan",
                                  "SP-Pref"),
                        ablations=("SP-EFE-DeltaA-DeltaB", "Random")):
    """Goal-rate and matrix-error panels for fog-walls learn-A."""
    def _curves(d, keys):
        maze = str(d["mazes"][0])
        n_ep, n_seeds = int(d["n_episodes"]), int(d["n_seeds"])
        s0 = int(d["start_seed"])
        xs = np.arange(1, n_ep + 1)
        win = max(1, n_ep // 10)
        for p in keys:
            hits = np.array(
                [[o == "GOAL" for o in d[f"{maze}|{p}|s{s0 + k}|outcome"]]
                 for k in range(n_seeds)], dtype=float)
            err = np.array([d[f"{maze}|{p}|s{s0 + k}|err"]
                            for k in range(n_seeds)])
            band(goal_ax, xs, hits, win, STYLE[p], PAPER_NAME[p], hi=1.0)
            band(err_ax, xs, err, 1, STYLE[p], PAPER_NAME[p])
        return n_ep, win

    n_ep, win = _curves(load_npz(LEARN_A_NPZ), planners)
    # The novelty ablation lives in a separate archive; it is the flat-zero
    # line that makes "novelty is the drive" visible rather than table-only.
    try:
        _curves(load_npz(LEARN_A_ABL_NPZ), ablations)
    except (FileNotFoundError, KeyError) as exc:
        print(f"fog-walls ablation curves omitted: {exc}", file=sys.stderr)

    goal_ax.set_ylim(-0.03, 1.03)
    goal_ax.set_ylabel(f"goal rate ({win}-ep. mean)")
    goal_ax.set_title("(b) goal rate", loc="left", pad=2.5 * POINT)
    err_ax.set_ylim(0.0, None)
    err_ax.set_ylabel(r"mean $|\hat{A}-A|$")
    err_ax.set_title("(c) matrix error", loc="left", pad=2.5 * POINT)
    for ax in (goal_ax, err_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, n_ep)
        ax.set_xticks([0, n_ep // 2, n_ep])
        ax.set_xlabel("episode", labelpad=1.0 * POINT)


def draw_learn_a(axes, planners=("SP-EFE", "SP-EFE-dyn", "SP-Pref-Plan",
                                 "SP-Pref"),
                 ablations=("SP-EFE-DeltaA-DeltaB", "Random")):
    """Fog-walls learn-A row: layout, goal rate, A-matrix error."""
    draw_fog_walls(axes[0])
    axes[0].set_title(r"(a) fog-walls: learn $\theta_A$", loc="left",
                      pad=2.5 * POINT)
    draw_learn_a_curves(axes[1], axes[2], planners, ablations)


# --- Row 2: unmapped detour (doors) learn-B --------------------------------
DETOUR_PLANNERS = ("SP-EFE", "SP-EFE-dyn", "SP-Pref-Plan", "SP-Pref", "Random")


def draw_learn_b_curves(goal_ax, route_ax=None, planners=DETOUR_PLANNERS,
                        err_ax=None):
    """Goal-rate panel, with optional detour-rate and B-error panels."""
    d = load_npz(LEARN_B_NPZ)
    n_ep, gate_p = int(d["n_episodes"]), float(d["gate_p"])
    xs = np.arange(1, n_ep + 1)
    win = max(1, n_ep // 10)

    for p in planners:
        band(goal_ax, xs, d[f"{p}|hit"], win, STYLE[p], PAPER_NAME[p], hi=1.0)
        if route_ax is not None:
            band(route_ax, xs, d[f"{p}|detour"].astype(float), win, STYLE[p],
                 PAPER_NAME[p], hi=1.0)
        if err_ax is not None:
            # Already a per-episode mean, so no trailing window (as in panel c).
            # The archive records this on the unknown columns only.
            band(err_ax, xs, d[f"{p}|err"], 1, STYLE[p], PAPER_NAME[p])
    goal_ax.axhline(gate_p, color="0.3", lw=0.7 * POINT, ls=(0, (4, 2)),
                    zorder=2)
    goal_ax.text(n_ep * 0.985, gate_p + 0.04, r"ice ceiling $p$", ha="right",
                 va="bottom", fontsize=6.0 * POINT, color="0.3")
    goal_ax.set_ylim(-0.03, 1.03)
    goal_ax.set_ylabel(f"goal rate ({win}-ep. mean)")
    goal_ax.set_title("(e) goal rate", loc="left", pad=2.5 * POINT)
    axes = [goal_ax]
    if route_ax is not None:
        route_ax.set_ylim(-0.03, 1.03)
        route_ax.set_ylabel(f"detour rate ({win}-ep. mean)")
        route_ax.set_title("(f) route taken", loc="left", pad=2.5 * POINT)
        axes.append(route_ax)
    if err_ax is not None:
        err_ax.set_ylim(0.0, None)
        err_ax.set_ylabel(r"mean $|\hat{B}-B|$")
        err_ax.set_title("(g) matrix error", loc="left", pad=2.5 * POINT)
        axes.append(err_ax)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, n_ep)
        ax.set_xticks([0, n_ep // 3, 2 * (n_ep // 3), n_ep])
        ax.set_xlabel("episode", labelpad=1.0 * POINT)
    return gate_p


def draw_learn_b(axes, planners=DETOUR_PLANNERS):
    """Unmapped-detour (doors) learn-B row: layout, goal rate, detour rate."""
    draw_detour(axes[0])
    axes[0].set_title(r"(d) detour: learn $\theta_B$", loc="left",
                      pad=2.5 * POINT)
    return draw_learn_b_curves(axes[1], axes[2], planners)


# --- legends ----------------------------------------------------------------
def planner_handles(keys=("SP-EFE", "SP-EFE-dyn", "SP-EFE-DeltaA-DeltaB",
                          "SP-Pref-Plan", "SP-Pref", "Random")):
    """Curve legend. The default order follows the paper's table."""
    return [Line2D([], [], label=PAPER_NAME[k], **STYLE[k]) for k in keys]


def glyph_handles(gate_p=0.5):
    """Cell/route vocabulary of panels (a) and (d)."""
    def patch(face, label, hatch=None, edge=MAZE["edge"], ls="-", lw=0.5):
        return patches.Patch(facecolor=face, edgecolor=edge, hatch=hatch,
                             linewidth=lw, linestyle=ls, label=label)

    return [
        patch(MAZE["fog"], "fog: uniform $A$ (a)", hatch="////"),
        patch(MAZE["fog"], "drift cell (a)", hatch="---",
              edge=MAZE["drift"], lw=0.8),
        patch(MAZE["noisy"], "noisy $A$ (a)"),
        patch(MAZE["door_face"], "unknown $B$ columns (d)", edge=MAZE["door"],
              ls=(0, (1.4, 1.1)), lw=0.7),
        patch(MAZE["ice"], rf"ice: G w.p. $p{{=}}{gate_p:g}$ (d)"),
        patch(MAZE["deadend"], r"dead end w.p. $1{-}p$ (d)"),
        patch(MAZE["wall"], "wall"),
        patch(MAZE["goal"], "goal G"),
        Line2D([], [], color=MAZE["safe"], ls="-", lw=1.2,
               label="safe/informative route"),
        Line2D([], [], color=MAZE["risky"], ls=(0, (2.6, 1.4)), lw=1.1,
               label="short risky route"),
    ]


def fog_glyph_handles():
    """Cell and route key for the fog-walls grid. No detour glyphs."""
    def patch(face, label, hatch=None, edge=MAZE["edge"], ls="-", lw=0.5):
        return patches.Patch(facecolor=face, edgecolor=edge, hatch=hatch,
                             linewidth=lw * POINT, linestyle=ls, label=label)

    return [
        patch(MAZE["fog"], r"fog: uniform $A$", hatch="////"),
        patch(MAZE["fog"], "drift cell", hatch="---",
              edge=MAZE["drift"], lw=0.8),
        patch(MAZE["noisy"], r"noisy $A$"),
        patch(MAZE["wall"], "wall"),
        patch(MAZE["goal"], "goal G"),
        Line2D([], [], color=MAZE["safe"], ls="-", lw=1.2 * POINT,
               label="safe/informative route"),
        Line2D([], [], color=MAZE["risky"], ls=(0, (2.6, 1.4)),
               lw=1.1 * POINT, label="short risky route"),
    ]


class _IceSwatch:
    """Legend proxy: ice fill plus the six-point crystal."""

    def __init__(self, label="ice"):
        self._label = label

    def get_label(self):
        return self._label


class _IceSwatchHandler(HandlerBase):
    """Draw the ice swatch the way the grid draws the cell."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        face = patches.Rectangle(
            (-xdescent, -ydescent), width, height,
            facecolor=MAZE["ice"], edgecolor=MAZE["edge"],
            linewidth=0.5 * POINT, transform=trans)
        artists = [face]
        cx = -xdescent + width / 2
        cy = -ydescent + height / 2
        radius = 0.36 * min(width, height)
        for xs, ys in _snowflake_segments(cx, cy, radius):
            artists.append(Line2D(
                xs, ys, color=MAZE["ice_mark"], lw=0.55 * POINT,
                solid_capstyle="round", transform=trans))
        return artists


def detour_glyph_handles(_gate_p=0.5):
    """Cell and route key for the detour grid, in left-to-right reading order.

    Two columns: unknown column | wall, goal | dead end, ice alone, then the
    two routes. The blank entry keeps ice on its own row. The probability is
    not repeated here; the crystal marks the ice.
    """

    def patch(face, label, hatch=None, edge=MAZE["edge"], ls="-", lw=0.5):
        return patches.Patch(facecolor=face, edgecolor=edge, hatch=hatch,
                             linewidth=lw * POINT, linestyle=ls, label=label)

    blank = patches.Patch(facecolor="none", edgecolor="none", linewidth=0,
                          label="")
    return [
        patch(MAZE["door_face"], r"unknown $B$ column", edge=MAZE["door"],
              ls=(0, (1.4, 1.1)), lw=0.7),
        patch(MAZE["wall"], "wall"),
        patch(MAZE["goal"], "goal G"),
        patch(MAZE["deadend"], "dead end"),
        _IceSwatch("ice"),
        blank,
        Line2D([], [], color=MAZE["risky"], ls=(0, (2.6, 1.4)),
               lw=1.1 * POINT, label="short risky route"),
        Line2D([], [], color=MAZE["safe"], ls="-", lw=1.2 * POINT,
               label="safe/informative route"),
    ]


def _scaled_rc(scale):
    """Journal rcParams with point sizes multiplied by ``scale``."""
    rc = dict(RC)
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "legend.fontsize", "xtick.labelsize", "ytick.labelsize",
                "xtick.major.size", "ytick.major.size", "xtick.major.pad",
                "ytick.major.pad", "axes.linewidth"):
        rc[key] = RC[key] * scale
    return rc


def _scaled_style(scale):
    """Curve styles with line widths multiplied by ``scale``."""
    return {key: {**spec, "lw": spec["lw"] * scale} for key, spec in STYLE.items()}


def write_slides():
    """Slide PNGs for fog-walls and the detour.

    Each scenario is a grid-plus-legend image and a curve image: two panels
    for fog-walls, three for the detour (goal, route, matrix error).
    Sized for a 13.3 in widescreen slide. Type keeps the journal figure's
    proportions, so a PNG can sit at full slide width without looking sparse.
    """
    global POINT, STYLE
    os.makedirs(FIG_OUT, exist_ok=True)
    scale = 13.0 / FIG_WIDTH_IN
    saved_style = STYLE
    POINT = scale
    STYLE = _scaled_style(scale)
    try:
        with plt.rc_context(_scaled_rc(scale)):
            _write_fog_grid()
            _write_fog_bc()
            _write_detour_grid()
            _write_detour_efg()
            _write_detour_eg()
    finally:
        POINT = 1.0
        STYLE = saved_style


def _row_major(handles, ncol):
    """Reorder handles so a column-filled legend reads left to right."""
    nrow = math.ceil(len(handles) / ncol)
    ordered = []
    for col in range(ncol):
        for row in range(nrow):
            i = row * ncol + col
            if i < len(handles):
                ordered.append(handles[i])
    return ordered


def _legend(fig, handles, y, ncol, fontsize, handler_map=None):
    """One frameless legend row, centered on the figure."""
    fig.legend(handles=_row_major(handles, ncol), ncol=ncol,
               loc="lower center", bbox_to_anchor=(0.5, y), frameon=False,
               fontsize=fontsize, handlelength=1.6, handleheight=0.9,
               handletextpad=0.45, columnspacing=1.05, borderaxespad=0.0,
               handler_map=handler_map)


def _write_fog_grid():
    """Panel (a) and the fog-walls glyph legend."""
    fig = plt.figure(figsize=(7.8, 8.2))
    ax = fig.add_axes([0.08, 0.28, 0.84, 0.62])
    draw_fog_walls(ax)
    ax.set_title(r"(a) fog-walls: learn $\theta_A$", loc="left", pad=2.5 * POINT)
    _legend(fig, fog_glyph_handles(), 0.035, ncol=2, fontsize=6.2 * POINT)
    out = os.path.join(FIG_OUT, "fog_walls_a.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def _write_fog_bc():
    """Panels (b) and (c), full width, with the planner legend only."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.30,
                        wspace=0.28)
    draw_learn_a_curves(axes[0], axes[1])
    _legend(fig, planner_handles(), 0.025, ncol=3, fontsize=6.5 * POINT)
    out = os.path.join(FIG_OUT, "fog_walls_bc.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def _write_detour_grid():
    """Panel (d) and the detour glyph legend."""
    gate_p = float(load_npz(LEARN_B_NPZ)["gate_p"])
    fig = plt.figure(figsize=(7.8, 8.2))
    ax = fig.add_axes([0.08, 0.28, 0.84, 0.62])
    draw_detour(ax)
    ax.set_title(r"(d) detour: learn $\theta_B$", loc="left", pad=2.5 * POINT)
    _legend(fig, detour_glyph_handles(gate_p), 0.035, ncol=2,
            fontsize=6.2 * POINT,
            handler_map={_IceSwatch: _IceSwatchHandler()})
    out = os.path.join(FIG_OUT, "detour_a.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def _detour_planner_legend(fig, y):
    """Planner key for the detour slides.

    Row-major, three columns. The blank third slot puts SP-Pref-Plan at the
    start of the lower row: EFE, EFE-dyn / Pref-Plan, Pref, Random.
    """
    by_name = dict(zip(DETOUR_PLANNERS, planner_handles(DETOUR_PLANNERS)))
    blank = Line2D([], [], color="none", lw=0, label="")
    _legend(fig, [
        by_name["SP-EFE"], by_name["SP-EFE-dyn"], blank,
        by_name["SP-Pref-Plan"], by_name["SP-Pref"], by_name["Random"],
    ], y, ncol=3, fontsize=6.5 * POINT)


def _write_detour_efg():
    """Panels (e), (f) and (g), with the planner legend only."""
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6))
    fig.subplots_adjust(left=0.044, right=0.985, top=0.91, bottom=0.30,
                        wspace=0.28)
    draw_learn_b_curves(axes[0], axes[1], err_ax=axes[2])
    _detour_planner_legend(fig, 0.025)
    out = os.path.join(FIG_OUT, "detour_efg.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def _write_detour_eg():
    """Panels (e) and (g) only, same size as the fog-walls curve slide."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.30,
                        wspace=0.28)
    draw_learn_b_curves(axes[0], err_ax=axes[1])
    _detour_planner_legend(fig, 0.025)
    out = os.path.join(FIG_OUT, "detour_eg.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    """One 2x3 composite: fog-walls learn-A (top), detour learn-B (bottom)."""
    os.makedirs(FIG_OUT, exist_ok=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            2, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
            gridspec_kw=dict(width_ratios=[1.06, 1.3, 1.3]))
        # Deterministic margins: the saved page is exactly the text width, so
        # LaTeX shows the figure 1:1 and these font sizes are the real ones.
        fig.subplots_adjust(left=0.05, right=0.982, top=0.945, bottom=0.285,
                            wspace=0.40, hspace=0.62)
        draw_learn_a(axes[0])
        gate_p = draw_learn_b(axes[1])

        fig.legend(handles=planner_handles(), ncol=6, loc="lower center",
                   bbox_to_anchor=(0.5, 0.122), frameon=False,
                   handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
                   borderaxespad=0.0)
        fig.legend(handles=glyph_handles(gate_p), ncol=5, loc="lower center",
                   bbox_to_anchor=(0.5, -0.004), fontsize=6.0, frameon=False,
                   handlelength=1.3, handleheight=0.9, handletextpad=0.45,
                   columnspacing=0.8, labelspacing=0.35, borderaxespad=0.0)

        out = os.path.join(FIG_OUT, "exp_learning.pdf")
        fig.savefig(out)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    if "--slides" in sys.argv:
        write_slides()
    else:
        main()
