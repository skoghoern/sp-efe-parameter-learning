"""Partially mapped grid: a known risky shortcut vs an unmapped safe detour.

Design (``docs/unmapped_detour_results.md``): the B-learning analogue of
fog-walls. B is *partially* known. The mapped part contains a route that
works with probability ``gate_p`` and whose failure is absorbing; the
unmapped part (flat Dirichlet columns) contains the deterministic route.
So

- ``SP-Pref`` and ``SP-Pref-Plan`` have a known route that is good enough to
  switch posterior-mean optimism off (a flat column looks like a 1/N teleport,
  which is worth less than the gate as long as ``gate_p`` is clearly above
  ``1 − (1 − 1/N)^(H−1)``), and neither has an epistemic term for B: both
  should sit at ``gate_p`` with a bit-identical B̂ on the unmapped columns;
- ``SP-EFE`` is drawn into the unmapped region by Δ_B, which is
  survivable there (no unmapped cell neighbours the dead end), learns the
  columns, and — once Δ_B has decayed — exploits the deterministic
  detour at 1.00.

Two layouts (``s = x + y * grid_x``, y pointing up):

``"default"`` (5×4, 24 flat columns)::

    y=3:  u  u  u  u  u      u = unmapped cell (flat prior on its 4 columns)
    y=2:  u  #  #  #  G      # = wall (bump), G = goal (absorbing, known)
    y=1:  S  .  .  .  X      X = gate: NORTH -> G w.p. gate_p, else -> D
    y=0:  .  .  .  .  D      D = dead end (absorbing, known, non-lethal)

  Shortcut ``→→→→↑`` (5 steps, w.p. ``gate_p``); detour ``↑↑→→→→↓`` (7,
  deterministic). The detour is the action-index-order walk (N, then E),
  which is what lets *uncorrected* ``SP-Pref`` find it: its action-entropy
  bonus pulls it into the flat columns and index-order tie-breaks walk it
  along the top row.

``"mirror"`` (6×4, no pocket; 28 flat columns in cells mode)::

    y=3:  u  u  u  u  u  u      former top-left goal is unmapped
    y=2:  G  #  #  #  #  u
    y=1:  I  .  .  .  .  S      I = slippery ice (NORTH -> G w.p. gate_p, else D)
    y=0:  D  .  .  .  .  .

  Shortcut ``←←←←←↑`` (6 steps, w.p. ``gate_p``); detour ``↑↑←←←←←↓`` (8).
  At ``(5,3)`` NORTH and EAST bump and only WEST progresses. ``"mirror-nopocket"``
  is an alias of ``"mirror"``. ``"mirror-rest"`` adds a known absorbing rest cell.

Identity A. Action order NORTH / EAST / SOUTH / WEST; ``action_order``
relabels the action indices (e.g. ``"WSEN"``) — Δ_B is invariant to this,
index-order tie-breaks are not.

``noisy_tv=True`` turns the mapped cell ``T = (2, 0)`` into a **known,
unlearnable scrambler**: every action from T lands uniformly on the mapped
free cells. Its column is clamped, so Δ_B is zero there, but its
predictive entropy is permanent (noisy-TV demo for the dyn channel).
"""

from __future__ import annotations

from collections import deque

import numpy as np

N_ACTIONS = 4
ACTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
ACTION_SYMBOLS = ["↑", "→", "↓", "←"]
ACTION_LETTERS = "NESW"
NORTH, EAST, SOUTH, WEST = range(N_ACTIONS)
ACTION_DXY = {NORTH: (0, 1), EAST: (1, 0), SOUTH: (0, -1), WEST: (-1, 0)}

LAYOUTS = {
    "default": dict(
        grid=(5, 4), start=(0, 1), goal=(4, 2), gate=(4, 1), deadend=(4, 0),
        walls=((1, 2), (2, 2), (3, 2)),
        unmapped=((0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)),
        noisy_tv=(2, 0),
    ),
    "mirror": dict(
        grid=(6, 4), start=(5, 1), goal=(0, 2), gate=(0, 1), deadend=(0, 0),
        walls=((1, 2), (2, 2), (3, 2), (4, 2)),
        unmapped=((0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2)),
        noisy_tv=(2, 0),
    ),
    # Known non-lethal absorber R=(4,0); not a lure (needs a goal chance).
    "mirror-rest": dict(
        grid=(6, 4), start=(5, 1), goal=(0, 2), gate=(0, 1), deadend=(0, 0),
        walls=((1, 2), (2, 2), (3, 2), (4, 2)),
        unmapped=((0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2)),
        rest=((4, 0),),
        noisy_tv=(2, 0),
    ),
}
LAYOUTS["mirror-nopocket"] = dict(LAYOUTS["mirror"])  # alias; pocket removed

DEFAULT_LAYOUT = "default"
UNMAPPED_MODES = ("cells", "doors")
DEFAULT_UNMAPPED_MODE = "cells"

# Default-layout constants (kept for callers / tests of the original grid).
GRID_X, GRID_Y = LAYOUTS[DEFAULT_LAYOUT]["grid"]
N_STATES = GRID_X * GRID_Y
START_XY = LAYOUTS[DEFAULT_LAYOUT]["start"]
GOAL_XY = LAYOUTS[DEFAULT_LAYOUT]["goal"]
GATE_XY = LAYOUTS[DEFAULT_LAYOUT]["gate"]
DEADEND_XY = LAYOUTS[DEFAULT_LAYOUT]["deadend"]
WALL_XY = LAYOUTS[DEFAULT_LAYOUT]["walls"]
UNMAPPED_XY = LAYOUTS[DEFAULT_LAYOUT]["unmapped"]
NOISY_TV_XY = LAYOUTS[DEFAULT_LAYOUT]["noisy_tv"]
GATE_ACTION = NORTH          # true (unpermuted) gate action

DEFAULT_GATE_P = 0.5
DEFAULT_DETOUR_LENGTH = 7    # default layout (mirror: 8)
DEFAULT_SHORTCUT_LENGTH = 5  # default layout (mirror: 6)
DEFAULT_HORIZON = DEFAULT_DETOUR_LENGTH
DEFAULT_STEPS = DEFAULT_DETOUR_LENGTH + 1   # slack 1; failure is absorbing


def xy_to_state(x, y, grid_x=GRID_X):
    return int(x) + int(y) * int(grid_x)


def state_to_xy(s, grid_x=GRID_X):
    return int(s) % int(grid_x), int(s) // int(grid_x)


def parse_action_order(order):
    """``"WSEN"`` → permutation ``perm[index] = true action``."""
    letters = str(order).upper()
    if sorted(letters) != sorted(ACTION_LETTERS):
        raise ValueError(f"action_order must be a permutation of "
                         f"{ACTION_LETTERS!r}, got {order!r}")
    return tuple(ACTION_LETTERS.index(c) for c in letters)


class UnmappedDetourGrid:
    """Grid with a stochastic gate and an unmapped deterministic detour."""

    def __init__(self, gate_p=DEFAULT_GATE_P, noisy_tv=False,
                 layout=DEFAULT_LAYOUT, action_order=ACTION_LETTERS,
                 unmapped_mode=DEFAULT_UNMAPPED_MODE):
        p = float(gate_p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"gate_p must be in [0, 1], got {p}")
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; choose from {tuple(LAYOUTS)}")
        if unmapped_mode not in UNMAPPED_MODES:
            raise ValueError(f"unknown unmapped_mode {unmapped_mode!r}; "
                             f"choose from {UNMAPPED_MODES}")
        spec = LAYOUTS[layout]
        self.layout = layout
        self.unmapped_mode = unmapped_mode
        self.gate_p = p
        # action index (as seen by the agent) -> true cardinal action
        self.action_perm = parse_action_order(action_order)
        self.action_order = str(action_order).upper()
        self.grid_x, self.grid_y = spec["grid"]
        self.n_states = self.grid_x * self.grid_y
        self.n_obs = self.n_states
        self.n_actions = N_ACTIONS
        self.start_xy, self.goal_xy = spec["start"], spec["goal"]
        self.gate_xy, self.deadend_xy = spec["gate"], spec["deadend"]
        self.start_idx = self.xy_to_state(*self.start_xy)
        self.goal_idx = self.xy_to_state(*self.goal_xy)
        self.gate_idx = self.xy_to_state(*self.gate_xy)
        self.deadend_idx = self.xy_to_state(*self.deadend_xy)
        self.gate_action = self.action_perm.index(GATE_ACTION)
        self.wall_idx = frozenset(self.xy_to_state(*c) for c in spec["walls"])
        self.rest_idx = frozenset(self.xy_to_state(*c) for c in spec.get("rest", ()))
        # Every absorbing non-goal cell (dead end + rest cells).
        self.absorb_idx = frozenset({self.deadend_idx}) | self.rest_idx
        self.unmapped_idx = tuple(self.xy_to_state(*c) for c in spec["unmapped"])
        self.unmapped_set = frozenset(self.unmapped_idx)
        self.noisy_tv = bool(noisy_tv)
        self.tv_idx = self.xy_to_state(*spec["noisy_tv"]) if self.noisy_tv else None
        self.tv_targets = self._mapped_free_cells() if self.noisy_tv else ()
        self.detour_path = self.shortest_path(allow_gate=False)
        self.shortcut_path = self.shortest_path(allow_gate=True)
        self.detour_length = len(self.detour_path)
        self.shortcut_length = len(self.shortcut_path)
        # Cells a detour trajectory is allowed to touch: exactly the detour.
        self.detour_cells = frozenset(self._path_states(self.detour_path))
        self._check_layout()
        # Flat Dirichlet columns ``(state, action)``:
        #   "cells": every action of every unmapped cell;
        #   "doors": only the detour's own (cell, action) pairs — the agent
        #            knows what the other three actions do at each detour
        #            cell (bumps / back), not where the door leads.
        if unmapped_mode == "cells":
            self.unmapped_cols = tuple((u, a) for u in self.unmapped_idx
                                       for a in range(N_ACTIONS))
        else:
            states = self._path_states(self.detour_path)
            self.unmapped_cols = tuple(
                (s, a) for s, a in zip(states[:-1], self.detour_path)
                if self.is_unmapped(s))

    # -- geometry -----------------------------------------------------------
    def xy_to_state(self, x, y):
        return xy_to_state(x, y, self.grid_x)

    def state_to_xy(self, s):
        return state_to_xy(s, self.grid_x)

    def is_unmapped(self, s):
        return int(s) in self.unmapped_set

    def true_action(self, action):
        """Agent action index → true cardinal action."""
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"invalid action {action}")
        return self.action_perm[action]

    def action_symbol(self, action):
        return ACTION_SYMBOLS[self.true_action(action)]

    def _mapped_free_cells(self):
        """Mapped, non-absorbing, non-gate, non-wall cells (TV targets)."""
        skip = {self.goal_idx, self.gate_idx} | self.absorb_idx | self.wall_idx
        return tuple(s for s in range(self.n_states)
                     if s not in skip and not self.is_unmapped(s))

    def move_xy(self, x, y, action):
        """Deterministic part of the dynamics (walls and edges bump)."""
        s = self.xy_to_state(x, y)
        if s == self.goal_idx or s in self.absorb_idx or s in self.wall_idx:
            return x, y                       # absorbing / unreachable
        dx, dy = ACTION_DXY[self.true_action(action)]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.grid_x and 0 <= ny < self.grid_y):
            return x, y
        if self.xy_to_state(nx, ny) in self.wall_idx:
            return x, y
        return nx, ny

    def move(self, s, action):
        return self.xy_to_state(*self.move_xy(*self.state_to_xy(s), action))

    def is_gate(self, s, action):
        return int(s) == self.gate_idx and int(action) == self.gate_action

    def is_noisy_tv(self, s):
        return self.noisy_tv and int(s) == self.tv_idx

    def transition_probs(self, s, action):
        """``{next_state: prob}`` including the gate and the scrambler."""
        if self.is_gate(s, action):
            return {self.goal_idx: self.gate_p,
                    self.deadend_idx: 1.0 - self.gate_p}
        if self.is_noisy_tv(s):
            return {t: 1.0 / len(self.tv_targets) for t in self.tv_targets}
        return {self.move(s, action): 1.0}

    def shortest_path(self, allow_gate):
        """BFS start → goal over deterministic moves (gate optional)."""
        start, goal = self.start_idx, self.goal_idx
        queue = deque([(start, ())])
        seen = {start}
        while queue:
            s, path = queue.popleft()
            if s == goal:
                return path
            if self.is_noisy_tv(s):
                continue                      # scrambler: no planned exit
            for action in range(N_ACTIONS):
                if self.is_gate(s, action):
                    if not allow_gate:
                        continue
                    nxt = self.goal_idx       # optimistic gate for BFS
                else:
                    nxt = self.move(s, action)
                if nxt == s or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + (action,)))
        raise RuntimeError("goal unreachable")

    def _path_states(self, path):
        s = self.start_idx
        states = [s]
        for a in path:
            s = self.move(s, a)
            states.append(s)
        return states

    def _check_layout(self):
        """Design invariants: detour is unmapped-only; unmapped is safe."""
        for s in self._path_states(self.detour_path)[1:]:
            if s != self.goal_idx and not self.is_unmapped(s):
                raise RuntimeError(f"detour crosses mapped cell {self.state_to_xy(s)}")
        if self._path_states(self.detour_path)[-1] != self.goal_idx:
            raise RuntimeError("detour does not end at the goal")
        # Rest cells are *allowed* next to unmapped cells: they are known,
        # so Δ_B is zero there and a curious agent learns the way in from a
        # sample or two; the dead end / gate / TV stay off-limits.
        for u in self.unmapped_idx:
            for a in range(N_ACTIONS):
                nxt = self.move(u, a)
                if (nxt == self.deadend_idx or self.is_gate(u, a)
                        or self.is_noisy_tv(nxt)):
                    raise RuntimeError(f"unmapped cell {self.state_to_xy(u)} "
                                       f"touches the dead end / gate / TV")

    def optimism_teleport_prob(self, horizon):
        """Chance a uniform-teleport walk of ``horizon − 1`` steps hits G.

        This is what a flat column is worth to a posterior-mean planner:
        keep ``gate_p`` clearly above it so KL prefers the known gate.
        """
        h = max(0, int(horizon) - 1)
        return 1.0 - (1.0 - 1.0 / self.n_states) ** h

    # -- tensors ------------------------------------------------------------
    def build_A(self):
        """Identity likelihood ``A[0, obs, state]`` as ``(1, S, S)``."""
        return np.eye(self.n_states, dtype=np.float32)[None]

    def build_B(self):
        """True transitions ``B[next, current, action]`` (gate stochastic)."""
        n = self.n_states
        B = np.zeros((n, n, N_ACTIONS), np.float32)
        for s in range(n):
            for action in range(N_ACTIONS):
                for nxt, p in self.transition_probs(s, action).items():
                    B[nxt, s, action] += p
        return B

    def build_alpha_B(self, prior_count=0.1, clamp=1.0e6, known=False):
        """Dirichlet counts: mapped columns clamped, unmapped flat.

        ``known=True`` clamps every column (the planning ceiling).
        """
        if prior_count <= 0:
            raise ValueError(f"prior_count must be > 0, got {prior_count}")
        alpha = float(clamp) * self.build_B()
        if not known:
            for s, a in self.unmapped_cols:
                alpha[:, s, a] = float(prior_count)
        return alpha.astype(np.float32)

    def unmapped_mask(self):
        """Boolean ``(S, S, U)`` mask selecting the flat columns."""
        mask = np.zeros((self.n_states, self.n_states, N_ACTIONS), bool)
        for s, a in self.unmapped_cols:
            mask[:, s, a] = True
        return mask

    def build_D(self):
        D = np.zeros(self.n_states, np.float32)
        D[self.start_idx] = 1.0
        return D

    def build_goal(self, preference=4.0):
        C = np.zeros(self.n_states, np.float64)
        C[self.goal_idx] = float(preference)
        g = np.exp(C - C.max())
        return (g / g.sum()).astype(np.float32)

    def _waypoints(self, path):
        """Cell centres along a plan; the icy cell jumps to the goal."""
        s = self.start_idx
        pts = [self.state_to_xy(s)]
        for a in path:
            if self.is_gate(s, a):
                pts.append(self.goal_xy)
                s = self.goal_idx
            else:
                s = self.move(s, a)
                pts.append(self.state_to_xy(s))
        return pts

    def depict_ax(self, ax=None, show_paths=True, legend=True, panel=None):
        """Camera-ready layout drawing. Returns the axes."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        gx, gy = self.grid_x, self.grid_y
        if ax is None:
            _, ax = plt.subplots(figsize=(0.55 * gx + 1.9, 0.55 * gy + 0.7))
        ax.set_xlim(-0.04, gx + 0.04)
        ax.set_ylim(-0.04, gy + 0.04)
        ax.set_aspect("equal")
        ax.axis("off")

        detour_cells = frozenset(self._path_states(self.detour_path))
        pocket = frozenset(
            s for s in self.unmapped_idx if s not in detour_cells)
        C = _LAYOUT_COLORS
        edge = dict(edgecolor=C["edge"], lw=0.55)

        def _face(x, y):
            s = self.xy_to_state(x, y)
            if (x, y) == self.start_xy:
                return C["start"], None
            if (x, y) == self.goal_xy:
                return C["goal"], None
            if (x, y) == self.gate_xy:
                return C["ice"], None
            if (x, y) == self.deadend_xy:
                return C["deadend"], None
            if s in self.rest_idx:
                return C["rest"], None
            if self.is_noisy_tv(s):
                return C["tv"], None
            if s in self.wall_idx:
                return C["wall"], None
            if s in pocket:
                return C["pocket"], False
            if s in self.unmapped_set:
                return C["unmapped"], True
            return C["mapped"], False

        for y in range(gy):
            for x in range(gx):
                face, stipple = _face(x, y)
                kw = dict(edgecolor=C["ice_edge"], lw=0.8) if (x, y) == self.gate_xy else edge
                ax.add_patch(patches.Rectangle(
                    (x, y), 1, 1, facecolor=face, **kw))
                if stipple:
                    self._stipple(ax, x, y)

        ix, iy = self.gate_xy
        self._draw_ice_crystal(ax, ix + 0.5, iy + 0.5)

        labels = [(self.start_xy, "S"), (self.goal_xy, "G")]
        for s in self.rest_idx:
            labels.append((self.state_to_xy(s), "R"))
        if self.noisy_tv:
            labels.append((self.state_to_xy(self.tv_idx), "T"))
        for (x, y), text in labels:
            color = "white" if (x, y) == self.start_xy else C["ink"]
            ax.text(x + 0.5, y + 0.5, text, ha="center", va="center",
                    fontsize=11, color=color, zorder=6)
        for (x, y) in (self.state_to_xy(s) for s in pocket):
            ax.add_patch(patches.Rectangle(
                (x + 0.12, y + 0.12), 0.76, 0.76, fill=False,
                edgecolor="0.35", lw=0.6, ls=(0, (2.2, 1.2)), zorder=3))
            ax.text(x + 0.5, y + 0.5, "P", ha="center", va="center",
                    fontsize=9, color="0.2", zorder=6)

        if show_paths:
            self._draw_route(ax, self._waypoints(self.shortcut_path),
                             color=C["shortcut"], ls=(0, (3.2, 1.6)),
                             lw=1.35, rad=0.0, shrink=0.30)
            self._draw_route(ax, self._waypoints(self.detour_path),
                             color=C["detour"], ls="-", lw=1.45, rad=0.0,
                             shrink=0.30)
            self._draw_route(ax, [self.gate_xy, self.deadend_xy],
                             color=C["slip"], ls=(0, (1.1, 1.1)),
                             lw=1.2, rad=0.0, shrink=0.32)
            self._annotate_slip(ax)

        name = {"default": "Default", "mirror": "Mirror",
                "mirror-nopocket": "Mirror",
                "mirror-rest": "Mirror (rest cell)"}.get(
                    self.layout, self.layout)
        title = f"{name} layout"
        if panel:
            title = f"({panel})  {title}"
        ax.set_title(title, loc="left", fontsize=10, pad=5)

        if legend:
            ax.legend(handles=_layout_legend_handles(bool(pocket)),
                      loc="center left", bbox_to_anchor=(1.03, 0.5),
                      frameon=False, fontsize=8, handlelength=1.8,
                      borderaxespad=0.0)
        return ax

    def _annotate_slip(self, ax):
        """$p$ on the ice→goal hop, $1-p$ on the slip into the dead end."""
        C = _LAYOUT_COLORS
        gx, gy = self.gate_xy
        tx, ty = self.goal_xy
        dx, dy = self.deadend_xy
        mx, my = 0.5 * (gx + tx) + 0.5, 0.5 * (gy + ty) + 0.5
        if abs(tx - gx) + abs(ty - gy) == 1:
            # Adjacent hop: park $p$ on the wall beside G, not on the ice.
            mx = (gx - 0.45) if gx >= self.grid_x / 2 else (gx + 1.45)
            my = 0.5 * (gy + ty) + 0.55
        box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88)
        ax.text(mx, my, rf"$p={self.gate_p:g}$", color=C["shortcut"],
                fontsize=8, ha="center", va="center", zorder=7, bbox=box)
        ax.text(dx + 0.5, dy + 0.28, r"$1-p$", color=C["slip"], fontsize=8,
                ha="center", va="center", zorder=7, bbox=box)

    @staticmethod
    def _stipple(ax, x, y, n=3):
        """Sparse dots marking an unmapped cell (replaces a heavy hatch)."""
        C = _LAYOUT_COLORS
        xs, ys = [], []
        inset = 0.22
        for i in range(n):
            for j in range(n):
                xs.append(x + inset + (1 - 2 * inset) * (i + 0.5) / n)
                ys.append(y + inset + (1 - 2 * inset) * (j + 0.5) / n)
        ax.scatter(xs, ys, s=5.5, c=C["stipple"], marker="o", linewidths=0,
                   zorder=2, clip_on=True)

    @staticmethod
    def _draw_ice_crystal(ax, cx, cy, radius=0.20):
        """Six-point ice mark on the slippery cell."""
        import numpy as np
        color = _LAYOUT_COLORS["ice_mark"]
        for deg in (0.0, 60.0, 120.0):
            a = np.deg2rad(deg)
            ax.plot([cx - radius * np.cos(a), cx + radius * np.cos(a)],
                    [cy - radius * np.sin(a), cy + radius * np.sin(a)],
                    color=color, lw=0.9, solid_capstyle="round",
                    zorder=6, clip_on=True)

    @staticmethod
    def _draw_route(ax, pts, color, ls, lw, rad=0.0, shrink=0.28):
        """Arrows between successive cells; arc when the hop is not adjacent."""
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            manhattan = abs(dx) + abs(dy)
            if manhattan == 0:
                continue
            style = ("arc3,rad=-0.22" if manhattan > 1
                     else f"arc3,rad={rad}")
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            t = shrink / length
            ax.annotate(
                "",
                xy=(x1 + 0.5 - t * dx, y1 + 0.5 - t * dy),
                xytext=(x0 + 0.5 + t * dx, y0 + 0.5 + t * dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, mutation_scale=9.5,
                                connectionstyle=style,
                                shrinkA=0, shrinkB=0),
                zorder=4)

    def depict(self):
        """ASCII map, y decreasing down the page."""
        lines = []
        for y in range(self.grid_y - 1, -1, -1):
            row = []
            for x in range(self.grid_x):
                s = self.xy_to_state(x, y)
                if (x, y) == self.start_xy:
                    cell = "S"
                elif (x, y) == self.goal_xy:
                    cell = "G"
                elif (x, y) == self.gate_xy:
                    cell = "X"
                elif (x, y) == self.deadend_xy:
                    cell = "D"
                elif s in self.rest_idx:
                    cell = "R"
                elif self.is_noisy_tv(s):
                    cell = "T"
                elif s in self.wall_idx:
                    cell = "#"
                elif s in self.unmapped_set:
                    cell = "u"
                else:
                    cell = "."
                row.append(f" {cell}")
            lines.append("".join(row) + f"  y={y}")
        lines.append(" " + " ".join(str(x) for x in range(self.grid_x))
                     + f"   layout={self.layout}  gate p={self.gate_p:.2f}  "
                       f"shortcut {self.shortcut_length}  detour {self.detour_length}"
                       f"  unmapped {self.unmapped_mode}: {len(self.unmapped_cols)} cols"
                       f"  actions {self.action_order}"
                       + ("  noisy TV" if self.noisy_tv else ""))
        return "\n".join(lines)


class UnmappedDetourEnv:
    """Identity-A env with a stochastic gate. ``step`` → ``(obs, done, outcome)``.

    ``outcome`` is ``"GOAL"``, ``"DEADEND"``, ``"REST"`` or ``None``.
    """

    def __init__(self, grid=None, seed=0):
        self.grid = grid or UnmappedDetourGrid()
        self.rng = np.random.default_rng(seed)
        self.state = self.grid.start_idx

    def reset(self):
        self.state = self.grid.start_idx
        return int(self.state)

    def step(self, action):
        probs = self.grid.transition_probs(self.state, action)
        states = list(probs)
        p = np.asarray([probs[s] for s in states], dtype=np.float64)
        self.state = int(states[self.rng.choice(len(states), p=p / p.sum())])
        outcome = ("GOAL" if self.state == self.grid.goal_idx
                   else "DEADEND" if self.state == self.grid.deadend_idx
                   else "REST" if self.state in self.grid.rest_idx
                   else None)
        return int(self.state), outcome is not None, outcome


# Okabe–Ito tints for fills; saturated Okabe for the two routes.
_LAYOUT_COLORS = dict(
    mapped="#ffffff",
    unmapped="#e8eef3",
    pocket="#d9e0e7",
    wall="#6a6a6a",
    start="#7eb6d9",
    goal="#8ecf9f",
    ice="#e8f4fb",
    ice_mark="#3e6f8a",
    ice_edge="#6a93ab",
    deadend="#f0d5cc",
    rest="#d0d0d0",
    tv="#e6d4ea",
    detour="#0072B2",
    shortcut="#E69F00",
    slip="#D55E00",
    edge="#333333",
    ink="#1a1a1a",
    stipple="#b7c0c8",
)

_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def _layout_legend_handles(pocket=False):
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    C = _LAYOUT_COLORS
    edge = dict(edgecolor=C["edge"], lw=0.5)
    handles = [
        patches.Patch(facecolor=C["start"], label="Start", **edge),
        patches.Patch(facecolor=C["goal"], label="Goal", **edge),
        patches.Patch(facecolor=C["ice"], label="Slippery ice", **edge),
        patches.Patch(facecolor=C["deadend"], label="Dead end", **edge),
        patches.Patch(facecolor=C["unmapped"], label="Unmapped", **edge),
        patches.Patch(facecolor=C["wall"], label="Wall", **edge),
        Line2D([0], [0], color=C["detour"], lw=1.6, label="Safe detour"),
        Line2D([0], [0], color=C["shortcut"], lw=1.5, ls=(0, (3.2, 1.6)),
               label="Icy shortcut"),
        Line2D([0], [0], color=C["slip"], lw=1.3, ls=(0, (1.1, 1.1)),
               label="Slip"),
    ]
    if pocket:
        handles.insert(5, patches.Patch(
            facecolor=C["pocket"], label="Decoy pocket", **edge))
    return handles


def save_layout_figures(out_dir="figures/figures", layouts=None, dpi=300):
    """Write per-layout and combined camera-ready PNGs (and PDFs) to ``out_dir``.

    The published mirror panel is ``mirror-nopocket`` (the decoy pocket is
    not load-bearing) but is still saved as ``unmapped_detour_layout_mirror``.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update(_PAPER_RC)
    # (grid layout, output stem)
    published = layouts or (("default", "default"), ("mirror", "mirror"))
    paths = []
    grids = [UnmappedDetourGrid(layout=name) for name, _ in published]
    stems = [stem for _, stem in published]
    panels = "abcdefghijklmnopqrstuvwxyz"

    for grid, stem, panel in zip(grids, stems, panels):
        fig, ax = plt.subplots(
            figsize=(0.52 * grid.grid_x + 1.85, 0.52 * grid.grid_y + 0.55))
        grid.depict_ax(ax=ax, panel=panel)
        out = os.path.join(out_dir, f"unmapped_detour_layout_{stem}")
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=dpi)
        plt.close(fig)
        paths.append(f"{out}.png")

    fig, axes = plt.subplots(
        1, len(grids),
        figsize=(0.48 * sum(g.grid_x for g in grids) + 0.9, 3.15),
        gridspec_kw={"width_ratios": [g.grid_x for g in grids]})
    if len(grids) == 1:
        axes = [axes]
    for ax, grid, panel in zip(axes, grids, panels):
        grid.depict_ax(ax=ax, legend=False, panel=panel)
    fig.legend(handles=_layout_legend_handles(pocket=False),
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8,
               handlelength=1.7, columnspacing=1.2)
    fig.tight_layout()
    combined = os.path.join(out_dir, "unmapped_detour_layouts")
    for ext in ("png", "pdf"):
        fig.savefig(f"{combined}.{ext}", dpi=dpi)
    plt.close(fig)
    paths.append(f"{combined}.png")
    return tuple(paths)
