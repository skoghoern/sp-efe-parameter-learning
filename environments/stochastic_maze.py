"""Stochastic maze, ported from the RxInferExamples.jl notebook
``EFE Minimization via Message Passing.ipynb``.

A 5×5 gridworld where the agent navigates toward a goal state, impeded by:

- **sink states** that trap the agent (every action keeps it in place);
- **stochastic transition cells** ("drift"): the intended move succeeds
  with probability ``1 - p`` and with probability ``p`` the agent drifts in
  one of the cell's drift directions (uniformly), walls = stay. The default
  layout is a *conveyor + trap*: the two approach cells (1,2), (2,2) drift
  East with ``p = 0.5`` (the row "carries" the agent toward the goal), the
  last cell (3,2) drifts up/down with ``p = 0.7`` — straight into the two
  sinks that flank it (see ``DRIFT_SPEC``);
- **noisy observation cells**: the agent perceives its true position only
  20% of the time and with 80% probability observes one of the in-bounds
  adjacent cells (including diagonals), uniformly.

There is a short path to the goal, but it crosses the stochastic drift and
noisy-observation cells; a longer path along the top row is fully
deterministic and noise-free.

The default drift layout is chosen so that the preference-driven planners
(SP-Pref / SP-Pref-Plan) are lured into the short row — their route is
A-independent, so the conveyor's pull toward the goal is all they see — and
never take the top row, whereas the ambiguity-aware SP-EFE planner takes
the safe top-row detour on every seed. The notebook's original uniform drift
(0.7 over up/down) is available via
``build_B(drift_spec=NOTEBOOK_DRIFT_SPEC)``.

Coordinates: the notebook uses Julia ``CartesianIndex(row, col)`` with
row 1 at the top and column-major linear indices. Here we use the repo
convention ``(x, y)`` with ``y`` pointing up and linear state
``s = x + y * GRID_X``; the mapping is ``x = col - 1``, ``y = n_rows - row``.
Notebook cells (translated):

- goal (3,5) -> (4, 2);  start (3,1) -> (0, 2)
- sinks (2,4), (4,4) -> (3, 3), (3, 1)
- stochastic drift (3,2), (3,3), (3,4) -> (1, 2), (2, 2), (3, 2)
- noisy observations: the 3x3 block around the central drift row plus
  (5,3) -> (1..3, 1..3) and (2, 0)

Actions: NORTH (y+1), EAST (x+1), SOUTH (y-1), WEST (x-1); walls = stay.
(The notebook enumerates South/North/East/West; the order is arbitrary.)

The observation model is a single modality (obs = state index, possibly
noisy). ``build_A`` returns the planner-native padded ``(F=1, C, S)``
layout; ``build_A_matrix`` the raw (C, S) matrix used by the environment.

``depict`` reproduces the notebook's visualization (green goal, red sinks,
grey noisy cells, tan drift bands, blue agent) and ``animate_trajectory``
its trajectory GIF.
"""

from __future__ import annotations

import numpy as np

GRID_X, GRID_Y = 5, 5
N_STATES = GRID_X * GRID_Y
N_ACTIONS = 4
ACTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
ACTION_SYMBOLS = ["↑", "→", "↓", "←"]
# action -> (dx, dy); NORTH increases y
ACTION_DXY = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# Notebook example_grid, translated to (x, y) with y up (see module doc)
SINK_STATES = ((3, 3), (3, 1))
STOCHASTIC_STATES = ((1, 2), (2, 2), (3, 2))
NOISY_OBSERVATIONS = (
    (1, 3), (2, 3), (3, 3),
    (1, 2), (2, 2), (3, 2),
    (1, 1), (2, 1), (3, 1),
    (2, 0),
)
GOAL_XY = (4, 2)
START_XY = (0, 2)
OBS_NOISE = 0.8         # off-cell observation mass in noisy cells

# Drift spec: {(x, y): (drift_prob, drift_actions)}; drift mass is spread
# uniformly over drift_actions, the remainder follows the intended action.
NORTH, EAST, SOUTH, WEST = range(4)
DRIFT_SPEC = {
    (1, 2): (0.5, (EAST,)),           # conveyor toward the goal
    (2, 2): (0.5, (EAST,)),
    (3, 2): (0.7, (NORTH, SOUTH)),    # trap: both neighbours are sinks
}
# Original uniform drift (0.7 over up/down), kept so tests can compare layouts.
NOTEBOOK_DRIFT_SPEC = {c: (0.7, (NORTH, SOUTH)) for c in STOCHASTIC_STATES}


def xy_to_state(x, y, grid_x=None):
    gx = GRID_X if grid_x is None else int(grid_x)
    return int(x) + int(y) * gx


def state_to_xy(s, grid_x=None):
    gx = GRID_X if grid_x is None else int(grid_x)
    return int(s) % gx, int(s) // gx


GOAL_STATE = xy_to_state(*GOAL_XY)
START_STATE = xy_to_state(*START_XY)

# Fog maze (docs/stochastic_maze_fog_results.md): four sinks, one non-lethal
# drift, a 5-cell fog block whose observations are uniform over the block.
SINKS4 = ((2, 1), (2, 3), (3, 1), (3, 3))
FOG_CELLS = ((1, 1), (1, 2), (1, 3), (2, 2), (3, 2))
FOG_DRIFT_SPEC = {(1, 2): (0.6, (NORTH, SOUTH))}
FOG_NOISY_OBSERVATIONS = ((2, 0),)
# Fog-walls (docs/fog_walls_learnA_results.md): the fog maze with its four
# sinks replaced by impassable walls. Mislocalisation in the fog now costs
# bumps and detours instead of the episode, so a novelty-driven learner can
# survive long enough to learn which cells are foggy.
FOG_WALLS = SINKS4
# Cells on the 8-step deterministic top-row route (start → goal).
TOP_ROW_CELLS = (
    (0, 2), (0, 3), (0, 4),
    (1, 4), (2, 4), (3, 4), (4, 4),
    (4, 3), (4, 2),
)


def _move(x, y, action, grid_x=None, grid_y=None):
    """Deterministic move destination; walls stay put."""
    gx = GRID_X if grid_x is None else int(grid_x)
    gy = GRID_Y if grid_y is None else int(grid_y)
    dx, dy = ACTION_DXY[action]
    nx, ny = x + dx, y + dy
    if 0 <= nx < gx and 0 <= ny < gy:
        return nx, ny
    return x, y


def build_B(sink_states=SINK_STATES, drift_spec=DRIFT_SPEC,
            grid_x=None, grid_y=None, wall_states=()):
    """Transition tensor B[next, current, action], (S, S, 4).

    Sinks self-loop. A drift cell ``(x, y): (p, dirs)`` mixes
    ``(1-p) * move(intended)`` with ``p/len(dirs) * Σ_d move(d)``; other
    cells move deterministically, grid edges = stay. ``wall_states`` are
    impassable interior cells: any move into one is a bump (stay), so they
    are never occupied (their own rows self-loop). Unlike sinks they are
    non-lethal — a bump costs a step, not the episode.
    """
    gx = GRID_X if grid_x is None else int(grid_x)
    gy = GRID_Y if grid_y is None else int(grid_y)
    n_states = gx * gy
    sink = {tuple(c) for c in sink_states}
    wall = {tuple(c) for c in wall_states}

    def _dest(x, y, action):
        nx, ny = _move(x, y, action, gx, gy)
        return (x, y) if (nx, ny) in wall else (nx, ny)

    B = np.zeros((n_states, n_states, N_ACTIONS), np.float64)
    for y in range(gy):
        for x in range(gx):
            s = xy_to_state(x, y, grid_x=gx)
            if (x, y) in sink or (x, y) in wall:
                B[s, s, :] = 1.0
                continue
            for a in range(N_ACTIONS):
                if (x, y) in drift_spec:
                    p, dirs = drift_spec[(x, y)]
                    nxt = xy_to_state(*_dest(x, y, a), grid_x=gx)
                    B[nxt, s, a] += 1.0 - p
                    for d in dirs:
                        dxt = xy_to_state(*_dest(x, y, d), grid_x=gx)
                        B[dxt, s, a] += p / len(dirs)
                else:
                    nxt = xy_to_state(*_dest(x, y, a), grid_x=gx)
                    B[nxt, s, a] = 1.0
    return B.astype(np.float32)


def _neighbors(x, y, grid_x=None, grid_y=None):
    """In-bounds neighbours, including diagonals."""
    gx = GRID_X if grid_x is None else int(grid_x)
    gy = GRID_Y if grid_y is None else int(grid_y)
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < gx and 0 <= ny < gy:
                out.append(xy_to_state(nx, ny, grid_x=gx))
    return out


def build_A_matrix(noisy_observations=NOISY_OBSERVATIONS,
                   obs_noise=OBS_NOISE, grid_x=None, grid_y=None):
    """Observation matrix A[obs, state], (S, S)."""
    gx = GRID_X if grid_x is None else int(grid_x)
    gy = GRID_Y if grid_y is None else int(grid_y)
    n_states = gx * gy
    A = np.eye(n_states, dtype=np.float64)
    for (x, y) in noisy_observations:
        s = xy_to_state(x, y, grid_x=gx)
        A[:, s] = 0.0
        A[s, s] = 1.0 - obs_noise
        adj = _neighbors(x, y, grid_x=gx, grid_y=gy)
        for a_s in adj:
            A[a_s, s] = obs_noise / len(adj)
    return A.astype(np.float32)


def build_A(noisy_observations=NOISY_OBSERVATIONS, obs_noise=OBS_NOISE,
            grid_x=None, grid_y=None):
    """Planner-native single-modality likelihood (1, S, S)."""
    return build_A_matrix(
        noisy_observations, obs_noise, grid_x=grid_x, grid_y=grid_y)[None]


def valid_A_mask(n_states=None):
    """(1, S, S) mask: single full modality, all valid."""
    n = N_STATES if n_states is None else int(n_states)
    return np.ones((1, n, n), bool)


def build_D(start_state=START_STATE, n_states=None):
    """Start prior: delta on ``start_state``."""
    n = N_STATES if n_states is None else int(n_states)
    D = np.zeros(n, np.float32)
    D[int(start_state)] = 1.0
    return D


def build_C(preference=4.0, goal_state=GOAL_STATE, n_states=None):
    """Log-preference over states: +pref at the goal, 0 elsewhere."""
    n = N_STATES if n_states is None else int(n_states)
    C = np.zeros(n, np.float32)
    C[int(goal_state)] = float(preference)
    return C


def build_goal(preference=4.0, goal_state=GOAL_STATE, n_states=None):
    """Normalize exp(C) -> probability goal for the conjugate planners."""
    C = build_C(preference=preference, goal_state=goal_state, n_states=n_states)
    g = np.exp(C - C.max())
    return (g / g.sum()).astype(np.float32)


def apply_fog_columns(A, fog_cells, grid_x):
    """Overwrite fog-cell columns to uniform over the fog block."""
    idxs = [xy_to_state(*c, grid_x=grid_x) for c in fog_cells]
    n = len(idxs)
    if n == 0:
        return A
    for s in idxs:
        A[:, s] = 0.0
        for o in idxs:
            A[o, s] = 1.0 / n
    return A


def build_fog_A_matrix():
    """5x5 fog-block likelihood: uniform 1/5 over FOG_CELLS, (2,0) still noisy."""
    A = build_A_matrix(noisy_observations=FOG_NOISY_OBSERVATIONS)
    return apply_fog_columns(A, FOG_CELLS, GRID_X).astype(np.float32)


MAZE_LAYOUTS = ("fog-walls", "fog", "trap")


def _meta(name, grid_x, grid_y, start_xy, goal_xy, sink_states, drift_spec,
          noisy_observations, fog_cells, top_row_cells, horizon,
          wall_states=()):
    n = grid_x * grid_y
    return dict(
        name=name, grid_x=grid_x, grid_y=grid_y, n_states=n,
        start_xy=start_xy, goal_xy=goal_xy,
        start_state=xy_to_state(*start_xy, grid_x=grid_x),
        goal_state=xy_to_state(*goal_xy, grid_x=grid_x),
        sink_states=sink_states, drift_spec=drift_spec,
        noisy_observations=noisy_observations, fog_cells=fog_cells,
        top_row_cells=top_row_cells, horizon=horizon,
        wall_states=tuple(wall_states),
    )


def maze_tensors(name="fog-walls"):
    """Planner A ``(1, C, S)``, B, and layout meta."""
    if name == "trap":
        A = build_A()
        B = build_B()
        meta = _meta(
            "trap", GRID_X, GRID_Y, START_XY, GOAL_XY, SINK_STATES,
            DRIFT_SPEC, NOISY_OBSERVATIONS, (), TOP_ROW_CELLS, 11)
    elif name == "fog":
        A = build_fog_A_matrix()[None]
        B = build_B(sink_states=SINKS4, drift_spec=FOG_DRIFT_SPEC)
        meta = _meta(
            "fog", GRID_X, GRID_Y, START_XY, GOAL_XY, SINKS4,
            FOG_DRIFT_SPEC, FOG_NOISY_OBSERVATIONS, FOG_CELLS, TOP_ROW_CELLS, 11)
    elif name == "fog-walls":
        A = build_fog_A_matrix()[None]
        B = build_B(sink_states=(), drift_spec=FOG_DRIFT_SPEC,
                    wall_states=FOG_WALLS)
        meta = _meta(
            "fog-walls", GRID_X, GRID_Y, START_XY, GOAL_XY, (),
            FOG_DRIFT_SPEC, FOG_NOISY_OBSERVATIONS, FOG_CELLS, TOP_ROW_CELLS,
            11, wall_states=FOG_WALLS)
    else:
        raise ValueError(f"unknown maze {name!r}; choose from {MAZE_LAYOUTS}")
    return A, B, meta


class StochasticMazeEnv:
    """One episode of the stochastic maze."""

    def __init__(self, rng=None, start_state=None,
                 goal_state=None, maze="fog-walls", A=None, B=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        A_pl, B_def, meta = maze_tensors(maze)
        if A is None:
            A = A_pl[0]
        if B is None:
            B = B_def
        self.B = np.asarray(B, np.float64)
        self.A = np.asarray(A, np.float64)
        self.maze = maze
        self.meta = meta
        self.grid_x = int(meta["grid_x"])
        self.n_states = int(self.B.shape[0])
        self.start_state = (int(start_state) if start_state is not None
                            else int(meta["start_state"]))
        self.goal_state = (int(goal_state) if goal_state is not None
                           else int(meta["goal_state"]))
        self.state = self.start_state

    def reset(self, start_state=None):
        """Reset to ``start_state`` and sample the initial observation."""
        self.state = int(start_state) if start_state is not None \
            else self.start_state
        return self.observe()

    def observe(self):
        """Sample an observation from the current state's A column."""
        p = self.A[:, self.state]
        n_obs = self.A.shape[0]
        return int(self.rng.choice(n_obs, p=p / p.sum()))

    def reward(self):
        """1 at the goal, 0 elsewhere."""
        return 1.0 if self.state == self.goal_state else 0.0

    def step(self, action):
        """Apply an action; returns ``(observation, reward)``."""
        p = self.B[:, self.state, int(action)]
        self.state = int(self.rng.choice(self.n_states, p=p / p.sum()))
        return self.observe(), self.reward()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def depict(player=None, ax=None, drift_spec=DRIFT_SPEC,
           sink_states=SINK_STATES, noisy_observations=NOISY_OBSERVATIONS,
           fog_cells=(), grid_x=None, grid_y=None, goal_xy=None,
           wall_states=()):
    """Draw the maze and return the axes."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    gx = GRID_X if grid_x is None else int(grid_x)
    gy = GRID_Y if grid_y is None else int(grid_y)
    goal = GOAL_XY if goal_xy is None else tuple(goal_xy)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, gx)
    ax.set_ylim(0, gy)
    ax.set_aspect("equal")
    ax.axis("off")

    sink = set(sink_states)
    wall = set(wall_states)
    noisy = set(noisy_observations)
    fog = set(fog_cells)
    for y in range(gy):
        for x in range(gx):
            color = ("tomato" if (x, y) in sink
                     else "0.25" if (x, y) in wall
                     else "palegreen" if (x, y) == goal
                     else "0.72" if (x, y) in fog
                     else "0.90" if (x, y) in noisy
                     else "white")
            ax.add_patch(patches.Rectangle(
                (x, y), 1, 1, facecolor=color, edgecolor="black", lw=0.8,
                hatch="////" if (x, y) in fog and (x, y) not in sink else None))
            if (x, y) in drift_spec:
                p, dirs = drift_spec[(x, y)]
                horizontal = all(d in (EAST, WEST) for d in dirs)
                for k in (1, 3, 5):  # three hatch bands
                    rect = ((x + k / 7, y, 1 / 7, 1) if horizontal
                            else (x, y + k / 7, 1, 1 / 7))
                    ax.add_patch(patches.Rectangle(
                        rect[:2], rect[2], rect[3], facecolor="tan",
                        edgecolor="none"))
                for d in dirs:
                    dx, dy = ACTION_DXY[d]
                    ax.annotate("", xy=(x + 0.5 + 0.38 * dx, y + 0.5 + 0.38 * dy),
                                xytext=(x + 0.5 + 0.12 * dx, y + 0.5 + 0.12 * dy),
                                arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                                color="saddlebrown"),
                                zorder=3)

    if player is not None:
        px, py = (state_to_xy(player, grid_x=gx) if np.isscalar(player)
                  else player)
        ax.add_patch(patches.Circle((px + 0.5, py + 0.5), 0.28,
                                    color="steelblue", zorder=5))
    return ax


def animate_trajectory(states, path="simulation.gif", fps=4):
    """Save a GIF of a state-index trajectory."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(5, 5))

    def show(i):
        ax.clear()
        depict(states[i], ax=ax)

    anim = FuncAnimation(fig, show, frames=len(states), interval=1000 / fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    return path
