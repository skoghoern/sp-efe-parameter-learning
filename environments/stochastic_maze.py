"""Fog-walls maze: the paper's learn-θ_A benchmark.

A 5×5 gridworld. The agent starts at ``(0, 2)`` and the goal is ``(4, 2)``
(``y`` points up; state ``s = x + y * GRID_X``). Two routes:

- a 4-step middle row through a drift cell and a fog block;
- an 8-step top row that is deterministic and observation-exact.

``FOG_CELLS`` share one observation: each column of ``A`` is uniform over
the block, so the agent cannot tell which fog cell it occupies. ``(1, 2)``
also drifts north or south with probability 0.6. ``(2, 0)`` is noisy in
the usual way: the true cell with probability ``1 - OBS_NOISE``, otherwise
a uniform in-bounds neighbor (diagonals included).

``FOG_WALLS`` are impassable interior cells (a move into one stays put).
Mislocalisation in the fog costs a bump and a detour, so a learner can
survive long enough to learn which cells are foggy.

Actions: NORTH (y+1), EAST (x+1), SOUTH (y-1), WEST (x-1); grid edges
stay. ``build_A`` is the planner-native ``(1, S, S)`` likelihood;
``build_A_matrix`` the ``(S, S)`` matrix the environment samples.
``depict`` draws the grid (green goal, dark walls, hatched fog, tan drift,
blue agent).
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

GOAL_XY = (4, 2)
START_XY = (0, 2)
OBS_NOISE = 0.8         # off-cell observation mass in the noisy cell

# Drift spec: {(x, y): (drift_prob, drift_actions)}; drift mass is spread
# uniformly over drift_actions, the remainder follows the intended action.
NORTH, EAST, SOUTH, WEST = range(4)


def xy_to_state(x, y, grid_x=None):
    gx = GRID_X if grid_x is None else int(grid_x)
    return int(x) + int(y) * gx


def state_to_xy(s, grid_x=None):
    gx = GRID_X if grid_x is None else int(grid_x)
    return int(s) % gx, int(s) // gx


GOAL_STATE = xy_to_state(*GOAL_XY)
START_STATE = xy_to_state(*START_XY)

# Fog block: observations are uniform over these cells. The drift cell is
# inside the block. FOG_WALLS are impassable (bump = stay).
FOG_CELLS = ((1, 1), (1, 2), (1, 3), (2, 2), (3, 2))
FOG_DRIFT_SPEC = {(1, 2): (0.6, (NORTH, SOUTH))}
FOG_NOISY_OBSERVATIONS = ((2, 0),)
FOG_WALLS = ((2, 1), (2, 3), (3, 1), (3, 3))
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


def build_B(sink_states=(), drift_spec=FOG_DRIFT_SPEC,
            grid_x=None, grid_y=None, wall_states=FOG_WALLS):
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


def build_A_matrix(noisy_observations=FOG_NOISY_OBSERVATIONS,
                   obs_noise=OBS_NOISE, grid_x=None, grid_y=None,
                   fog_cells=FOG_CELLS):
    """Observation matrix A[obs, state], (S, S).

    Noisy cells put ``obs_noise`` on in-bounds neighbors. Fog-cell columns
    are then overwritten to uniform over ``fog_cells``.
    """
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
    if fog_cells:
        apply_fog_columns(A, fog_cells, gx)
    return A.astype(np.float32)


def build_A(noisy_observations=FOG_NOISY_OBSERVATIONS, obs_noise=OBS_NOISE,
            grid_x=None, grid_y=None, fog_cells=FOG_CELLS):
    """Planner-native fog-walls likelihood (1, S, S)."""
    return build_A_matrix(
        noisy_observations, obs_noise, grid_x=grid_x, grid_y=grid_y,
        fog_cells=fog_cells)[None]


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
    """Fog-walls likelihood: uniform 1/5 over FOG_CELLS, (2, 0) still noisy."""
    return build_A_matrix()


MAZE_LAYOUTS = ("fog-walls",)


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
    """Planner A ``(1, C, S)``, B, and layout meta for fog-walls."""
    if name != "fog-walls":
        raise ValueError(f"unknown maze {name!r}; choose from {MAZE_LAYOUTS}")
    A = build_A()
    B = build_B()
    meta = _meta(
        "fog-walls", GRID_X, GRID_Y, START_XY, GOAL_XY, (),
        FOG_DRIFT_SPEC, FOG_NOISY_OBSERVATIONS, FOG_CELLS, TOP_ROW_CELLS,
        11, wall_states=FOG_WALLS)
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
def depict(player=None, ax=None, drift_spec=FOG_DRIFT_SPEC,
           sink_states=(), noisy_observations=FOG_NOISY_OBSERVATIONS,
           fog_cells=FOG_CELLS, grid_x=None, grid_y=None, goal_xy=None,
           wall_states=FOG_WALLS):
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
