# Fog-walls: a learn-A environment where SP-EFE beats SP-Pref-Plan

Session of 2026-09-09/10. Companion notes:
[fog maze (sinks)](stochastic_maze_fog_results.md),
[5×5 / mini learn-A](stochastic_maze_learnA_results.md),
[planner corrections](planner_corrections.md),
[curiosity, learn-B and horizon](curiosity_horizon_learnb.md),
[wall with one opening](wall_opening_results.md),
[row-code grid (negative result)](row_code_grid_results.md).

Question: an environment in which the agent has to **explore and learn A**
to do well, such that `SP-EFE*` (in particular `SP-EFE-DeltaB`) beats
`SP-Pref-Plan`.

## Why the previous learn-A environments could not show it

1. `SP-Pref-Plan` is **A-blind in planning** (`μ_obs→x = Σ_y Â(y|x)` is
   constant). Wherever the goal is reachable open-loop from a known start
   with deterministic B, KL wins without touching A (trap learn-A: 60/60 from
   episode 1). Learning A can only matter where blind control fails.
2. Coupled's A-exploration is count-based novelty (2·Δ_A) plus ambiguity
   avoidance (`H[Â(·|x)]`). Under a flat prior every cell is equally novel;
   if the unknown cells are lethal the drive is a magnet into sinks (fog
   learn-A: coupled 2/60, absorbed early, never learns the clean rows).

So the recipe: A-knowledge must change the *route* (ambiguity), KL's route
must be A-independent and worse, and exploration must be non-lethal.

## Environment: `--maze fog-walls`

The [fog maze](stochastic_maze_fog_results.md) with its four sinks
`(2,1),(2,3),(3,1),(3,3)` replaced by **impassable walls** (a move into a
wall is a bump = stay). Everything else is unchanged: start `(0,2)`, goal
`(4,2)`, drift only at `(1,2)` (p = 0.6 north/south), fog block
`(1,1),(1,2),(1,3),(2,2),(3,2)` with observations uniform over the block,
`(2,0)` notebook-noisy, deterministic observation-exact top row (8 steps).

Nothing is lethal any more. Mislocalisation in the fog costs bumps and
detours (from `(1,1)`, EAST is now a bump; recovery is SOUTH → bottom row or
NORTH back into the drift cell), so an explorer that wanders into the fog
keeps walking and keeps learning.

Layout meta: `sink_states=()`, `wall_states=FOG_WALLS`
(`environments/stochastic_maze.py`, `build_B(wall_states=...)`).

## Known A and B (30 seeds)

Terminal-only preference 4, shrinking H, argmax, 100 iters, damping 0.5.
One episode per seed (the drift is the only randomness). SP-EFE-dyn / SP-EFE /
SP-EFE-DeltaB are identical on-policy: B is clamped, so Δ_B is idle and the dyn
channel does not change the first action.

```
uv run python run_stochastic_maze_coupled.py --maze fog-walls --scenario known
uv run python run_stochastic_maze_coupled.py --maze fog-walls --scenario known --horizon 8
```

Raw arrays: [`figures/stochastic_maze_fog_walls_known.npz`](../figures/stochastic_maze_fog_walls_known.npz)
(per-seed `outcome` / `steps` / `top` / `sink` and first-step `q0` for each
planner at H = 11 and H = 8).

### H = 11 (layout default; slack of 3 beyond the 8-step top row)

| planner | goal | top row | steps | first q(N E S W) |
| --- | --- | --- | --- | --- |
| SP-EFE-dyn / SP-EFE / SP-EFE-DeltaB | **30/30** | **30/30** | 10.0 | [0.76, 0.00, 0.04, 0.20] NORTH |
| SP-Pref | 22/30 | 0 | 8.1 | [0.23, **0.30**, 0.23, 0.24] EAST |
| SP-Pref-Plan | 29/30 | 0 | 8.1 | [0.20, **0.43**, 0.20, 0.18] EAST |

Walls are not lethal, so with three spare steps KL can bump after a wrong
landing and still often hit the goal. Coupled still takes the top row every
seed (mean 10 steps: two extra wall-hugs at the corners of the 8-step
route). The planning correction does not switch the KL first action — it
commits EAST more sharply.

### H = 8 (tight: length of the safe route)

| planner | goal | top row | steps | first q(N E S W) |
| --- | --- | --- | --- | --- |
| SP-EFE-dyn / SP-EFE / SP-EFE-DeltaB | **30/30** | **30/30** | **8.0** | [1.00, 0.00, 0.00, 0.00] NORTH |
| SP-Pref | 17/30 | 0 | 6.8 | [0.22, **0.32**, 0.22, 0.24] EAST |
| SP-Pref-Plan | 18/30 | 0 | 6.8 | [0.13, **0.73**, 0.13, 0.00] EAST |

This is the same split as [known-A fog](stochastic_maze_fog_results.md)
(coupled 30/30 top row, KL EAST into the drift). Both KL planners take the
middle route (they plan as if they will know where they land); the ~40 %
that land in `(1,2)` finish in 4–5 steps, the rest need the full 8 or time
out. `q(EAST)` for corr is 0.73 — the correction commits the fog shortcut
rather than restoring the top row. **H = 8 is the discriminating dial** for
learn-A: at H = 11 the KL floor is already ~1.00 by recovery, so learning
A cannot show a gap.

## Learn-A (flat Dirichlet 0.1 on A, B known; 100 eps × 5 seeds, H = 8↓)

```
uv run python run_stochastic_maze_coupled.py --compare --scenario learn-A \
    --maze fog-walls --horizon 8 --episodes 100 --n-seeds 5
```

Raw arrays: [`figures/stochastic_maze_fog_walls_learnA_H8.npz`](../figures/stochastic_maze_fog_walls_learnA_H8.npz).
Figure (mean ± 1 std across the 5 seeds; shaded):
[`figures/stochastic_maze_fog_walls_learnA_H8.png`](../figures/stochastic_maze_fog_walls_learnA_H8.png)
 / [`pdf`](../figures/stochastic_maze_fog_walls_learnA_H8.pdf).
Means ± std across seeds 0–4; `last-20` is wins in episodes 81–100;
`per-seed` is total goals.

| planner | goal | last-20 | top row | errA | 1–10 | 11–20 | 21–30 | 31–60 | 61–80 | 81–100 | per-seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SP-EFE-dyn | 94.4±2.0 | **20.0/20** | 20.2±31.9 | **0.026** | 0.74 | 0.72 | 0.98 | 1.00 | 1.00 | 1.00 | 97,91,94,95,95 |
| SP-EFE | **95.8±0.7** | **20.0/20** | 3.8±0.4 | **0.026** | 0.76 | 0.84 | 0.98 | 1.00 | 1.00 | 1.00 | 95,95,97,96,96 |
| **SP-EFE-DeltaB** | 94.4±2.2 | **20.0/20** | 20.0±32.5 | **0.025** | 0.72 | 0.78 | 0.94 | 1.00 | 1.00 | 1.00 | 95,90,96,96,95 |
| SP-Pref-Plan | 81.4±4.8 | 16.2±0.7/20 | 0 | 0.037 | 0.74 | 0.86 | 0.76 | 0.83 | 0.84 | 0.81 | 77,76,80,87,87 |
| SP-Pref | 47.4±3.1 | 9.6±1.0/20 | 0 | 0.049 | 0.36 | 0.48 | 0.36 | 0.51 | 0.52 | 0.48 | 50,44,46,52,45 |

Every coupled variant is at 1.00 from episode ~31 onward on every seed
(last-20 = 20/20 on all 15 runs). `SP-EFE ≈ SP-EFE-DeltaB` in return (B is
clamped, so Δ_B is idle); they differ in *which* clean row they lock onto.
Seed 0 of SP-EFE-dyn / SP-EFE-DeltaB takes the top row (~84/100); the other four seeds
take the bottom row via `(2,0)`, hence the huge top-row std. SP-EFE
almost never uses the top row (3.8/100) but still hits 1.00. The dyn
channel does not hurt here: the only stochastic transition is the drift
cell the agent visits during the novelty phase, and the bumps are
deterministic.

`SP-Pref-Plan` plateaus at 0.81±0.04 in the last block and `SP-Pref` at
0.48±0.05. They never take the top row. **Learning A changes nothing for
them**: the plan never consults A, and their table error stays 1.5–2×
higher because they only see the middle route. The 3-seed run (77.7 /
46.7) sits inside the 5-seed intervals.

## Learn-B (flat Dirichlet 0.1 on B, A known; 200 eps × 5 seeds, H = 8↓)

A is clamped to the true fog-walls likelihood; B starts flat. Episodes
**do not stop at the goal** (the maze runner's learn-B default: otherwise
the goal columns stay unlearnable). Same preference / shrinking H / argmax
as learn-A; 200 episodes (default learn-B is 20).

```
uv run python run_stochastic_maze_coupled.py --compare --scenario learn-B \
    --maze fog-walls --horizon 8 --episodes 200 --n-seeds 5
```

Raw arrays: [`figures/stochastic_maze_fog_walls_learnB_H8.npz`](../figures/stochastic_maze_fog_walls_learnB_H8.npz).
Figure (mean ± 1 std across the 5 seeds; shaded):
[`figures/stochastic_maze_fog_walls_learnB_H8.png`](../figures/stochastic_maze_fog_walls_learnB_H8.png)
 / [`pdf`](../figures/stochastic_maze_fog_walls_learnB_H8.pdf).
`last-40` is wins in episodes 161–200; `per-seed` is total goals.

| planner | goal | last-40 | top row | errB | 1–10 | 21–30 | 61–80 | 101–140 | 181–200 | per-seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SP-EFE-dyn | 0.4±0.5 | 0.2/40 | 29.2±3.4 | **0.052** | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 1,0,0,1,0 |
| SP-EFE | **0.0±0.0** | 0/40 | 15.2±2.1 | 0.054 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0×5 |
| **SP-EFE-DeltaB** | 0.6±0.5 | 0.2/40 | 15.6±2.2 | **0.052** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1,0,0,1,1 |
| SP-Pref-Plan | 0.0±0.0 | 0/40 | **196.2±1.8** | 0.070 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0×5 |
| SP-Pref | 95.8±79.2 | 24.0±19.6/40 | 14.8±13.9 | 0.063 | 0.00 | 0.20 | 0.60 | 0.60 | 0.60 | 148,0,182,0,149 |

This is the opposite of learn-A. Coupled's Δ_B (and, for SP-EFE, the dyn
channel) never satiates in 200 episodes — errB is still falling at 0.052
vs a flat-prior floor of ~0.07 — so they keep exploring and almost never
arrive (`stop_at_goal=False` also means a lucky visit to `(4,2)` does not
end the episode). That matches the open-grid learn-B note: the dyn
channel blocks exploitation, and even SP-EFE-dyn needs Δ_B to decay below the
preference, which does not happen here in 200 eps.

`SP-Pref-Plan` is a frozen NORTH / top-wall loop: 196/200 episodes stay
inside the top-row cell set, but it never turns SOUTH into the goal
(`(4,4)→(4,3)→(4,2)`). Action-index optimism picks NORTH first; the
planning correction then *commits* to that wall-hug. errB stays at 0.070
(almost the prior): it is not collecting new columns.

`SP-Pref` (no correction) is seed-fragile optimism. Three seeds lock onto
a working policy from ~ep 20–60 and then score 1.00 for the rest of the
run (148, 182, 149/200); two seeds freeze at 0/200. The 0.60 last-block
mean is 3/5, not a 60 % success rate per episode. The huge std band on
the figure is that split.

**Fog-walls is an A-learning benchmark, not a B-learning one.** Known A
already tells coupled which route is unambiguous; unknown B then drowns
that signal in novelty. KL never consults A, so its learn-B behaviour is
the same optimism/commitment pathology as on an open grid
([curiosity, horizon, learn-B](curiosity_horizon_learnb.md)).

### What coupled learns, and when it switches

- Episodes 1–~20: novelty phase. Flat A makes the middle row as attractive
  as any other cell; the agent enters the fog, bumps, wanders, and reaches
  the goal ~70 % of the time (it is not being punished for exploring —
  that is the difference from the sink version).
- The fog columns are learned as "uniform over the five fog symbols" even
  though the agent does not know which fog cell it is in: the aliased cells
  share the same true column, so the smeared Dirichlet credit is harmless.
  The clean-row columns are learned from certain states and become sharp.
- Once `H[Â(·|x)]` on the fog block is ≈ log 5 and ≈ 0 on the clean rows,
  the obs-channel ambiguity term re-routes the plan along a deterministic
  row (top row on seed 0 of SP-EFE-dyn/SP-EFE-DeltaB, bottom row via `(2,0)` on the
  other seeds, hence the top-row std) and Δ_A has decayed enough not to
  pull it back. From then on 1.00 at exactly 8 steps.

## Controls

- `--scenario known` (tables above): the environment is solvable, coupled
  1.00 from a perfect model at both H = 8 and H = 11; the learn-A curve
  is the price of learning that model.
- KL known vs learn-A at H = 8: same policy (EAST, never the top row);
  goal rate is the luck of the drift, not of A.
- `--scenario learn-B`: coupled explores B and does not exploit (0–1
  goals / 200); `SP-Pref-Plan` freezes on the top wall (0/200); `SP-Pref`
  is a 3/5 optimism lottery. Not a coupled > corr result.
- Sink version ([fog](stochastic_maze_fog_results.md)): same tables, lethal
  cells → coupled 2/60. The only change between the two results is that
  exploration is survivable.

## Reproduction

```
uv run python run_stochastic_maze_coupled.py --maze fog-walls --scenario known
uv run python run_stochastic_maze_coupled.py --maze fog-walls --scenario known --horizon 8
uv run python run_stochastic_maze_coupled.py --compare --scenario learn-A --maze fog-walls --horizon 8 --episodes 100 --n-seeds 5
uv run python run_stochastic_maze_coupled.py --compare --scenario learn-B --maze fog-walls --horizon 8 --episodes 200 --n-seeds 5
```

Regression: `test_fog_walls_layout` in
[`tests/test_stochastic_maze.py`](../tests/test_stochastic_maze.py).
