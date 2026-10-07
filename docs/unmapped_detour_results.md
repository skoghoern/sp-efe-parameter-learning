# Unmapped detour: a learn-B environment where SP-EFE beats SP-Pref-Plan

Session of 2026-09-10. Companion notes:
[curiosity, learn-B and horizon](curiosity_horizon_learnb.md) (§5.4 is the
prototype of this layout),
[wall with one opening](wall_opening_results.md),
[fog-walls learn-A](fog_walls_learnA_results.md) (the A-side dual),
[DeepSea learn-B](deep_sea_results.md),
[planner corrections](planner_corrections.md).

## Significant results

![Default and mirror layouts](../figures/unmapped_detour_layouts.png)

The environment is a 5×4 grid with two routes to an absorbing goal. The
mapped shortcut `→→→→↑` ends at a **slippery ice** cell (crystal mark):
NORTH reaches the goal with probability \(p=0.5\) and otherwise **slips**
into an absorbing dead end. The **safe detour** `↑↑→→→→↓` is deterministic
and runs through unmapped transitions. Identity A, terminal-only
preference, shrinking \(H=7\), \(T=8\). Learning \(B\) is the only way to
beat \(p\).

There are two findings, of increasing strength.

### 1. Coupled learns the detour; corrected KL never tries (cells layout)

Unknown region = six whole cells, 24 flat Dirichlet columns. Learn-B,
200 episodes × 5 seeds
([figure](../figures/unmapped_detour_p0.50_learn-B_H7.png)):


| planner         | what it does                                                      | goal rate   | last 20 eps |
| --------------- | ----------------------------------------------------------------- | ----------- | ----------- |
| `SP-EFE-dyn` | explores by Δ_B, satiates ~ep 75, then takes the detour forever   | **159/200** | **1.00**    |
| `SP-Pref-Plan`   | never leaves the mapped shortcut (0 unmapped columns, B̂ = prior) | 98/200 ≈ p  | 0.50        |
| `EFE`  | enumerated EFE, same curve as `SP-EFE-dyn`                     | 65/100      | 1.00        |


This is the cleanest learn-B separation in the repo against `SP-Pref-Plan`.
The three controls hold at once:

- **known B** — coupled and `SP-Pref-Plan` both 1.00 via the detour
(learning is *sufficient*);
- **frozen B** — nobody beats p; `SP-Pref-Plan` still gambles and still
never touches an unknown column (learning is *necessary*);
- **learn-B** — only Δ_B collects the data.

`SP-EFE` (dyn channel on) covers every column but under-exploits
(35/200): satiation is blocked. `SP-EFE-DeltaB` sits in between (115/200,
still climbing).

**This layout does not beat uncorrected** `SP-Pref`**.** That planner scores
185/200 (985/1000 on a long run) by a different mechanism: its Bethe
action-entropy bonus is maximal wherever all four actions look alike, and
a whole unknown cell is exactly that. The bonus pulls it into the flat
region; the columns it crosses sharpen; the now-known detour beats the
gamble. It then **stops learning** (ident frozen at the 7 on-path columns).
So it wins the return with the same term that makes it *gamble under a
perfect model* (known-B table: 10/20 on the ice). The planning correction
removes that term — and with it every visit to the unmapped region.

### 2. Coupled beats both KL planners (doors layout)

The lever against `SP-Pref` is *what* is unknown, not where. `--unmapped doors` flattens only the six detour actions themselves; at each detour
cell the other three actions are known (bump or back). Same grid, 6
columns instead of 24. Learn-B, 300 episodes × 3 seeds
([figure](../figures/unmapped_detour_p0.50_doors_learn-B_H7.png)):


| planner                              | goal              | detour         | then what                                                                                                                      |
| ------------------------------------ | ----------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `SP-EFE-dyn` / `SP-EFE` / `SP-EFE-DeltaB` | **298–300 / 300** | all of them    | detour from ep 1–3, never leaves                                                                                               |
| `SP-Pref`                             | 158 ± 8           | **25, then 0** | walks the detour while doors are smeared; once they are identified, switches to the ice and gambles for the remaining 275 eps |
| `SP-Pref-Plan`                        | 146 ± 7 ≈ p       | **0**          | never opens a door (B̂ = prior)                                                                                                |


The number that matters is the **detour count**, not the goal count.
`SP-Pref`'s 158 goals are almost all coin-flip wins on the shortcut;
`SP-EFE*`'s 300 are all the safe route. After episode 25, `SP-Pref` is
statistically indistinguishable from `SP-Pref-Plan`.

Why this is stronger than result 1:

- **Learning makes** `SP-Pref` **worse.** While the doors are uncertain they
pay its entropy bonus, so it follows them. Once they are sharp the
detour is entropy-free and it does what it does under known B: gamble.
A better model removes the only term that was steering it to the safe
route.
- `SP-EFE-dyn` **is monotone in model quality.** Frozen B → 0 (farms
the first door because Δ_B never decays); learned B → 1.00. That is
the behaviour the epistemic term is supposed to produce.
- Enumerated `EFE` is 100/100 via the detour, so the coupled
result is the EFE objective, not loopy BP. Enumerated `EFE-noamb-noig` also
stays on the detour (terminal-state entropy, not the Bethe action
bonus), which isolates the gamble as a *loopy* KL pathology.

Caveat on the dyn-channel variants (`SP-EFE`, `SP-EFE-DeltaB`): with
*frozen* doors they also score 1.00, because "take the most uncertain
column" and "take the detour" are the same action at every cell. Their
learn-B 1.00 is therefore not evidence that they needed to learn. Only
`SP-EFE-dyn` fails without learning and succeeds with it.

Robust to `--action-order WSEN` (relabelling does not change the ranking)
and to `--layout mirror` (WEST+SOUTH detour, 7 doors, H=8: coupled 299–300
via the detour; `SP-Pref` walks the doors for 101 episodes then still
abandons them; `SP-Pref-Plan` never opens a door). Geometry alone — mirror
layout with whole unknown cells — does **not** beat `SP-Pref`: the bonus
plus optimism is a frontier heuristic that consumes every flat column
regardless of index order.

### What this does not show

- It does not show that `SP-EFE` (full AIF-MP, dyn channel on) is the
right learn-B planner on large unknown regions: on the cells layout it
explores best and exploits worst. The doors layout is small enough that
even the dyn channel satiates.
- It does not show that KL-control as an *objective* refuses the detour:
enumerated `EFE-noamb-noig` takes it. The failure is the loopy
implementation (`SP-Pref` / `SP-Pref-Plan`).
- The noisy-TV cell separates parameter curiosity from entropy-seeking
only under **known** B. Under learning nobody visits it.

Question originally asked: an environment in which the agent must
**explore and learn B** to do well, such that SP-EFE beats
`SP-Pref` and `SP-Pref-Plan`. Result 1 answers for `SP-Pref-Plan`; result 2 answers for both.

## Why earlier learn-B layouts did not discriminate

- `SP-Pref` and `SP-Pref-Plan` have no epistemic term for B. They move through
**posterior-mean optimism** only: a flat Dirichlet column looks like a
1/N teleport, so unknown actions are attractive exactly when every known
option is worse. Each sweeps its frontier in action-index order and, once
the planning correction finds any loop that satisfies the terminal
preference, freezes. On open grids / wall-opening / DeepSea this gave
frozen loops, deadline-parking or lucky index-order sweeps — none of
them a controlled measurement.
- `SP-EFE*` explores through Δ_B (count-based novelty over *all* unknown
columns) and has to **satiate** before it exploits. Satiation fails when
the unknown region is large (fog-walls learn-B: 100 columns, 0/200), when
the dyn channel is on, or when unknown cells are lethal (novelty magnet).



## Design

The B-analogue of fog-walls: B-knowledge must change the *route*, KL must
have a known route that is mediocre but good enough to switch optimism
off, and exploring the unknown part must be cheap and survivable.

1. **Partially mapped B.** Most columns are clamped; a small connected
  region has flat-prior columns. Removes the dependence on index-order
   sweeps.
2. **Known route with success probability p<1 and absorbing failure.**
  Caps KL at p per episode and rules out retry-with-slack and
   deadline-parking. Under terminal-only preference the two routes differ
   in success *probability*, which is the one quantity the objective sees
   (a shorter deterministic route would be a tie).
3. **Deterministic route inside the unmapped region.** With B known every
  planner should take it (ceiling); with B frozen at the prior nobody can
   (floor).
4. **Nothing lethal among the unknowns; ice cell known.** §5.4 of the
  curiosity notes showed that an unknown trap flips the result.
5. **Optimism cannot rival p.** A flat column is worth
  1-(1-1/N)^{H-1} to a posterior-mean planner (uniform-teleport walk
   hitting G). N=20, H=7: 0.265. Keep p clearly above it.
6. **Small unknown region** (24 columns at prior 0.1) so Δ_B satiates in
  tens of episodes.
7. **H = detour length, shrinking.** Default H=7 so enumerated
  `EFE` / `EFE-noamb-noig` (4^7=16384) are cheap; mirror H=8
  (4^8=65536, ~12 min / 100 eps) is still enumerable.
8. Goal absorbing and known; the final transition is folded into the
  counts. Terminal-only preference 4.



## Environment: `run_unmapped_detour.py`

[`environments/unmapped_detour.py`](../environments/unmapped_detour.py).
Identity A, `s = x + y·grid_x`, action order N/E/S/W. Figures:
[`unmapped_detour_layout_default.png`](../figures/unmapped_detour_layout_default.png),
[`unmapped_detour_layout_mirror.png`](../figures/unmapped_detour_layout_mirror.png),
combined panel
[`unmapped_detour_layouts.png`](../figures/unmapped_detour_layouts.png)
(`uv run python run_unmapped_detour.py --depict`).

![Default layout](../figures/unmapped_detour_layout_default.png)

**Default** (5×4). The icy shortcut `→→→→↑` is 5 steps and succeeds w.p.
`gate_p = 0.5` from the slippery-ice cell (crystal); otherwise it slips
into the dead end. The detour `↑↑→→→→↓` is 7 steps, deterministic, and
runs through the six stippled unmapped cells (24 flat columns; none
adjacent to the ice or the dead end). \(N=20\).

![Mirror layout](../figures/unmapped_detour_layout_mirror.png)

**Mirror** (`--layout mirror`, 6×4, no pocket). Goal at `(0,2)` (figure
row 1, column 0); the vacated top-left cell `(0,3)` is unmapped. Detour
`↑↑←←←←←↓` (8 steps, H=8). The ice cell is adjacent to G (NORTH, w.p.
`gate_p`). \(N=24\).

Default: H=7↓, T=8. Mirror: H=8↓, T=9. Argmax, 100 iters, damping 0.5;
episodes end at G or the dead end. Layout invariants (detour is
unmapped-only; unmapped cells never reach the dead end or the ice) are
checked in the constructor.

Scenarios: `known` (all columns clamped), `frozen` (unmapped columns flat,
never updated), `learn-B` (counts persist). Planners: five MP planners,
`EFE` / `EFE-noamb-noig`, `Random` (uniform actions, learning on).
Table column *gate* is the icy shortcut (code still uses `gate_p`).

```
uv run python run_unmapped_detour.py --depict
uv run python run_unmapped_detour.py --compare
uv run python run_unmapped_detour.py --scenario learn-B --episodes 100 --n-seeds 3 \
    --planners EFE EFE-noamb-noig SP-EFE-dyn SP-EFE-DeltaB SP-Pref-Plan
uv run python run_unmapped_detour.py --scenario learn-B --episodes 1000 --n-seeds 1 \
    --planners SP-Pref SP-EFE-dyn
```



## Known B — ceiling (20 eps, seed 0)

Raw: [`figures/unmapped_detour_p0.50_known_H7.npz`](../figures/unmapped_detour_p0.50_known_H7.npz)
/ [png](../figures/unmapped_detour_p0.50_known_H7.png).


| planner                      | goal      | detour | gate | steps | q0 (N E S W)                 |
| ---------------------------- | --------- | ------ | ---- | ----- | ---------------------------- |
| SP-EFE-dyn / SP-EFE / SP-EFE-DeltaB | **20/20** | 20     | 0    | 7.0   | [1.00, 0, 0, 0]              |
| SP-Pref-Plan                  | **20/20** | 20     | 0    | 7.0   | [1.00, 0, 0, 0]              |
| SP-Pref                       | 10/20     | 0      | 20   | 5.0   | [0.23, **0.30**, 0.23, 0.24] |
| Random                       | 0/20      | 0      | 0    | —     | —                            |


The task is solvable and the detour is the correct plan: every coupled
variant and the corrected KL take it at exactly 7 steps. Uncorrected
`SP-Pref` gambles (10/20 = p) — the soft-min action-entropy bonus at the
absorbing dead end / ice, the trap-maze pathology of
[planner corrections §2](planner_corrections.md).

## Frozen B — floor (20 eps, seed 0)

Raw: [`figures/unmapped_detour_p0.50_frozen_H7.npz`](../figures/unmapped_detour_p0.50_frozen_H7.npz).


| planner                      | goal      | gate | unmapped cols visited | note                                                  |
| ---------------------------- | --------- | ---- | --------------------- | ----------------------------------------------------- |
| SP-Pref-Plan                  | **10/20** | 20   | 0                     | exactly p; q0 EAST 0.70                               |
| SP-EFE-dyn / SP-EFE / SP-EFE-DeltaB | 0/20      | 0    | 2                     | park in the first unmapped cell (Δ_B can never decay) |
| SP-Pref                       | 0/20      | 0    | 2                     | same: flat columns maximise its entropy bonus         |
| Random                       | 0/20      | 0    | 11                    | —                                                     |


Without learning nobody beats the gamble. `SP-Pref-Plan` never touches an
unmapped column, confirming requirement 5: the flat region is worth less
than the ice to optimism.

## Learn-B — the benchmark (200 eps × 5 seeds, H≤7↓, T=8, prior 0.1)

Raw: [`figures/unmapped_detour_p0.50_learn-B_H7.npz`](../figures/unmapped_detour_p0.50_learn-B_H7.npz).
Figure (goal rate, detour rate, unmapped-column error, coverage; mean ± std
over seeds): [`figures/unmapped_detour_p0.50_learn-B_H7.png`](../figures/unmapped_detour_p0.50_learn-B_H7.png)
/ [pdf](../figures/unmapped_detour_p0.50_learn-B_H7.pdf).

Planners that never enter the ice see a deterministic environment, so
their five seeds coincide (std 0). `detour` = goal reached with every
visited cell in {S, unmapped, G}; `cols` = unmapped (s,u) columns visited
(of 24); `ident` = unmapped columns whose MAP landing is right with mass
≥ 0.8; `errB` = mean |B̂ − B| on the unmapped columns (flat-prior value
0.095).


| planner           | goal          | last-20   | detour  | gate    | dead end | steps | cols  | ident | errB      | 40-ep blocks                  |
| ----------------- | ------------- | --------- | ------- | ------- | -------- | ----- | ----- | ----- | --------- | ----------------------------- |
| **SP-EFE-dyn** | **159**       | **20/20** | **159** | 0       | 0        | 7.9   | 23    | 19    | 0.016     | .30 .78 **1.00** .90 **1.00** |
| SP-EFE        | 35            | 8/20      | 35      | 0       | 0        | 8.0   | 24    | 24    | **0.005** | .03 .15 .17 .23 .30           |
| SP-EFE-DeltaB   | 115           | 17/20     | 115     | 0       | 0        | 8.0   | 23    | 20    | 0.012     | .20 .55 .68 .70 .75           |
| SP-Pref            | 185           | 20/20     | 185     | 0       | 0        | 7.0   | 23    | 7     | 0.031     | .62 1.00 1.00 1.00 1.00       |
| **SP-Pref-Plan**   | **97.6±12.5** | 10.0±1.7  | **0**   | **200** | 102±13   | 5.0   | **0** | 0     | **0.095** | .46 .52 .47 .49 .49           |
| Random            | 0.4±0.5       | 0         | 0       | 5.6     | 6.0      | —     | 17.4  | 10.6  | 0.043     | .00 .01 .01 .00 .00           |




### SP-EFE-dyn vs SP-Pref-Plan: the intended result

- `SP-Pref-Plan` gambles on **every one of 1000 episodes** (5 seeds × 200),
sits at 0.49 = p, visits **zero** unmapped columns and ends with B̂
bit-identical to the prior. Learning is available and it never collects
a single sample where it matters.
- `SP-EFE-dyn` enters the unmapped region on episode 1 (q0 NORTH 1.00),
covers 21/24 columns within 10 episodes and 23/24 by 20, reaches the
goal for the first time at ep 17, and is at 1.00 from a 20-episode
perfect streak starting at **ep 75**. All 159 goals are the detour;
no ice visit, no dead end. Mean 7.9 steps (7 = optimal; residual
wander early on).
- The three ingredients hold at once: with known B every planner is at
1.00 (learning is *sufficient*), with frozen B nobody beats p
(learning is *necessary*), and only Δ_B-driven exploration collects the
data.



### The other coupled variants

- `SP-EFE` (dyn channel on): explores best (24/24 columns identified,
errB 0.005, lowest of all) and exploits worst (35/200, 0.30 in the last
block). Same mechanism as the open-grid / wall-opening notes: the
support-uniform dyn kernel blocks Δ_B satiation.
- `SP-EFE-DeltaB` (dyn on, Δ_B off): 115/200, 0.75 and still climbing.
Spreading through the dyn channel explores, but commits slowly.



### Uncorrected SP-Pref: solved by the pathology

`SP-Pref` reaches the detour first (streak from ep 23) and locks it in
(1000-episode run: 985/1000, last 900 episodes perfect; raw
[`…_1k_learn-B.npz`](../figures/unmapped_detour_p0.50_learnB_H7_1k_learn-B.npz)).
This is not B-curiosity. Its soft-min bonus rewards states where all
actions look alike — absorbing cells with known B (hence the gamble in
the known-B table) and *flat columns* with unknown B (q0 NORTH 0.48 on
the prior). The bonus drags it into the unmapped region, the columns it
crosses sharpen, and the now-known deterministic route beats the 0.5
gamble. It then stops learning: `ident` is frozen at 7 (the seven on-path
columns) and errB at 0.031 from ep 50 to ep 1000, while `SP-EFE-dyn`
identifies 19 and reaches 0.015. So on this layout `SP-Pref` wins the
return with the same term that makes it lose under a perfect model; the
planning correction removes the term and with it every visit to the
unmapped region.

### Analytic reference (100 eps × 3 seeds)

Raw: [`…_analytic_learn-B.npz`](../figures/unmapped_detour_p0.50_learnB_H7_analytic_learn-B.npz)
/ [png](../figures/unmapped_detour_p0.50_learnB_H7_analytic_learn-B.png).


| planner         | goal | last-10 | detour | cols | ident | errB  | 10-ep blocks                                |
| --------------- | ---- | ------- | ------ | ---- | ----- | ----- | ------------------------------------------- |
| EFE    | 65   | 10/10   | 65     | 24   | 18    | 0.015 | .00 .30 .60 .50 .80 1.00 .40 .90 1.00 1.00  |
| SP-EFE-dyn   | 63   | 10/10   | 63     | 23   | 18    | 0.018 | .00 .10 .70 .40 .70 .70 .90 .80 1.00 1.00   |
| EFE-noamb-noig     | 79   | 10/10   | 79     | 23   | 11    | 0.025 | .10 .50 .70 .90 .90 .80 1.00 1.00 1.00 1.00 |
| SP-EFE-DeltaB | 41   | 5/10    | 41     | 23   | 18    | 0.018 | .00 .00 .30 .50 .50 .50 .60 .60 .60 .50     |
| SP-Pref-Plan     | 52±8 | 3.3/10  | 0      | 0    | 0     | 0.095 | ≈ 0.5 throughout                            |


Enumerated state-space EFE tracks `SP-EFE-dyn` almost exactly (65 vs
63, same coverage / ident / errB): the exploration–then–exploitation curve
is the objective, not loopy BP. `EFE-noamb-noig` also finds the detour: the
state-space KL objective keeps \(\mathbb{E}[\log \hat B]\), i.e. a
predictive-entropy reward on the transition, which a flat column
maximises — the enumerated version of what pulls uncorrected `SP-Pref` in.

## Noisy-TV variant: separating parameter curiosity from entropy seeking

`--noisy-tv` turns the mapped cell T = (2, 0) into a known scrambler:
every action lands uniformly on the eight mapped free cells. Its column
is clamped, so Δ_B is zero there, but its predictive entropy (log 8 ≈ 2.1
nats) never decays. A planner that seeks *parameter* information ignores
T; a planner that seeks *state* entropy (uncorrected KL, the dyn channel)
should be drawn to it.

Known B, 10 eps (`--noisy-tv --scenario known`):


| planner                     | goal  | detour | gate | TV visits |
| --------------------------- | ----- | ------ | ---- | --------- |
| SP-EFE-dyn, SP-Pref-Plan  | 10/10 | 10     | 0    | 0         |
| SP-EFE, SP-EFE-DeltaB | 2/10  | 0      | 4    | **10**    |
| SP-Pref                      | 3/10  | 0      | 10   | 0         |


The dyn channel goes to the TV on every episode with a perfect model
(the slippery-grid result of [planner corrections §3](planner_corrections.md#3-the-dyn-channel-rxxu--double-counting-and-the-ablation)
reproduced in one cell). Uncorrected `SP-Pref` still prefers the ice:
its bonus is action entropy, which the absorbing dead end maximises just as well.

Learn-B, 200 eps × 3 seeds (`--noisy-tv --scenario learn-B`; raw
[`…_tv_learn-B_H7.npz`](../figures/unmapped_detour_p0.50_tv_learn-B_H7.npz)
/ [png](../figures/unmapped_detour_p0.50_tv_learn-B_H7.png)):


| planner         | goal   | last-20 | detour | gate | TV  | ident | errB  |
| --------------- | ------ | ------- | ------ | ---- | --- | ----- | ----- |
| SP-EFE-dyn   | 159    | 20/20   | 159    | 0    | 0   | 19    | 0.017 |
| SP-EFE      | 34     | 6/20    | 34     | 0    | 0   | 24    | 0.005 |
| SP-EFE-DeltaB | 108    | 15/20   | 108    | 0    | 0   | 20    | 0.014 |
| SP-Pref          | 185    | 20/20   | 185    | 0    | 0   | 8     | 0.031 |
| SP-Pref-Plan     | 104±11 | 11/20   | 0      | 200  | 0   | 0     | 0.095 |
| Random          | 1.0    | 0       | 0      | 5    | 50  | 10    | 0.045 |


**Negative result for the TV as a discriminator under learning.** No
planner visits T in learn-B: on the prior a flat column (log 20 ≈ 3.0
nats plus Δ_B) outbids the scrambler (2.1 nats), so every explorer goes
NORTH first, and once the detour is found the goal preference keeps it
there. The learn-B numbers are the default layout's within a few goals.
`SP-Pref`'s lock-in survives because its bonus is *action* entropy: at each
detour cell the three off-path columns stay smeared forever (ident 8 =
the on-path columns only), so every detour cell keeps paying the bonus
that absorbing dead end pays in the known-B table. A permanent entropy source
would have to be *absorbing* to compete with that — and an absorbing
non-goal cell is the dead end, which requirement 4 keeps away from the
unknowns. The TV variant therefore stays as a known-B demonstration of the
dyn-channel attraction; the coupled-vs-`SP-Pref-Plan` result is unchanged
by it.

## Reading

- The environment does what the design asked: B-knowledge changes the
route, the known gamble switches optimism off, the unknowns are
survivable, and Δ_B satiates in ~75 episodes. `SP-EFE-dyn` 1.00 vs
`SP-Pref-Plan` 0.49 with a bit-identical B̂ is the cleanest learn-B
separation in the repo, and enumerated EFE reproduces it.
- Against *uncorrected* `SP-Pref` the default layout does not separate:
its entropy bonus is a (badly calibrated) exploration drive that happens
to point into the flat region. The known-B and frozen tables show why it
is not a coherent objective — it gambles with a perfect model and parks
with a frozen one. A non-absorbing entropy source (the TV variant) does
not dislodge it under learning; only the planning correction does, and
that removes the exploration with it.
- Dials: `gate_p` sets KL's ceiling (keep it above the optimism value
printed by the runner), the number of unmapped columns sets coupled's
satiation time, H = detour length sets the commitment requirement.



## Doors variant: SP-EFE beats both KL planners

Goal: make `SP-EFE*` superior to **both** `SP-Pref-Plan` and `SP-Pref` in
learn-B. Two ideas were tried; the geometric one failed and is kept as a
negative result, the model-structure one works.

### What did not work: geometry (`--layout mirror`)

The 6×4 mirrored layout puts the detour against action-index order (at
`(5,3)` NORTH and EAST bump, only WEST progresses). Goal is at `(0,2)`;
the top-left cell is unmapped. No decoy pocket.

![Mirror layout](../figures/unmapped_detour_layout_mirror.png)

Icy shortcut `←←←←←↑` (6, \(p=0.5\)); detour `↑↑←←←←←↓` (8). A two-cell
decoy pocket next to S was tried earlier and did not help (`SP-Pref`
consumed it and still locked the detour), so it is omitted.

Known / frozen behave as the default layout (coupled and `SP-Pref-Plan`
1.00 via the detour; `SP-Pref` gambles 0.50; frozen: only `SP-Pref-Plan`
scores, on the ice). Learn-B, 150 eps, seed 0, whole-cell flat columns:


| planner       | goal    | detour | cols | ident | 15-ep blocks                                  |
| ------------- | ------- | ------ | ---- | ----- | --------------------------------------------- |
| SP-Pref        | **120** | 120    | 32   | 10    | .00 .27 .80 1.00 .93 1.00 1.00 1.00 1.00 1.00 |
| SP-EFE-dyn | 72      | 72     | 32   | 27    | .00 .00 .13 .40 .27 .87 .53 .80 .93 .87       |
| SP-Pref-Plan   | 83      | 0      | 0    | 0     | ≈ 0.5                                         |


`SP-Pref` is not a pass-through walker that parks on the first self-loop:
it visited all 32 columns. A bumped column becomes a *known* self-loop and
loses its bonus, so the remaining flat columns win the next tie — the
bonus plus optimism is a frontier heuristic that consumes flat columns,
and against-index-order moves do not stop it. A known non-lethal absorbing
"rest" cell (`--layout mirror-rest`, R two mapped steps from S) does not
lure it either under known B (rest 0/6, ice 6/6): the absorbing
attraction needs a goal chance attached (the ice), a pure absorber is
never preferred to the goal. So neither geometry nor a permanent entropy
source dislodges `SP-Pref` from a region of whole flat cells.

### What works: unknown *doors*, not unknown cells (`--unmapped doors`)

The two `SP-Pref` results above fix the lever. With **known B** it gambles
(the detour has no entropy, so the bonus favours the absorbing gamble);
with **learned B** it keeps the detour only because the three *off-path*
columns of every detour cell stay smeared forever and pay the bonus at
each step (ident stuck at the on-path columns). Remove the smear: in
`doors` mode only the detour's own \((\mathrm{cell}, \mathrm{action})\) pairs
are flat — the agent knows that the other three actions at a detour cell
bump or lead back, it does not know where the door leads.

Same geometry as the default figure above — only which \(B\)-columns start
unknown changes. Six flat columns:
`(0,2)↑ (0,3)→ (1,3)→ (2,3)→ (3,3)→ (4,3)↓`; all other columns of those
cells are clamped (bumps / back). Icy shortcut `→→→→↑` (5, \(p=0.5\));
detour `↑↑→→→→↓` (7).

![Default layout (doors mode uses this grid)](../figures/unmapped_detour_layout_default.png)

Same D, C, T = 8, H ≤ 7↓, prior 0.1, 100 iterations, damping 0.5. Raw:
[`…_doors_{known,frozen,learn-B}_H7.npz`](../figures/unmapped_detour_p0.50_doors_learn-B_H7.npz)
/ [png](../figures/unmapped_detour_p0.50_doors_learn-B_H7.png).

**Learn-B, 300 eps × 3 seeds** (the benchmark):


| planner         | goal      | last-30 | detour | gate | dead end | cols  | ident | errB   | 30-ep blocks                                     |
| --------------- | --------- | ------- | ------ | ---- | -------- | ----- | ----- | ------ | ------------------------------------------------ |
| SP-EFE-dyn   | **298**   | 30/30   | 298    | 0    | 0        | 6/6   | 6     | 0.0006 | .93 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |
| SP-EFE      | **300**   | 30/30   | 300    | 0    | 0        | 6/6   | 6     | 0.0006 | 1.00 …                                           |
| SP-EFE-DeltaB | **300**   | 30/30   | 300    | 0    | 0        | 6/6   | 6     | 0.0006 | 1.00 …                                           |
| SP-Pref          | 158 ± 8   | 13/30   | 25     | 275  | 142      | 6/6   | 6     | 0.0070 | .91 .59 .48 .49 .57 .49 .50 .41 .41 .43          |
| SP-Pref-Plan     | 146 ± 7   | 12/30   | 0      | 300  | 154      | 0/6   | 0     | 0.0950 | .53 .56 .53 .48 .54 .46 .56 .37 .44 .40          |
| Random          | 1.3 ± 0.5 | 0       | 0      | 9    | 9        | 4.7/6 | 3     | 0.036  | 0                                                |


- `SP-EFE-dyn` takes the detour from episode 3 and never leaves it
(298/300, the two misses are the first two episodes); `SP-EFE` and
`SP-EFE-DeltaB` from episode 1. With only six flat columns Δ_B
satiates within a handful of visits, and every visit is a step along
the route.
- `SP-Pref` walks the detour for exactly its first 25 episodes on every
seed (bit-identical, no stochastic element touched), while the doors are
smeared enough to pay its bonus. Once all six doors are identified
(ident 6, errB 0.007) the detour is entropy-free, and it does what it
does under known B: switches to the ice and gambles for the remaining
275 episodes (0.5 in the last blocks). **Learning makes it worse** — a
better model removes the only term that was steering it to the safe
route.
- `SP-Pref-Plan` never touches a door (cols 0/6, B̂ unchanged) and gambles
throughout: the optimism value of a single flat door (1/20 teleport,
0.27 over the horizon) is below `gate_p`.

Both KL planners sit at the ice ceiling with a 100-episode-scale
stochastic wobble; every coupled variant is at 1.00. Random reaches the
goal once per 300 episodes and identifies 3/6 doors.

**Known B (ceiling, 300 eps × 3 seeds):** SP-EFE-dyn / SP-EFE / SP-EFE-DeltaB
and `SP-Pref-Plan` 300/300 via the 7-step detour; `SP-Pref` 146 ± 7 on the
ice (0/300 detours); Random 1.3. Identical to the cells layout — known B
is the same tensor.

**Frozen B (floor, 300 eps × 3 seeds):** `SP-Pref-Plan` 146 ± 7 (ice);
`SP-EFE-dyn` 0; Random 1.3. But `SP-EFE`, `SP-EFE-DeltaB` **and**
`SP-Pref` score 300/300 via the detour *without learning anything*
(ident 0, errB 0.095 unchanged). In doors mode "take the most uncertain
column" and "take the detour" are the same action at every cell, so any
entropy follower (dyn channel or action-entropy bonus) is guided along the
route by the frozen flat columns, one door at a time. That is the
confound to keep in mind when reading the learn-B table for `SP-EFE`
/ `SP-EFE-DeltaB`: they would win frozen too. `SP-EFE-dyn` under frozen B
instead farms the first door (path `↑→↑↓↑↓↑→`, cols 2/6): with counts that
never move, Δ_B never decays, and the door whose Δ_B is reliably
reachable (known SOUTH brings it back) beats a chain of doors each landing
uniformly — the correct behaviour under a model that cannot update, and
exactly what learning fixes.

Uncorrected `SP-Pref` therefore has *two* regimes on this map, and both
are wrong for a different reason: with smeared doors it follows entropy
(frozen 1.00, learn-B episodes 1–25); with sharp doors it gambles
(known 0.49, learn-B episodes 26–300). Only Δ_B-driven `SP-EFE-dyn`
is monotone in model quality: 0 frozen → 1.00 learned.

### Robustness


| variant                                        | SP-EFE-dyn               | SP-Pref                                                                      | SP-Pref-Plan | raw                                                                                             |
| ---------------------------------------------- | --------------------------- | --------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| doors, action order **WSEN** (200 eps, seed 0) | **198** (1.00 from block 2) | 101 (25 detours, then ice)                                                 | 89          | [`…_doors_WSEN_learn-B_H7.npz`](../figures/unmapped_detour_p0.50_doors_WSEN_learn-B_H7.npz)     |
| doors, **mirror** layout (300 eps × 3 seeds)   | **298** (SP-EFE / SP-EFE-DeltaB 300)  | 248 ± 1 (194 detours, then ice: 1.00 for six blocks, then .77 .51 .50 .49) | 150 ± 5     | [`…_mirror_doors_learn-B_H7.npz`](../figures/unmapped_detour_p0.50_mirror_doors_learn-B_H7.npz) |


Relabelling the action indices leaves every number of the headline table
within seed noise (`SP-Pref`'s 25 detour episodes are reproduced exactly),
so nothing rests on tie-break luck. The H7 mirror row is the earlier
six-door WEST map; the current 6×4 (G at `(0,2)`, 7 doors, last step
SOUTH, H=8↓, T=9) is the table below (300 eps × 3 seeds,
[figure](../figures/unmapped_detour_p0.50_mirror_doors_learn-B_H8.png)):


| planner         | goal          | detour | then what                                                                                          | raw                                                                                                                           |
| --------------- | ------------- | ------ | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| SP-EFE-dyn   | **299**       | 299    | miss is ep 1; 1.00 from block 2                                                                    | [`…_mirror_doors_learn-B_H8.npz`](../figures/unmapped_detour_p0.50_mirror_doors_learn-B_H8.npz)                                 |
| SP-EFE / `SP-EFE-DeltaB` | **300** | 300    | detour from ep 1                                                                                   | same                                                                                                                          |
| SP-Pref          | 200.3 ± 5.9   | **101, then 0** | identifies all 7 doors then switches to the ice (blocks `1.00 1.00 1.00 0.76 0.51 …`) | same                                                                                                                          |
| SP-Pref-Plan     | 149.7 ± 5.2 ≈ p | **0**  | never opens a door (cols 0/7, B̂ = prior)                                                         | same                                                                                                                          |
| EFE    | **100/100**   | 100    | detour from ep 1 (q0 NORTH 0.69); 100 eps, seed 0                                                  | [`…_mirror_doors_learnB_H8_analytic_learn-B.npz`](../figures/unmapped_detour_p0.50_mirror_doors_learnB_H8_analytic_learn-B.npz) |


WEST corridor plus a SOUTH last door (index order tries EAST before SOUTH
at `(0,3)`) does not save `SP-Pref`: it still has the two-regime pathology,
only the smear window is 101 episodes instead of 25. `EFE` tracks
coupled, so this is still the EFE objective. The coupled variants are
unaffected.

### Analytic reference (doors, learn-B, 100 eps, seed 0)

`EFE` **100/100** and `EFE-noamb-noig` **100/100**, both via the
detour from episode 1 (q0 NORTH 0.71 / 0.65; raw
[`…_doors_learnB_H7_analytic_learn-B.npz`](../figures/unmapped_detour_p0.50_doors_learnB_H7_analytic_learn-B.npz)).
Enumerated EFE confirms the coupled result. Enumerated KL control also
takes the door route, for its own reason: \(\mathrm{KL}[q(x_T)\,\|\,p]\)
contains \(-H[q(x_T)]\), and a terminal belief smeared by six flat doors
(KL ≈ 1.1 nats) scores better than the ice's half-mass on the dead end (≈ 1.6) —
the exact-enumeration counterpart of `SP-Pref`'s bonus, at the level of
states rather than actions. Once the doors are sharp the deterministic
detour (0.3 nats) beats the ice outright, so, lacking a Bethe action-
entropy term, `EFE-noamb-noig` has no second regime and stays. The gamble
of `SP-Pref` is therefore specific to the loopy KL planner, not to KL
control as an objective.

### Reading

- The lever against `SP-Pref` is *what* is unknown, not *where*. Whole
unknown cells hand its action-entropy bonus a permanent source at every
detour cell; unknown doors do not, and then its known-model pathology
(gambling on an absorbing failure) takes over as soon as it has learned
the route. Δ_B does not care about the difference: a flat door is worth
exactly the information it carries, the visits are steps along the
route, and the term is gone once the route is known.
- "Unknown doors" is a natural partial map — corridors whose walls are
known but whose passages are not — and the cheapest learn-B benchmark in
the repo (6 columns, 5×4, H = 7). The frozen table shows its one caveat:
entropy followers walk it without learning, so the learn-B claim for the
dyn-channel variants needs the frozen row read alongside.



## Reproduction

```
uv run python run_unmapped_detour.py --depict
uv run python run_unmapped_detour.py --compare
uv run python run_unmapped_detour.py --scenario learn-B --episodes 100 --n-seeds 3 \
    --planners EFE EFE-noamb-noig SP-EFE-dyn SP-EFE-DeltaB SP-Pref-Plan \
    --out figures/unmapped_detour_p0.50_learnB_H7_analytic
uv run python run_unmapped_detour.py --scenario learn-B --episodes 1000 --n-seeds 1 \
    --planners SP-Pref SP-EFE-dyn --out figures/unmapped_detour_p0.50_learnB_H7_1k
uv run python run_unmapped_detour.py --noisy-tv --scenario known --episodes 10
uv run python run_unmapped_detour.py --noisy-tv --scenario learn-B --episodes 200 --n-seeds 3
# doors variant (beats both KL planners)
uv run python run_unmapped_detour.py --unmapped doors --compare --episodes 300 --n-seeds 3
uv run python run_unmapped_detour.py --unmapped doors --action-order WSEN --scenario learn-B \
    --episodes 200 --n-seeds 1 --planners SP-Pref SP-EFE-dyn SP-Pref-Plan
uv run python run_unmapped_detour.py --unmapped doors --layout mirror --scenario learn-B --episodes 300 --n-seeds 3
uv run python run_unmapped_detour.py --unmapped doors --layout mirror --scenario learn-B --episodes 100 --n-seeds 1 \
    --planners EFE --out figures/unmapped_detour_p0.50_mirror_doors_learnB_H8_analytic
uv run python run_unmapped_detour.py --unmapped doors --scenario learn-B --episodes 100 --n-seeds 1 \
    --planners EFE EFE-noamb-noig --out figures/unmapped_detour_p0.50_doors_learnB_H7_analytic
# geometry-only attempt (negative result)
uv run python run_unmapped_detour.py --layout mirror --scenario learn-B --episodes 150 --n-seeds 1 \
    --planners SP-Pref SP-EFE-dyn SP-Pref-Plan
```

Regression: [`tests/test_unmapped_detour.py`](../tests/test_unmapped_detour.py)
(tensors, layout invariants, ice-slip frequency, doors-mode columns, mirror /
rest-cell layouts, action relabelling, known detour for `SP-EFE-dyn` /
`SP-Pref-Plan`, frozen `SP-Pref-Plan` gambles).