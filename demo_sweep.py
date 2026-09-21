#!/usr/bin/env python
"""Walk through SP-EFE on a POMDP small enough to print. Start here.

A 5-cell corridor, 2 actions, identity observations. The goal sits one step to
the LEFT; the two transitions to the RIGHT of the start are *unknown* (flat
Dirichlet), everything else is pinned to the truth:

        [0] --- [G] --- [S] --- [3] --- [4]
                         ^start  ?       ?        RIGHT -> +1, LEFT -> -1

So the two drives disagree, which is the paper's mechanism in miniature: LEFT
reaches the goal in one step, RIGHT is the only way to learn what the unknown
columns do. A planner with no novelty term takes the goal; SP-EFE pays one step
of preference to resolve the model first.

Everything the scheme is made of is visible here:

  1. the Dirichlet tables  Ã, B̃      (spmp/tables.py)
  2. the information gain  Δ_B       (spmp/info_gain.py)
  3. the channels r(u|x), r(x'|x,u)  refreshed to a fixed point (spmp/sp_efe.py)
  4. q(u_1), and how the ablations change it

Run:  uv run python demo_sweep.py
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from agent import CLAMP, FLAT, choose_action
from spmp import (
    dirichlet_log_tables,
    dirichlet_posterior_kl,
    sp_pref_planning,
    sp_efe_planning,
)

np.set_printoptions(precision=3, suppress=True, linewidth=100)

N_STATES, N_ACTIONS = 5, 2
LEFT, RIGHT = 0, 1
ACTION_NAME = {LEFT: "LEFT", RIGHT: "RIGHT"}
START, GOAL = 2, 1
UNKNOWN_CELLS = (2, 3)      # the (cell, RIGHT) columns the agent must learn
HORIZON = 3
PREFERENCE = 4.0


def build_tensors():
    """Identity A; deterministic corridor B; flat priors on two B columns."""
    A = np.eye(N_STATES, dtype=np.float32)[None]          # (1, obs, state)

    B = np.zeros((N_STATES, N_STATES, N_ACTIONS), np.float32)
    for x in range(N_STATES):
        B[max(x - 1, 0), x, LEFT] = 1.0                   # walls bump
        B[min(x + 1, N_STATES - 1), x, RIGHT] = 1.0
    B[:, GOAL, :] = 0.0
    B[GOAL, GOAL, :] = 1.0                                # goal absorbs

    # A is known; B is known except the RIGHT column of each unknown cell.
    alpha_A = jnp.asarray(CLAMP * A)
    alpha_B = np.asarray(CLAMP * B, dtype=np.float32)
    for x in UNKNOWN_CELLS:
        alpha_B[:, x, RIGHT] = FLAT                       # flat over successors
    return alpha_A, jnp.asarray(alpha_B), B


def show_tables(alpha_A, alpha_B):
    """Step 1: θ enters the graph only through two expected-log tables."""
    print(__doc__.split("Run:")[0].rstrip())
    print("\n" + "=" * 72)
    print("1. Dirichlet tables  (spmp/tables.py)")
    print("=" * 72)
    _, log_mean_B = dirichlet_log_tables(alpha_B, axis=0)
    m_B = np.exp(np.asarray(log_mean_B))
    print("\nposterior-mean m_B(x' | x, RIGHT): rows x', columns x")
    print(m_B[:, :, RIGHT])
    print(f"\nColumns {UNKNOWN_CELLS} are flat (mass 1/{N_STATES} each): the agent")
    print("does not know where RIGHT leads from there. Every other column is sharp.")


def show_info_gain(alpha_B):
    """Step 2: Δ_B, the per-entry information gain, as the weight e^Δ ≥ 1."""
    print("\n" + "=" * 72)
    print("2. Information gain  Δ_B  (spmp/info_gain.py)")
    print("=" * 72)
    a = jnp.asarray(alpha_B)
    a0 = a.sum(axis=0, keepdims=True)
    nbar = jnp.where(a > 0, a / jnp.maximum(a0, 1e-12), 0.0)
    delta_B = np.asarray(dirichlet_posterior_kl(a, nbar, axis=0, cond_axes=(1, 2)))
    w = np.exp(delta_B[:, :, RIGHT])
    print("\ninformation-gain weight e^Δ_B for (x' | x, RIGHT): rows x', columns x")
    print(w)
    known = [x for x in range(N_STATES) if x not in UNKNOWN_CELLS]
    print(f"\ne^Δ ≈ 1 on the known columns {known} (nothing left to learn) and")
    print(f"e^Δ > 1 on the flat columns {list(UNKNOWN_CELLS)}. This weight is what")
    print("makes visiting an unknown transition attractive; it decays as counts grow.")


def show_fixed_point(alpha_A, alpha_B, goal, q0):
    """Step 3: iterate {sweep -> refresh channels -> damp} and watch it settle."""
    print("\n" + "=" * 72)
    print("3. Channel fixed point  (spmp/sp_efe.py -> spmp/sweeps.py)")
    print("=" * 72)
    print("\nEach sweep is one forward_pass + one backward_pass; the channels are")
    print("then refreshed from the region beliefs and damped. Watch q(u_1) settle:\n")
    print(f"  {'sweeps':>7}  {'q(LEFT)':>9} {'q(RIGHT)':>9}   r(u=RIGHT | x) per cell")
    for n in (1, 2, 3, 5, 10, 50):
        ad, log_r_xxu, log_r_ux, _ = sp_efe_planning(
            q0, alpha_A, alpha_B, goal, HORIZON, n, damping=0.5)
        ad = np.asarray(ad)
        r_ux = np.exp(np.asarray(log_r_ux))     # (T, S, U)
        print(f"  {n:>7}  {ad[LEFT]:>9.4f} {ad[RIGHT]:>9.4f}   "
              f"{np.round(r_ux[0, :, RIGHT], 3)}")
    print("\nr(u|x) is the policy channel: the planning correction Σ H[q(u|x)]")
    print("reparameterized into the dynamics kernel. Uniform channels (0.5 here)")
    print("would recover standard Bethe BP.")
    return ad


def show_ablations(alpha_A, alpha_B, goal, q0):
    """Step 4: the ablations are the same function under different flags."""
    print("\n" + "=" * 72)
    print("4. The ablations are flags, not separate planners")
    print("=" * 72)
    variants = [
        ("SP-EFE",                  dict(dyn_channel=True,  dyn_tilt_coef=1, obs_tilt_coef=2)),
        ("SP-EFE^-dyn",             dict(dyn_channel=False, dyn_tilt_coef=1, obs_tilt_coef=2)),
        ("SP-EFE^-Db",              dict(dyn_channel=True,  dyn_tilt_coef=0, obs_tilt_coef=2)),
        ("SP-EFE^-Da-Db",           dict(dyn_channel=True,  dyn_tilt_coef=0, obs_tilt_coef=0)),
        ("SP-EFE^-Da-Db-dyn",       dict(dyn_channel=False, dyn_tilt_coef=0, obs_tilt_coef=0)),
    ]
    print(f"\n  {'planner':<22} {'q(LEFT)':>9} {'q(RIGHT)':>9}  action")
    for name, flags in variants:
        ad = np.asarray(sp_efe_planning(
            q0, alpha_A, alpha_B, goal, HORIZON, 50, damping=0.5, **flags)[0])
        print(f"  {name:<22} {ad[LEFT]:>9.4f} {ad[RIGHT]:>9.4f}  "
              f"{ACTION_NAME[choose_action(ad, N_ACTIONS)]}")
    for label, corr in (("SP-Pref-Plan", True), ("SP-Pref", False)):
        ad = np.asarray(sp_pref_planning(
            q0, alpha_A, alpha_B, goal, HORIZON, 50, damping=0.5,
            planning_correction=corr)[0])
        print(f"  {label:<22} {ad[LEFT]:>9.4f} {ad[RIGHT]:>9.4f}  "
              f"{ACTION_NAME[choose_action(ad, N_ACTIONS)]}")

    print("\nRead this table carefully -- it reproduces the two effects the paper")
    print("separates in Table 1:")
    print("  * The preference-only baselines go LEFT: they cannot see the unknown")
    print("    columns at all, so they just collect the goal.")
    print("  * Dropping BOTH novelty terms but keeping the dynamics channel still")
    print("    goes RIGHT. That is not novelty, it is entropy-following: the")
    print("    channel -H[q(x'|x,u)] rewards transitions that look uncertain, and a")
    print("    flat column is maximally uncertain. This is the dagger footnote in")
    print("    the paper's Table 1, and why the frozen-B column is reported.")
    print("  * Only with the channel off as well does the novelty-free planner")
    print("    behave like a pure goal-seeker -- and then it agrees with")
    print("    SP-Pref-Plan to four decimals, which is the paper's claim that")
    print("    stripping every correction recovers the preference-only planner.")
    print("    SP-Pref, lacking the planning correction too, is the softer one.")
    print("\ncoupled_planner_kwargs() maps each planner name onto these flags;")
    print("see spmp/sp_efe.py. dispatch.py chooses between SP-EFE and SP-Pref.")


def main():
    alpha_A, alpha_B, _ = build_tensors()
    goal = np.full(N_STATES, np.exp(-PREFERENCE), np.float32)
    goal[GOAL] = 1.0
    goal = jnp.asarray(goal / goal.sum())
    q0 = jnp.asarray(np.eye(N_STATES, dtype=np.float32)[START])

    show_tables(alpha_A, alpha_B)
    show_info_gain(alpha_B)
    ad = show_fixed_point(alpha_A, alpha_B, goal, q0)
    show_ablations(alpha_A, alpha_B, goal, q0)

    print("\n" + "=" * 72)
    print(f"Agent acts: {ACTION_NAME[choose_action(ad, N_ACTIONS)]}  "
          f"(argmax of q(u_1); see agent.choose_action)")
    print("=" * 72)
    print("\nNext: agent.py for the filter/learning loop every planner shares,")
    print("then experiments/ for the two grid POMDPs of the paper.")


if __name__ == "__main__":
    main()
