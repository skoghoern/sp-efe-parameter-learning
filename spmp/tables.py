"""Dirichlet expected-log tables, the entry point to every conjugate kernel.

θ is frozen during planning, so its entire influence on the graph collapses into
two precomputed log-tables per parameter block:

- Geometric (digamma) table: log ᾱ = E_q[log θ] = ψ(α) − ψ(Σ α),
  replacing every E_{q_θ}[log ·] in the modified kernels.
- Arithmetic (mean) table:   log α̂ = log(α / Σ α),
  replacing every arithmetic θ-marginalization in the channel refresh.

The per-entry Jensen gap log_mean − log_geo ≥ 0, large at low counts, is the
parameter information gain. SP-EFE uses the arithmetic table.

All computation is in log-space, with LOG_ZERO for structural zeros.
"""

import jax.numpy as jnp
from jax.scipy.special import digamma

from .sweeps import LOG_ZERO

# Structural-zero tolerance (relative to each Dirichlet's column total). After
# conjugate count learning, impossible events accumulate denormal "dust" — tiny
# belief-leak counts (~1e-30..1e-18) added on top of their exact-zero prior. Such
# entries are NOT real evidence and must stay structural zeros: digamma(α)≈−1/α
# for α≈1e-38 otherwise explodes the θ-novelty KL to ~1e37 (then NaN) and leaks
# kernel probability onto impossible transitions. The gap between dust
# (≲1e-18·α₀) and any genuine count (≳1e-3·α₀) is enormous, so a single relative
# floor cleanly separates them and leaves exact-zero priors untouched.
STRUCTURAL_ZERO_RTOL = 1e-6


def dirichlet_log_tables(alpha, axis=0):
    """Compute geometric and arithmetic expected-log tables of a Dirichlet collection.

    Treats `alpha` as a collection of independent Dirichlets: one distribution
    over `axis` per remaining index (e.g. alpha_A (Y, X) with axis=0 is one
    Dirichlet over y per state x; alpha_B (X', X, U) with axis=0 is one
    Dirichlet over x' per (x, u) pair).

    Args:
        alpha: concentration tensor; entries negligible relative to their
            column total (≤ STRUCTURAL_ZERO_RTOL · Σ_axis α, including exact 0
            and post-learning denormal "dust") are treated as structural zeros
        axis: axis along which each Dirichlet is normalized

    Returns:
        (log_geo, log_mean), same shape as alpha:
        - log_geo:  ψ(α) − ψ(Σ_axis α), i.e. E_{Dir(α)}[log θ]
        - log_mean: log(α / Σ_axis α),  i.e. log E_{Dir(α)}[θ]
        Structural zeros map to LOG_ZERO in both tables. Elementwise
        log_mean ≥ log_geo, with the gap vanishing as counts grow.
    """
    alpha_sum = jnp.sum(alpha, axis=axis, keepdims=True)
    # Relative structural-zero test: drop denormal learned "dust" (see
    # STRUCTURAL_ZERO_RTOL) so digamma/log never see ~0 concentrations.
    valid = alpha > STRUCTURAL_ZERO_RTOL * alpha_sum
    # Substitute 1.0 in invalid slots so digamma/log are never evaluated at 0;
    # results there are overwritten with LOG_ZERO below.
    safe_alpha = jnp.where(valid, alpha, 1.0)

    sum_valid = alpha_sum > 0  # all-zero slices (no valid Dirichlet) → LOG_ZERO rows
    safe_sum = jnp.where(sum_valid, alpha_sum, 1.0)

    ok = valid & sum_valid
    log_geo = jnp.where(ok, digamma(safe_alpha) - digamma(safe_sum), LOG_ZERO)
    log_mean = jnp.where(ok, jnp.log(safe_alpha) - jnp.log(safe_sum), LOG_ZERO)
    return log_geo, log_mean
