"""Frozen-θ Dirichlet log-tables for the conjugate kernels.

Geometric (digamma) E[log θ] is used in factor functions; arithmetic (mean) log E[θ] is used in channel updates r()
refresh. Their gap is the parameter information gain (SP-EFE uses arithmetic).
Structural zeros are LOG_ZERO.
"""

import jax.numpy as jnp
from jax.scipy.special import digamma

from .sweeps import LOG_ZERO

# Counts ≤ rtol · column total are structural zeros. Learned impossible events
# pick up leakage (~1e-30..1e-18); digamma(α)≈−1/α would explode the
# novelty KL. Leakage ≪ 1e-6·α₀ ≪ any genuine count.
STRUCTURAL_ZERO_RTOL = 1e-6


def dirichlet_log_tables(alpha, axis=0):
    """Geometric and arithmetic expected-log tables for independent Dirichlets.

    One Dirichlet over `axis` per remaining index. Entries ≤
    STRUCTURAL_ZERO_RTOL · Σ_axis α are structural zeros (LOG_ZERO).

    Returns (log_geo, log_mean), same shape as alpha:
        log_geo = ψ(α) − ψ(Σ α) = E[log θ]
        log_mean = log(α / Σ α) = log E[θ]
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
