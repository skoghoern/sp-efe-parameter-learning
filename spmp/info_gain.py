"""Per-entry Dirichlet information-gain weights Δ_A and Δ_B.

Δ = KL(Dir(α+e) ‖ Dir(α+n̄)) is the one-observation conjugate information
gain from ``dirichlet_posterior_kl``. Δ ≥ 0, so e^Δ is an epistemic bonus;
it b-averages to the novelty terms and vanishes as counts grow.

Full-argument channels are exact posteriors Dir(α+e); cavity channels are
their b-weighted geometric mixture Dir(α+n̄), with n̄ = b(outcome|x).
Structural zeros map to LOG_ZERO and must carry n̄ = 0.
"""

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, polygamma

from .sweeps import LOG_ZERO
from .tables import STRUCTURAL_ZERO_RTOL

# Above this, gammaln(α+δ) − gammaln(α) is computed by Taylor expansion: the
# direct difference suffers catastrophic float32 cancellation (gammaln(1e6) ≈
# 1.3e7, so an O(10) difference would carry O(1) error). Both branches are
# accurate to ~1e-5 at the crossover.
_GAMMALN_TAYLOR_THRESHOLD = 10.0


def _gammaln_diff(alpha, delta):
    """Numerically stable logΓ(α+δ) − logΓ(α) for fractional counts 0 ≤ δ ≤ 1.

    Direct difference for small α; for large α the Taylor series
    Σ_{k=1..4} δ^k/k! · ψ^{(k−1)}(α), whose truncation error is
    O(δ⁵·ψ⁽⁴⁾(α)) ≲ 1e-5 at the threshold and shrinks as α grows.
    """
    direct = gammaln(alpha + delta) - gammaln(alpha)
    taylor = (delta * digamma(alpha)
              + delta**2 / 2.0 * polygamma(1, alpha)
              + delta**3 / 6.0 * polygamma(2, alpha)
              + delta**4 / 24.0 * polygamma(3, alpha))
    return jnp.where(alpha > _GAMMALN_TAYLOR_THRESHOLD, taylor, direct)


def _validate_axes(ndim, axis, cond_axes):
    """Resolve and sanity-check the axis layout (trace-time, shapes are static)."""
    cond_axes = tuple(sorted(a % ndim for a in cond_axes))
    axis = axis % ndim
    if axis in cond_axes:
        raise ValueError(f"Dirichlet axis {axis} cannot be a conditioning axis {cond_axes}")
    sum_axes = tuple(i for i in range(ndim) if i not in cond_axes)
    return axis, cond_axes, sum_axes


def dirichlet_posterior_kl(alpha, nbar, axis, cond_axes):
    """Per-entry Δ ≥ 0.

        Δ_k = KL(Dir(α+e_k) ‖ Dir(α+n̄)) + Σ_{c ≠ col} KL(Dir(α_c) ‖ Dir(α_c+n̄_c))

    ``cond_axes (0, 2)`` (α_A): one column per slice, so the sum is empty.
    ``cond_axes (1,)`` (α_B): the sum runs over the u-columns of a state.
    """
    axis, cond_axes, sum_axes = _validate_axes(alpha.ndim, axis, cond_axes)

    alpha0 = jnp.sum(alpha, axis=axis, keepdims=True)
    # Relative structural-zero test (STRUCTURAL_ZERO_RTOL): denormal learned
    # "dust" (α≈1e-38) is excluded so digamma(α)≈−1/α cannot blow the KL up to
    # ~1e37 (the obs-EFE/contrast NaN on learned frozen-lake counts).
    valid = alpha > STRUCTURAL_ZERO_RTOL * alpha0
    safe_alpha = jnp.where(valid, alpha, 1.0)
    col_valid = alpha0 > 0
    safe_alpha0 = jnp.where(col_valid, alpha0, 1.0)
    ok = valid & col_valid
    m = jnp.sum(nbar, axis=axis, keepdims=True)

    # Column sums: Sg = Σ_c [logΓ(α_c+n̄_c) − logΓ(α_c)], Sψ = Σ_c n̄_c ψ(α_c)
    Sg = jnp.sum(jnp.where(ok, _gammaln_diff(safe_alpha, nbar), 0.0),
                 axis=axis, keepdims=True)
    Spsi = jnp.sum(jnp.where(ok, nbar * digamma(safe_alpha), 0.0),
                   axis=axis, keepdims=True)

    # KL(Dir(α+e_k) ‖ Dir(α+n̄)) expanded with stable gammaln differences
    kl_entry = (jnp.log(safe_alpha0) - _gammaln_diff(safe_alpha0, m)
                - jnp.log(safe_alpha) + Sg
                + digamma(safe_alpha + 1.0) - nbar / safe_alpha
                - Spsi - (1.0 - m) * digamma(safe_alpha0 + 1.0))

    # KL(Dir(α) ‖ Dir(α+n̄)) per column (k-independent)
    kl0 = jnp.where(col_valid,
                    -_gammaln_diff(safe_alpha0, m) + Sg - Spsi
                    + m * digamma(safe_alpha0), 0.0)

    # Cross-column aggregation over the conditioning slice
    other_axes = tuple(a for a in sum_axes if a != axis)
    kl0_total = kl0.sum(axis=other_axes, keepdims=True) if other_axes else kl0

    return jnp.where(ok, kl_entry - kl0 + kl0_total, LOG_ZERO)


def uniform_over_valid(valid, axes):
    """Uniform fractional counts over the valid entries of each conditioning slice.

    The admissible initial n̄ for ``dirichlet_posterior_kl``: no mass on
    structural zeros, otherwise the gammaln column totals would be
    inconsistent with the masked per-entry sums.
    """
    counts = jnp.sum(valid, axis=axes, keepdims=True)
    return jnp.where(valid, 1.0 / jnp.maximum(counts, 1), 0.0)
