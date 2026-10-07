"""Dirichlet θ-posterior contrasts: the per-entry information-gain weights.

This is where the paper's Δ terms are computed. Per kernel entry, the contrast
is the exact one-observation conjugate information gain

    Δ = E_{Dir(α+e)}[log r(θ|·) − log r(θ|x)]
      = KL(Dir(α+e) ‖ Dir(α+n̄))   (+ cross-column terms from the
                                     product-Dirichlet factorization)

assembled by ``dirichlet_posterior_kl``. Δ ≥ 0, so e^Δ ≥ 1 is an epistemic
bonus, and it b-averages to +I_b(θ; outcome | x) — the novelty terms Δ_A and
Δ_B of the objective. Δ → 0 pointwise as counts grow, which is the point-mass
(fixed-parameter) limit.

Two projections make the contrast closed-form. Full-argument θ-posterior
channels r(θ|x,y), r(θ|x',x,u) are exact one-observation posteriors Dir(α + e);
the cavity channels r(θ_A|x), r(θ_B|x_{t-1}) are geometric projections of
Dirichlet mixtures, and a b-weighted geometric mixture of one-observation
posteriors is again Dirichlet, Dir(α + n̄), with fractional expected counts
n̄ = b(outcome|x).

All of it follows from the expected log-density of one Dirichlet under another
with the same support:

    E_{Dir(α)}[log Dir(θ; β)]
        = Σ_j (β_j − 1)(ψ(α_j) − ψ(α_0)) + logΓ(β_0) − Σ_j logΓ(β_j),

with α_0 = Σ_j α_j, β_0 = Σ_j β_j. Structural zeros (α = 0) are excluded from
every sum and map to LOG_ZERO in Δ; fractional counts must respect them
(n̄ = 0 wherever α = 0), which holds when n̄ comes from block conditionals
built on LOG_ZERO-masked kernels.
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
    """Joint-average θ-posterior contrast Δ ≥ 0 (design decision 3').

    Per kernel entry k of a column (one Dirichlet of the collection),

        Δ_k = KL( Dir(α_col + e_k) ‖ Dir(α_col + n̄_col) )
            + Σ_{other columns c in the conditioning slice}
                  KL( Dir(α_c) ‖ Dir(α_c + n̄_c) ),

    i.e. the exact E_{b(θ|entry)}[log r(θ|entry) − log r(θ|x)] for the
    product-of-Dirichlets block posterior: the observed column moves to its
    one-observation posterior, all other columns of the same conditioning
    slice contribute their prior-vs-cavity cross terms (constant in k and in
    the column choice only through which column is excluded). For alpha_A
    (cond_axes (0, 2)) each slice holds a single column, so Δ reduces to the
    plain KL; for alpha_B (cond_axes (1,)) the u-columns of a state x
    aggregate.

    Numerically stable at large α via `_gammaln_diff`. Structural zeros map
    to LOG_ZERO; `nbar` must be 0 there (as produced by masked kernels).

    Args:
        alpha, nbar, axis, cond_axes: as in `dirichlet_cavity_tilt`.

    Returns:
        delta: same shape as `alpha`.
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
