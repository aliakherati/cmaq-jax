"""Vertical diffusion — the ACM2 solvers.

Ports `vdiff/acm2_m3dry/`. CMAQ's science driver runs vertical diffusion
**first**, before coupling and transport (`sciproc.F`), on uncoupled
concentrations — so it is not part of the coupled transport block that
:func:`cmaq_jax.api.transport_step` implements.

ACM2 (Pleim 2007) splits each sub-step in two:

* a **non-local convective stage**, active only where the column is convective,
  in which mass leaves the surface layer and arrives directly in every layer of
  the convective boundary layer, returning by layer-to-layer subsidence. Because
  every row couples to column 1, the matrix is *tridiagonal plus a full first
  column*, and needs :func:`solve_acm1` rather than :func:`solve_tridiagonal`;
* a **local diffusion stage**, always active, which is ordinary vertical
  diffusion and uses the Thomas algorithm.

Both stages are Crank–Nicolson (`THETA = 0.5`, `vdiffacmx.F:94`), so the
sub-step limit is an *accuracy* constraint rather than a stability one — a real
difference from horizontal diffusion, where the analogous constant sits past the
stability boundary.

Both solvers take one matrix and many right-hand sides: the column's
factorisation is shared across every species, and doing it per species would be
both slower and a different computation to read.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["solve_acm1", "solve_tridiagonal"]


def solve_tridiagonal(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """Solve a tridiagonal system by the Thomas algorithm. Ports ``tri.F``.

    The matrix follows CMAQ's storage (``tri.F:40-46``): row ``k`` holds
    ``sub[k]`` at column ``k-1``, ``diag[k]`` at ``k`` and ``sup[k]`` at ``k+1``.
    So ``sub[0]`` and ``sup[-1]`` are never referenced — the golden generator
    fills them with a poison value so that a port which touches one disagrees
    loudly rather than subtly.

    ``rhs`` is ``(nspcs, nlays)``; the result has the same shape. The layer axis
    is last because that is the axis the recurrence runs along, and CMAQ stores
    it that way too.

    The recurrence is sequential by nature. At ``nlays ~ 35`` a scan is
    inexpensive; on a GPU a short sequential scan can dominate, which is a
    question for the benchmark rather than an assumption to make here.
    """
    sub, diag, sup, rhs = (jnp.asarray(a) for a in (sub, diag, sup, rhs))
    nlays = diag.shape[0]

    def forward(
        carry: tuple[Array, Array], k: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        bet, prev = carry
        gam = bet * sup[k - 1]
        bet = 1.0 / (diag[k] - sub[k] * gam)
        x = bet * (rhs[:, k] - sub[k] * prev)
        return (bet, x), (x, gam)

    bet0 = 1.0 / diag[0]
    x0 = bet0 * rhs[:, 0]
    (_, _), (xs, gams) = jax.lax.scan(forward, (bet0, x0), jnp.arange(1, nlays))

    # xs/gams cover layers 1..nlays-1; prepend layer 0 (whose gamma is unused).
    x_all = jnp.concatenate([x0[:, None], xs.T], axis=1)
    gam_all = jnp.concatenate([jnp.zeros_like(gams[:1]), gams])

    def backward(nxt: Array, k: Array) -> tuple[Array, Array]:
        # GAM(K+1), not GAM(K) -- tri.F:113 indexes the gamma of the layer
        # *above* the one being corrected.
        x = x_all[:, k] - gam_all[k + 1] * nxt
        return x, x

    _, back = jax.lax.scan(backward, x_all[:, -1], jnp.arange(nlays - 2, -1, -1))

    # `back` is layers nlays-2 .. 0, so reverse it and re-attach the top layer.
    return jnp.concatenate([back[::-1].T, x_all[:, -1:]], axis=1)


def solve_acm1(col: Array, diag: Array, sup: Array, rhs: Array, kl: Array | int) -> Array:
    """Solve the ACM1 system — tridiagonal plus a full first column.

    Ports ``matrix1.F``. Over rows ``1..kl`` (1-based, as CMAQ counts):

    * row 1: ``diag[0]·x[0] + sup[1]·x[1]``
    * rows 2..kl-1: ``col[L]·x[0] + diag[L]·x[L] + sup[L+1]·x[L+1]``
    * row kl: ``col[kl]·x[0] + diag[kl]·x[kl]``

    ``col`` is a *column*, not a subdiagonal, and ``sup[L]`` sits in row ``L-1``.
    ``col[0]`` and ``sup[0]`` are unused. This layout was confirmed by residual
    against the compiled Fortran, not read off the comment block.

    ``kl`` is the top of the convective boundary layer and varies per column in
    a real run, so it is handled by masking rather than by slicing: rows above it
    are returned as zero, matching a caller that zeroes ``X`` before the call.

    The elimination uses a running product ``alpha[L] = Π(-sup/diag)``, which
    over a deep CBL gets very small — measured at 1.9e-38 on the
    ``alpha_underflow`` golden, right at float32's smallest normal. That is
    harmless: ``alpha`` weights contributions to ``gama`` that are already
    negligible beside ``diag[0]``, so losing them to underflow costs nothing.
    Guarding it would be guarding against the arithmetic working correctly.
    """
    col, diag, sup, rhs = (jnp.asarray(a) for a in (col, diag, sup, rhs))
    nlays = diag.shape[0]
    layer = jnp.arange(nlays)
    kl = jnp.asarray(kl)

    # Rows 1..kl-1 (0-based) take part in the elimination; row 0 seeds it.
    inner = (layer >= 1) & (layer < kl)

    # alpha[L] = prod_{k=1..L} (-sup[k]/diag[k]); 1 outside the active range so
    # the cumulative product is unaffected, then zeroed so it contributes
    # nothing to the sums below.
    ratio = jnp.where(inner, -sup / diag, 1.0)
    alpha = jnp.where(inner, jnp.cumprod(ratio), 0.0)

    beta = rhs[:, 0] + (alpha * rhs).sum(axis=1)
    gama = diag[0] + (alpha * col).sum()
    x0 = beta / gama

    # Row kl-1 has no super-diagonal term inside the system, so zero sup there
    # and one downward recurrence covers every active row including the last.
    sup_eff = jnp.where(layer < kl, sup, 0.0)

    def backward(nxt: Array, k: Array) -> tuple[Array, Array]:
        upper = jnp.where(k + 1 < nlays, sup_eff[jnp.minimum(k + 1, nlays - 1)], 0.0)
        x = (rhs[:, k] - col[k] * x0 - upper * nxt) / diag[k]
        return x, x

    _, back = jax.lax.scan(backward, jnp.zeros_like(x0), jnp.arange(nlays - 1, 0, -1))

    solution = jnp.concatenate([x0[:, None], back[::-1].T], axis=1)
    return jnp.where(layer < kl, solution, 0.0)
