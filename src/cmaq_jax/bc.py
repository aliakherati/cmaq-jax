"""Boundary conditions and the halo they fill.

A PPM sweep reads three cells beyond the domain edge, so the array carries a
``halo_width`` ghost region at each end. What goes in it depends on which way the
wind blows across that edge:

* **inflow** -- take the value from the boundary-condition field, which in CMAQ
  comes from a coarser parent run or a climatology (``rdbcon.F``);
* **outflow** -- extrapolate from inside the domain with the zero-flux-divergence
  condition, which lets material leave without reflecting back
  (``zfdbc.f``, after Pleim 1991).

The halo is deliberately a **first-class region of the array** even though a
single device fills it locally. Under ``shard_map`` the same region is filled by
a collective permute from the neighbouring shard instead, and nothing else in
the sweep changes. Deleting it now would mean rebuilding it later.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cmaq_jax.config import DEFAULT_PPM, PPMConstants

__all__ = ["fill_halo", "zfdbc"]


def zfdbc(
    c1: Array,
    c2: Array,
    v1: Array,
    v2: Array,
    ppm: PPMConstants = DEFAULT_PPM,
) -> Array:
    """Zero-flux-divergence outflow boundary value (Pleim, JGR 1991).

    Ports ``zfdbc.f`` exactly. ``c1``/``v1`` are the cell and face nearest the
    boundary, ``c2``/``v2`` the next ones in. Returns the value to place in the
    ghost cells.

    Three outcomes, in the Fortran's order:

    * the near-boundary wind is negligible (``|v1| < 1e-3``) -- pass ``c1``
      through, since there is no meaningful flux to extrapolate along;
    * the two velocities disagree in sign, so the flow diverges right at the
      edge -- pass ``c1`` through, because the extrapolation is meaningless
      there (the Fortran comments this as "nothing changes for wind divergence
      at edge");
    * otherwise extrapolate ``c1 - (v2/v1)(c2 - c1)``, **clamped at zero**.

    The clamp is not cosmetic. Without it a steep gradient at an outflow edge
    manufactures negative concentrations, which then propagate inward.
    """
    c1, c2 = jnp.asarray(c1), jnp.asarray(c2)
    v1, v2 = jnp.asarray(v1), jnp.asarray(v2)

    # Guard the division so the unused branch cannot produce inf/nan. Under
    # jnp.where both branches are evaluated, and a nan here would poison the
    # gradient even though the value is discarded.
    safe_v1 = jnp.where(jnp.abs(v1) >= ppm.zfdbc_small_wind, v1, 1.0)
    extrapolated = jnp.maximum(0.0, c1 - (v2 / safe_v1) * (c2 - c1))

    usable = (jnp.abs(v1) >= ppm.zfdbc_small_wind) & (v1 * v2 > 0.0)
    return jnp.where(usable, extrapolated, c1)


def fill_halo(
    con: Array,
    vel: Array,
    bcon_lo: Array,
    bcon_hi: Array,
    ppm: PPMConstants = DEFAULT_PPM,
) -> Array:
    """Fill the ghost cells at both ends of the sweep axis.

    Ports the boundary blocks of ``x_ppm.F:418-441`` (and their mirror in
    ``y_ppm.F``). ``con`` is the padded array with the sweep axis first; ``vel``
    holds the ``ni + 1`` face velocities; ``bcon_lo``/``bcon_hi`` are the inflow
    concentrations for the low and high edges, broadcastable against ``con``'s
    trailing axes.

    Every ghost cell on a side gets the *same* value -- CMAQ assigns the whole
    ``1-SWP:0`` slice at once -- so the reconstruction sees a flat approach to
    the boundary and the limiter reads no artificial gradient there.

    Which side is "outflow" depends on the sign convention: the low edge flows
    out when ``vel[0] < 0`` (leftward, out of the domain) and the high edge when
    ``vel[-1] > 0`` (rightward, out of the domain).
    """
    con = jnp.asarray(con)
    vel = jnp.asarray(vel)
    swp = ppm.halo_width
    ni = con.shape[0] - 2 * swp
    if ni < 2:
        raise ValueError(f"con has {ni} interior cells; need at least 2 for the outflow stencil")

    interior = con[swp : swp + ni]

    lo_value = jnp.where(
        vel[0] < 0.0,
        zfdbc(interior[0], interior[1], vel[0], vel[1], ppm),
        jnp.asarray(bcon_lo),
    )
    hi_value = jnp.where(
        vel[-1] > 0.0,
        zfdbc(interior[-1], interior[-2], vel[-1], vel[-2], ppm),
        jnp.asarray(bcon_hi),
    )

    lo_block = jnp.broadcast_to(lo_value, (swp, *con.shape[1:]))
    hi_block = jnp.broadcast_to(hi_value, (swp, *con.shape[1:]))
    return jnp.concatenate([lo_block, interior, hi_block])
