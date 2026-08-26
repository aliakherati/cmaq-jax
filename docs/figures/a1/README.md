# A1 figures — horizontal advection through the real driver

Regenerate with:

```bash
python scripts/make_a1_figures.py
```

The [A0 flow figures](../a0/README.md) applied the bare 1-D kernel alternately
along each axis with periodic wrapping, and carried a caveat saying so. These
use `cmaq_jax.hadv.hadv_step` — the actual port of `hadvppm.F` — so everything
the driver does is active: `BCON` on inflow, zero-flux-divergence on outflow,
the X-Y/Y-X alternation, per-layer sub-stepping, and the ρ·J ride-along.

---

## `rotation_driver.png`

Zalesak's cone and slotted cylinder carried once around by a rigid rotation,
96×96 cells, 900 steps at Courant 0.34. The exact answer after a full turn is
the initial field.

| | |
|---|---|
| **Phase error** | 0.006 cells — the shapes come back where they started |
| **Mass** | conserved to 1e-10 |
| **ρ·J** | held at 1.0 with **zero** drift |
| **Undershoot** | 1.2e-57 |

The ρ·J result is the one worth pausing on. The rotation wind is *discretely*
non-divergent — `u` varies only with row and `v` only with column, so every
cell's flux divergence is exactly zero — and the scheme reproduces that
exactly, not approximately. Any drift there would mean the flux-form update was
not telescoping properly, and every mixing ratio in the model would inherit the
error.

The error panel's max of 0.74 is edge diffusion at the slot walls, not
displacement; see the [A0 notes](../a0/README.md#reading-the-error-map-in-rotation_2dpng)
for why phase error and L-infinity error need separating.

---

## `periodic_vs_driver.png`

The same rotation run both ways, to show what the boundary treatment is
actually worth.

For the first few steps the two are **bit-identical**. The shapes sit far from
the domain edge, so a periodic halo and a clean inflow both supply zero, and the
arithmetic is the same. Only once numerical diffusion has spread a tail right
round the domain do the halos start to differ — the periodic case wraps that
tail back in, while the driver's `BCON` supplies exactly zero. That seed then
grows with the flow:

| steps | max &#124;driver − periodic&#124; |
|---|---|
| 1 | 0 (bit-identical) |
| 5 | 0 (bit-identical) |
| 50 | 7e-21 |
| 300 | 1.7e-11 |
| 900 | 2.5e-8 |

The rightmost panel plots that growth against the float64 epsilon line. Two
things follow. The A0 figures were **not** misleading about the interior — for
the timescales they showed, the two agree to round-off. And the difference that
does exist is a boundary effect that takes hundreds of steps to become visible,
which is a more useful thing to know than a bare tolerance number.

---

## What is not shown here

The deformational swirl is not repeated through the driver. Its velocity field
vanishes on the domain boundary by construction, so `BCON` and the outflow
condition never engage and the result would be indistinguishable from the A0
version. The rotation is the case where the boundary actually participates.
