# A2 — Vertical advection (ZADV)

Parent: [`../PLAN-advection.md`](../PLAN-advection.md) · Depends on A0

Wrap the non-uniform-spacing PPM kernel in CMAQ's vertical machinery. Unlike
HADV, the vertical velocity is **not read from meteorology** — it is diagnosed
from mass continuity so that advected density reproduces met density.

**Gate:** advected ρ·J matches met ρ·J to the VPPM tolerance (1e-3 relative), and
column mass is conserved.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **A2.1** ✅ | `vadv.py`: flux/velocity diagnosis (`zadvppmwrf.F:340-411`) | `FLX_1 = 0` (impermeable surface); upwinded `VEL = FLX/RJT`; matches a hand-worked column | `pytest tests/unit/test_zadv_flux.py` |
| **A2.2** ✅ | `vadv.py`: `vppm_adjust_velocity` (`vppm.F:200-246`) — fixed-count sqrt-Newton with per-face masking | Converges to the Fortran's 1e-3 tolerance in ≤8 iterations on all golden cases; residual reported | `pytest tests/unit/test_vel_adjust.py` |
| **A2.3** ✅ | `vadv.py`: `vppm()` — adjusted velocity applied to all species | Matches `vppm` goldens end-to-end | `pytest tests/regression -k vppm` |
| **A2.4** ✅ | `vadv.py`: `zadv()` — per-column CFL sub-stepping (`zadvppmwrf.F:412-459`) | Fixed-count `fori_loop` with `dt_remaining` masking; non-convergence surfaces in diagnostics | `pytest tests/unit/test_zadv.py` |
| **A2.5** | Property tests: column mass conservation, ρ·J reproduction, positivity, vertical constancy | All pass across sigma profiles and CFL regimes | `pytest tests/properties -k zadv` |
| **A2.6** | Figures `docs/figures/a2/` | Diagnosed `w` profile; ρ·J reproduction error vs. layer; sub-step count map | `python scripts/make_a2_figures.py` |

## Notes

**Two nested ragged loops**, both replaced by fixed-count `lax.fori_loop`:

1. The per-face velocity adjustment (`GO TO 66/77`, up to 50 iterations) — a
   sqrt-Newton, `vel ← vel·sqrt(F_target/F_ppm)`. 8 iterations is ample at the
   1e-3 tolerance. **Guard the ratio against non-positive `F_ppm`** or the
   `sqrt` produces NaN. Expose a `stop_gradient` switch — the fixed point is not
   cleanly differentiable.
2. The per-column CFL sub-step (`GO TO 111`) — carry `dt_remaining`, mask with
   `jnp.where(dt_remaining > 0, …)`.

Fortran calls `M3EXIT` when either exceeds its cap. That failure mode must stay
**visible** — record the post-loop residual in a diagnostics dict rather than
letting non-convergence pass silently.

**`FBLN` is identically 1.0** (`zadvppmwrf.F:249`; the sigmoid is commented out),
so the blend at line 372 collapses to `FLX`. Dropped, and recorded as a
deviation.

**`ds` is constant in space and time** — `DS(L) = |X3FACE_GD(L) - X3FACE_GD(L-1)|`
is a fixed sigma thickness. That is why Fortran can `SAVE` the mesh coefficients,
and why they become precomputed constants on `GridConfig`.
