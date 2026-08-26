# Plan — port the CMAQ advection core to JAX

Parent: [`../ULTRAPLAN.md`](../ULTRAPLAN.md) · Chunk IDs `A0.x` … `A3.x`

## Scope

The complete advection operator as `sciproc.F` invokes it: **HADV** (horizontal,
PPM, X and Y sweeps) and **ZADV** (vertical, PPM with mass-continuity-derived
velocity), including the ρ·J ride-along that conserves mass.

Target configuration is CMAQ v5.5 default: `ModAdv = wrf_cons`
(`bldit_cctm.csh:115`), which selects `hadv/ppm` + `vadv/wrf_cons`.
`hadv/ppm` is the only horizontal scheme remaining in v5.5 — the `yamo`
variants were removed in 2019.

**Fidelity:** physics-faithful free rewrite. The PPM scheme and its monotonicity
constraints are reproduced exactly; the surrounding structure is free to change
for GPU efficiency and differentiability (float64, flux form, branchless
`where`, fixed-count loops).

## The Fortran being ported

Vendored verbatim under [`reference/fortran/`](../../reference/PROVENANCE.md).

| File | Lines | Role |
|---|---|---|
| `hppm.F` | 477 | **1-D PPM kernel, uniform spacing.** Both horizontal sweeps. |
| `vppm.F` | 544 | **1-D PPM kernel, non-uniform spacing** + iterative velocity adjustment. |
| `hadvppm.F` | 266 | HADV driver: layer loop, sub-cycling, X-Y/Y-X alternation |
| `x_ppm.F` / `y_ppm.F` | 665 / 659 | Slab gather, halo, boundary conditions, call HPPM |
| `hcontvel.F` | 359 | Contravariant velocity: `UHAT = UHAT_JD / mean(DENSA_J)` |
| `rdbcon.F` | 597 | Boundary concentrations, coupled by Jacobian/msfx² |
| `zfdbc.f` | 43 | Zero-flux-divergence outflow BC (Pleim 1991) |
| `zadvppmwrf.F` | 499 | ZADV: diagnose FLX/VEL from ρ·J continuity, CFL sub-stepping |
| `advstep.F` | 539 | CFL analysis → sync step + per-layer `ASTEP` |

Only `hppm.F` and `vppm.F` are compiled (for goldens). The rest are read.

## Invariants that must not break

1. **ρ·J rides along as advected species `N_SPC_ADV`.** This *is* the
   mass-conservation mechanism (`x_ppm.F:312`, `zadvppmwrf.F`). Not an
   optimisation to be removed.
2. **Monotonicity and positivity.** PPM's limiter (Colella & Woodward eqs. 1.8,
   1.10) is the whole point of the scheme. No new extrema; non-negative in →
   non-negative out.
3. **Constancy preservation.** A uniform mixing ratio under an arbitrary
   divergent wind must stay uniform. This is the test most likely to catch a
   coupling error.

## Phases

| Phase | Subplan | Gate |
|---|---|---|
| **A0** Foundation | [`subplans/A0-foundation.md`](subplans/A0-foundation.md) | Both kernels match Fortran goldens; 1-D analytic tests pass |
| **A1** HADV | [`subplans/A1-hadv.md`](subplans/A1-hadv.md) | 2-D rotating cone; horizontal constancy preservation |
| **A2** ZADV | [`subplans/A2-zadv.md`](subplans/A2-zadv.md) | Advected ρ·J reproduces met ρ·J; column mass conserved |
| **A3** Integrate | [`subplans/A3-integrate.md`](subplans/A3-integrate.md) | Full `advect_step` jits; `jax.grad` works; GPU benchmark |

## Deliberate deviations from the Fortran

Each is recorded in `README.md` with its reason.

| Deviation | Reason |
|---|---|
| float64 default (Fortran float32) | Accuracy; float32 selectable via `GridConfig.dtype` |
| `FBLN` blend term dropped | Hard-set to 1.0 upstream (`zadvppmwrf.F:249`); the term is identically a no-op |
| Fixed-count loops replace `GO TO` loops | Unbounded data-dependent loops don't jit; residual is reported instead of `M3EXIT` |
| ISAM / DDM-3D / IPR branches omitted | Out of scope (see ultraplan) |
| MPI halo exchange removed, halo *abstraction* kept | Single-device today, `shard_map` later without a rewrite |

## Known external dependency

There is **no benchmark meteorology available locally** — CMAQ ships run scripts
but not data, and `$CMAQ_DATA` must be downloaded separately
(`DOCS/CMAQ_Data.md`). A0–A2 therefore run entirely on synthetic fields.
`io_mcip.py` (A3) is written against the MCIP variable names already identified
(`UHAT_JD`, `VHAT_JD`, `DENSA_J`, `JACOBM`, `ZF`, `MSFX2`) but can only be
exercised once real input exists. This is the one deliverable gated on something
outside the repo.
