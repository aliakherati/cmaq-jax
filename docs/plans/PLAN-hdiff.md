# cmaq-jax — horizontal diffusion

Parent: [`../ULTRAPLAN.md`](../ULTRAPLAN.md) · Owns chunk IDs `B0`–`B2` ·
Depends on the advection port (`A0`–`A3`) for the grid, halo and harness
machinery it reuses wholesale.

## What this is

Operator 2 of the ultraplan. `sciproc.F` runs it immediately after `ZADV`:

```
VDIFF → COUPLE → HADV → ZADV → HDIFF → DECOUPLE → PHOT → CLDPROC → CHEM → AERO
```

Deformation-dependent horizontal eddy diffusion, in generalised coordinates.
Four files, 1581 lines, one build option (`multiscale` — there is no
alternative to select):

```
hdiff/multiscale/hdiff.F      606   driver: RK11/RK22 faces, sub-step loop, update
hdiff/multiscale/hcdiff3d.F   256   *** eddy diffusivity from deformation ***
hdiff/multiscale/deform.F     439   *** wind deformation from UHAT_JD/VHAT_JD ***
hdiff/multiscale/rho_j.F      280   rho*J on the computational grid
```

Verified identical on `origin/5.5+` and `main` (`git diff main origin/5.5+ --
CCTM/src/hdiff/` is empty), same as advection was.

## The numerics

Much simpler than PPM: an **explicit 5-point Laplacian in mixing-ratio space**,
sub-cycled for stability. No limiter, no reconstruction, no iteration.

**1. Deformation** (`deform.F:424-432`) — the standard Smagorinsky total
deformation, stretching and shearing combined:

```
DF1 = du/dx - dv/dy          DF2 = dv/dx + du/dy
DEFORM = sqrt(DF1^2 + DF2^2)
```

Winds are `UHAT_JD`/`VHAT_JD` divided by face-interpolated density — the same
quantity `hcontvel.F` builds, so `cmaq_jax.velocity.face_velocity_from_flux`
already covers it. The cross terms use gradients *of averages*
(`deform.F:396-418`), not averages of gradients, and are set to zero on the
first and last row/column.

**2. Eddy diffusivity** (`hcdiff3d.F:188-200`):

```
KHA  = (DXB^2)/(dx1*dx2) * KH        resolution-adjusted base, KH = 2000 m^2/s
ACOEF = ALP^2 * dx1 * dx2            ALP = 0.28
KHD  = max(KHMIN, ACOEF * DEFORM)    KHMIN = 200 m^2/s
EDDYH = MSFD2 * KHA*KHD / (KHA + KHD)
```

The last line is a parallel-resistor blend: it rises with deformation but
saturates at `KHA`, so a strongly sheared cell cannot run away.

Then flux-averaged onto faces (`hcdiff3d.F:210-230`), each averaging *across*
its own direction:

```
K11BAR(C,R) = 0.5*(EDDYH(C,R+1) + EDDYH(C,R))     x faces
K22BAR(C,R) = 0.5*(EDDYH(C,R)   + EDDYH(C+1,R))   y faces
```

**3. Stability step** (`hcdiff3d.F:253`) — `DT = CFC * dx1*dx2 / max(K)`,
`CFC = 0.300`. `hdiff.F` then takes `NSTEPS = int(DTSEC/DT) + 1` uniform
sub-steps.

**4. Update** (`hdiff.F:509-523`), per species, layer and sub-step, with
`RK11 = 0.5*(rhoJ(C) + rhoJ(C-1)) * K11BAR` and `q = CGRID/rhoJ`:

```
CGRID(C,R) = rhoJ(C,R)*q(C,R)
           + dt/dx1^2 * ( RK11(C+1,R)*(q(C+1,R) - q(C,R)) - RK11(C,R)*(q(C,R) - q(C-1,R)) )
           + dt/dx2^2 * ( RK22(C,R+1)*(q(C,R+1) - q(C,R)) - RK22(C,R)*(q(C,R) - q(C,R-1)) )
```

Diffusion acts on the **mixing ratio**, and the flux carries `rho*J` — which is
why `rho*J` is *not* advected along as an extra slot here, unlike in advection.
That is the single biggest structural difference from the `A` chunks and the
thing most likely to be got wrong by analogy.

## Two things found while reading, before writing any code

**The deformation's extra row and column are defined as zero, deliberately.**
`hcdiff3d.F` declares `DEFORM3D` as `(NCOLS+1, NROWS+1, NLAYS)` while
`deform.F` computes values only on `(1:NCOLS, 1:NROWS)` — but `deform.F:337-343`
zeroes the full extent first, commented "deformation at all boundary cells are
defined to be zero". So the last row and column of `K11BAR`/`K22BAR` are built
from zeros, and that is documented intent rather than an accident.

This is worth stating because the arithmetic makes it easy to assume otherwise:
`EDDYH3D` is *not* zero there — it is `MSFD2 * KHA*KHMIN/(KHA+KHMIN)`, since
`KHD = max(KHMIN, ACOEF*0) = KHMIN` — so the boundary coefficient is a nonzero
floor value, not zero. `hcdiff3d.F:216,226` then explicitly zeroes
`K11BAR(:,NROWS+1)` and `K22BAR(NCOLS+1,:)`, which is a *different* edge from
the one the deformation zeroing covers. The port has to reproduce both, and they
are easy to conflate.

**The halo is frozen across sub-steps.** `HALO_SOUTH`/`NORTH`/`WEST`/`EAST` are
filled once, before the `DO 344` sub-step loop (`hdiff.F:355-400`), from the
*initial* mixing ratio; `CONC` is reloaded from `CGRID` every sub-step but the
halo is not. So the zero-gradient boundary holds exactly only on the first
sub-step, and drifts slightly after. This is deliberate — the 2009 revision note
at `hdiff.F:66` records fixing a related sub-cycling bug — and it is a
behavioural detail a "clean" rewrite would silently change, so it is ported as
written and pinned by a test.

## A third thing, found while making the figures

**CMAQ's diffusion sub-step is past the scheme's stability limit whenever
sub-stepping engages.** `hcdiff3d.F:253` sets `DT = CFC·dx1·dx2/max(K)` with
`CFC = 0.300`, while an explicit five-point Laplacian needs `r = K·dt/dx² ≤ 0.25`.
Since `NSTEPS = int(DTSEC/DT) + 1`, once sub-stepping engages `dt → DT` and
`r → CFC = 0.300`, and the grid-scale mode grows by `|1 − 8r| = 1.4` per
sub-step.

Verified against the Fortran: `hdiff.F` compiled unmodified produces the same
blow-up on the same inputs, to the same values, so this is CMAQ's behaviour and
not a porting artefact.

It does not affect CMAQ's own configurations. At 12 km the stable step is
~2×10⁵ s, `NSTEPS` is 1, and `dt` is the sync step — far below the limit. It also
needs an extended region of near-maximal `K`: the 1 km and 500 m golden cases are
formally past the limit yet stay bounded, because a 7×6 domain with one hot spot
gives the unstable mode nowhere to grow. Both conditions have to hold at once,
which is presumably why it has gone unremarked.

The port reproduces it rather than correcting it, on the same principle as the
frozen halo. `docs/figures/b2/substep_stability.png` shows where the boundary is.

## Execution order

| Phase | Subplan | Gate |
|---|---|---|
| **B0** | [`subplans/B0-foundation.md`](subplans/B0-foundation.md) | Vendored Fortran compiles; goldens for all three kernels |
| **B1** | [`subplans/B1-kernels.md`](subplans/B1-kernels.md) | `deform`, `eddy_diffusivity`, face coefficients match goldens |
| **B2** | [`subplans/B2-driver.md`](subplans/B2-driver.md) | `hdiff_step` matches; properties hold; wired into `advect_step`'s sibling |

## Deliberate deviations

Same list as advection, plus:

- ISAM and DDM-3D (`#ifdef sens`) branches omitted — they are ~40% of
  `hdiff.F`'s line count and duplicate the same update for sensitivity arrays.
- `MSFD2` (map scale factor at dot points) is read from `GRID_DOT_2D`; on the
  Lambert benchmark grid it is ~1, but it is carried rather than assumed.

## Why this is much smaller than advection

No reference-harness invention is needed: `reference/Makefile`, the stub layer,
the one-process-per-configuration rule and `generate_goldens.py --check` all
came out of `A0` and are reused as they stand. The numerics are ~120 lines
against PPM's ~350, with no limiter and no iteration. The realistic risk is not
the scheme, it is the two structural traps above.
