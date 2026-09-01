# Ultraplan — CMAQ on GPU via JAX

## The goal

Run CMAQ's science on GPUs, differentiably, without a Fortran toolchain.

CMAQ (Community Multiscale Air Quality model) is ~500k lines of Fortran built
around MPI domain decomposition. A wholesale rewrite is not the plan. The plan
is to port **one operator at a time**, each validated against the Fortran it
replaces, so that at every point there is a working, tested artifact rather than
a half-finished model.

## Why JAX rather than CUDA Fortran or OpenACC

- `jax.grad` through the transport and chemistry operators → parameter
  estimation, adjoint sensitivity, and data assimilation without hand-writing an
  adjoint. CMAQ's existing DDM-3D and adjoint variants are separate, partially
  maintained code paths; in JAX the derivative is free.
- JIT + device-agnostic execution: one source runs on laptop CPU, workstation
  GPU, and multi-node clusters via `shard_map`.
- Composability with the sibling ports (`som-jax`, and the planned `saprc-jax`,
  `tomas-jax`) that already share `atmos-jax-common`.

## Operator order, and why

CMAQ's science driver (`sciproc.F`) runs, per sync step:

```
VDIFF → COUPLE → HADV → ZADV → HDIFF → DECOUPLE → PHOT → CLDPROC → CHEM → AERO
```

We port in order of *isolation*, not execution order — cleanest interfaces first,
so each port is validatable on its own.

| # | Operator | Repo/module | Why this position | Status |
|---|---|---|---|---|
| 1 | **Advection** (HADV + ZADV) | `cmaq-jax` | Two self-contained PPM kernels; ~350 lines of real numerics; no chemistry coupling; analytic test cases exist | **done** (`A0`–`A3`) |
| 2 | Horizontal diffusion | `cmaq-jax` | Small, shares the halo machinery advection builds | **done** (`B0`–`B2`) |
| 3 | Vertical diffusion (ACM2) | `cmaq-jax` | Implicit solve; introduces a tridiagonal solver; couples to deposition | **done** (`C0`–`C3`) |
| 4 | Gas chemistry | `saprc-jax` (planned) | Stiff ODE; `som-jax` already proves the diffrax approach | **next** |
| 5 | Aerosol | `tomas-jax` (planned) | Largest and most coupled; last | not started |

Advection first is the deliberate choice: it is the only operator whose kernels
compile standalone with `gfortran` and no I/O API, which means the reference
harness is cheap and the validation is airtight.

## Explicitly out of scope for now

Named so they don't creep in:

- **ISAM** (source apportionment) and **DDM-3D** (`#ifdef sens`) branches — JAX
  autodiff supersedes DDM-3D; ISAM is a separate feature.
- **Process analysis / IPR budget** diagnostics.
- **Plume-in-grid**, **two-way WRF-CMAQ coupling**, **MPAS** grids.
- **I/O API** file formats as a runtime dependency. Meteorology is read once
  into arrays via `xarray`; nothing downstream knows about I/O API.
- **Nested grids** and the `local_cons` legacy vertical advection option.

## Shared infrastructure

`atmos-jax-common` holds the Fortran-reference plumbing every port needs:
`real4` (float64↔float32 downcast for faithful comparison), `compare`
(tolerance-aware diff primitives), `fortran_runner` (build/run wrappers), and
the committed-goldens-with-drift-check pattern. New generic plumbing belongs
there, not here.

## Completed

- [`plans/PLAN-advection.md`](plans/PLAN-advection.md) — `A0.1` … `A3.7`.
- [`plans/PLAN-hdiff.md`](plans/PLAN-hdiff.md) — `B0.1` … `B2.5`.
- [`plans/PLAN-vdiff.md`](plans/PLAN-vdiff.md) — `C0.1` … `C3.4`.

`cmaq_jax.api.science_step` runs VDIFF → HADV → ZADV → HDIFF, in `sciproc.F`'s
order. Every kernel is matched against unmodified Fortran in both precisions.

**Open verification gap:** `io_mcip` is tested against synthetic I/O API files,
not real MCIP output. Closing it needs a `$CMAQ_DATA` download.

**Not ported, deliberately:** coupling/decoupling (a unit conversion, not
transport), deposition-velocity calculation (`depv/*`), aerosol sedimentation,
ISAM and DDM-3D.

## Three upstream findings

All reproduced rather than corrected, all verified against unmodified Fortran,
and none of them affecting CMAQ's own benchmark configurations. Each is
documented where it lives.

1. **`hdiff.F`'s halo is frozen across sub-steps**, making it a Dirichlet
   condition pinned at t=0 rather than the no-flux condition it is described as.
   Mass flows toward the pinned value — measured at −0.12% to +8.9% depending on
   the field.
2. **`hdiff.F`'s diffusion sub-step (`CFC = 0.300`) sits past the explicit
   scheme's stability limit of 0.25** whenever sub-stepping engages. Needs a
   fine grid *and* an extended region of near-maximal `K`.
3. **`vdiffacmx.F`'s local stage leaks through the model top** — the top row's
   diagonal carries an upward-flux term with no matching right-hand side. It is
   multiplied by zero only because `eddyx.F` returns zero diffusivity there, so
   the scheme is conservative by coincidence of two facts rather than by
   construction.

## Next

Gas chemistry, in a new repo `saprc-jax`. A stiff ODE system rather than a PDE
operator — a different kind of problem from everything above, and the one
`som-jax` has already proven the `diffrax` approach on.
