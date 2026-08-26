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
| 1 | **Advection** (HADV + ZADV) | `cmaq-jax` | Two self-contained PPM kernels; ~350 lines of real numerics; no chemistry coupling; analytic test cases exist | **in progress** |
| 2 | Horizontal diffusion | `cmaq-jax` | Small, shares the halo machinery advection builds | not started |
| 3 | Vertical diffusion (ACM2) | `cmaq-jax` | Implicit solve; introduces a tridiagonal solver; couples to deposition | not started |
| 4 | Gas chemistry | `saprc-jax` (planned) | Stiff ODE; `som-jax` already proves the diffrax approach | not started |
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

## Current project

[`plans/PLAN-advection.md`](plans/PLAN-advection.md) — chunks `A0.1` … `A3.x`.
