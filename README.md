# cmaq-jax

Differentiable JAX port of the **CMAQ advection core** — the piecewise parabolic
method (PPM) horizontal and vertical transport operators from the Community
Multiscale Air Quality model.

**Why a port?** JAX gives us:

- `jax.grad` through the transport operator → adjoint sensitivity and data
  assimilation without hand-writing an adjoint (CMAQ's DDM-3D is a separate,
  partially maintained code path).
- JIT + device-agnostic execution: one source on laptop CPU, workstation GPU,
  and multi-node clusters via `shard_map`.
- Composability with the sibling ports ([`som-jax`](https://github.com/aliakherati/som-jax),
  and the planned `saprc-jax`, `tomas-jax`) that share
  [`atmos-jax-common`](https://github.com/aliakherati/atmos-jax-common).

## Status

**Phase A0 complete.** The two 1-D PPM kernels are ported and validated against
the CMAQ Fortran; the golden harness and property suite are in place. Next is
A1 (horizontal advection: boundary conditions, contravariant velocity, sweeps).

Agreement with CMAQ, after downcasting to float32: the uniform-spacing sweep
matches `hppm.F` to a worst case of 1.7 float32 ULPs across ten cases, three of
them bit-identical. The non-uniform reconstruction matches `vppm.F`'s inner
`PPM` to 0.8 / 0.8 / 1.6 / 4.6 ULPs for `cl` / `cr` / `dc` / `c6`.

## Scope

The complete advection operator as CMAQ's `sciproc.F` invokes it: **HADV**
(horizontal PPM, X and Y sweeps) and **ZADV** (vertical PPM with
mass-continuity-derived velocity), including the ρ·J ride-along that conserves
mass.

Target configuration is the CMAQ v5.5 default, `ModAdv = wrf_cons`. `hadv/ppm`
is the only horizontal scheme remaining in v5.5 — the `yamo` variants were
removed in 2019.

Out of scope: ISAM, DDM-3D, process analysis, plume-in-grid, two-way WRF-CMAQ,
MPAS grids, nested grids. See [`docs/ULTRAPLAN.md`](docs/ULTRAPLAN.md).

## Real-field California validation

[`examples/conus404/`](examples/conus404/README.md) runs the complete HADV→ZADV
pair over California with native 4 km, 50-layer CONUS404 meteorology and daily
FINN fire CO.  It includes conservative 8 km comparison, positivity and mass
diagnostics, `rhoJ` closure, vertical-centroid diagnostics, static summaries,
and animated plume GIFs.  The older five-level NARR example remains a visual
smoke test and is not used as physical validation.

## Full-CONUS projected-2023 validation

[`examples/epa_2023/`](examples/epa_2023/README.md) runs a 24-hour inert-CO
case on EPA's complete 459 × 299 cell `12US1` domain at 12 km.  It combines all
CO in the platform's final hourly merged gridded file with matching real MCIP
meteorology on 35 layers, saves exact 15-minute states, and produces polished
column and lowest-layer GIFs.  EPA's `2023gf` label denotes projected-2023
emissions evaluated with 2016 meteorology; the example keeps both dates visible.

## Install

```bash
git clone https://github.com/aliakherati/cmaq-jax.git
cd cmaq-jax
uv venv && uv pip install -e ".[dev]"
```

Requires Python ≥ 3.11. Regenerating Fortran goldens additionally needs
`gfortran`.

## Project status

| Module | Purpose | Chunk | Status |
|---|---|---|---|
| repo scaffold, vendored Fortran, plan hierarchy | — | A0.1 | alpha |
| `cmaq_jax.config` | `GridConfig`, `PPMConstants` — every constant traced to its Fortran line | A0.2 | alpha |
| Fortran golden harness | compile `hppm.F`/`vppm.F` unmodified, emit reference arrays | A0.3–A0.4 | alpha |
| `cmaq_jax.ppm` | 1-D PPM, uniform spacing: parabola, upwind flux, conservative update | A0.5, A0.7 | alpha |
| `cmaq_jax.ppm` | 1-D PPM, non-uniform spacing (vertical): mesh coefficients + parabola | A0.6 | alpha |
| `cmaq_jax.bc` | `zfdbc` outflow BC, width-3 halo fill | A1.1–A1.2 | alpha |
| `cmaq_jax.velocity` | face velocity: C-staggered pass-through, legacy density-weighted fallback | A1.3 | alpha |
| `cmaq_jax.hadv` | axis-generic sweep, layer grouping, X-Y/Y-X alternation | A1.4–A1.5 | alpha |
| `cmaq_jax.vadv` | column solve, flux-matching velocity adjustment, flux diagnosis, CFL sub-stepping | A2.1–A2.4 | alpha |
| `cmaq_jax.advstep` | CFL and divergence limits → sync step + per-layer `ASTEP` | A3.1 | alpha |
| `cmaq_jax.api` | `advect_step` — HADV then ZADV, jittable and differentiable | A3.2 | alpha |
| `cmaq_jax.io_mcip` | MCIP/IOAPI meteorology reader | A3.5 | ✅ tested on synthetic IOAPI and EPA 12US1 MCIP output |
| `cmaq_jax.hdiff` | Horizontal diffusion: deformation, eddy diffusivity, driver | B0–B2 | ✅ matches `deform.F`/`hcdiff3d.F`/`hdiff.F` |
| `cmaq_jax.vdiff` | ACM2 vertical diffusion: solvers, `Kz`, driver | C0–C2 | ✅ matches `tri.F`/`matrix1.F`/`eddyx.F`/`vdiffacmx.F` |
| `cmaq_jax.bench` | CPU/GPU timing at benchmark resolution | A3.6 | alpha |

Tracked in [`docs/plans/PLAN-advection.md`](docs/plans/PLAN-advection.md) as
chunks `A0.1` … `A3.7`.

### Figures

Each scientific chunk ships figures under `docs/figures/<chunk-id>/`,
regenerable via `scripts/make_<chunk>_figures.py`.

**A0 — the PPM kernels**

| File | What it shows |
|---|---|
| [`fortran_agreement.png`](docs/figures/a0/fortran_agreement.png) | Per-case disagreement between the JAX port and `hppm.F`, against the float32 epsilon line. Worst case is ~1.7 float32 ULPs and three of ten cases are bit-identical, so the residual is precision, not algorithm. |
| [`convergence.png`](docs/figures/a0/convergence.png) | Mean absolute error against grid refinement for a Gaussian advected a quarter domain, annotated with the measured order at each step and bracketed by 2nd- and 3rd-order reference slopes. Lands near 2: the limiter clips the peak, which is the correct behaviour. |
| [`limiter_action.png`](docs/figures/a0/limiter_action.png) | The reconstructed parabola in every cell for a smooth profile, the same profile with a step, and with a spike. Circles mark cells the limiter collapsed to a constant — 4/40 for smooth, 6/40 when a discontinuity is added, showing it fires locally rather than everywhere. |
| [`transport.png`](docs/figures/a0/transport.png) | A Gaussian and a square wave after one full revolution (400 steps). The Gaussian loses 0.9% of its peak to numerical diffusion; the square wave keeps full amplitude with no undershoot below zero or overshoot above one. |
| [`vertical_stretching.png`](docs/figures/a0/vertical_stretching.png) | The non-uniform reconstruction on 35 CMAQ-like sigma layers, 6× thicker aloft than at the surface, with the resulting edge values, slope and curvature. |

**A3 — the assembled operator**

Described in [`docs/figures/a3/README.md`](docs/figures/a3/README.md).

| File | What it shows |
|---|---|
| [`adjoint_footprint.png`](docs/figures/a3/adjoint_footprint.png) | One reverse pass answers a source-receptor question — which upwind cells the concentration at a chosen cell came from. No adjoint was written; this is the forward code differentiated. Checked against finite differences where that check is itself reliable. |
| [`scaling.png`](docs/figures/a3/scaling.png) | Cost against domain size, species count and the vertical sub-step cap. Throughput improves with species count as fixed overhead amortises. |

**A2 — vertical advection**

Described in [`docs/figures/a2/README.md`](docs/figures/a2/README.md).

| File | What it shows |
|---|---|
| [`flux_diagnosis.png`](docs/figures/a2/flux_diagnosis.png) | There is no vertical wind to read — the flux is diagnosed from the density mismatch. The ground is pinned closed; the model top closes itself, because the sigma thicknesses sum to one and the flux recurrence cancels there. |
| [`vertical_transport.png`](docs/figures/a2/vertical_transport.png) | One sync step, and what repeating it converges to. The density gap stops at exactly the column-mean mismatch — vertical flux conserves column mass, so it can redistribute an offset but never remove one. |
| [`substepping.png`](docs/figures/a2/substepping.png) | Mismatch drives Courant, Courant drives sub-steps, and a grid of columns needing one to several sub-steps advances together in one masked fixed-count loop. |

**A1 — through the real driver**

Described in [`docs/figures/a1/README.md`](docs/figures/a1/README.md).
Regenerate with `python scripts/make_a1_figures.py`.

| File | What it shows |
|---|---|
| [`rotation_driver.png`](docs/figures/a1/rotation_driver.png) | Solid-body rotation through `hadv_step`, with boundary conditions, the X-Y/Y-X alternation and the ρ·J ride-along all active. Phase error 0.006 cells, mass conserved to 1e-10, ρ·J held at 1.0 with zero drift under a discretely non-divergent wind. |
| [`precision.png`](docs/figures/a1/precision.png) | Agreement with the Fortran in **both** working precisions across all four golden families. Native float32 is frequently closer to the reference than float64-then-downcast, since it does the same arithmetic in the same precision. Worst case anywhere: 4.8 float32 ULPs. |
| [`periodic_vs_driver.png`](docs/figures/a1/periodic_vs_driver.png) | The same rotation under a periodic halo and under the driver's real boundaries. Bit-identical for the first few steps, diverging to 2.5e-8 over a full turn once diffusion spreads a tail to the edge — so the A0 figures were not misleading about the interior, and this is the scale of what they left out. |

**A0 — what the scheme does to a field**

Each figure is described in detail, with what to look for, in
[`docs/figures/a0/README.md`](docs/figures/a0/README.md).

The two benchmarks anyone working with transport schemes will recognise, run
with the A0 kernel applied alternately along each axis. Regenerate with
`python scripts/make_a0_flow_figures.py`.

| File | What it shows |
|---|---|
| [`rotation.gif`](docs/figures/a0/rotation.gif) | Solid-body rotation of Zalesak's cone and slotted cylinder, animated through a full turn with a live min/max readout. The slot narrows but never closes and no ringing appears. |
| [`deformation.gif`](docs/figures/a0/deformation.gif) | LeVeque's swirl winding the blob into a filament and unwinding it after the flow reverses at the half-period. |
| [`rotation_2d.png`](docs/figures/a0/rotation_2d.png) | The same rotation at each quarter turn plus an error map. Phase error is 0.003 cells — the shapes return where they started, so the error is edge diffusion, not displacement. The slot fills 23%, the cone loses 8% of its peak, the cylinder plateau holds at 1.000, and mass is conserved to 7e-13. |
| [`rotation_3d.png`](docs/figures/a0/rotation_3d.png) | The rotation as a surface, before and after. The cylinder keeps vertical sides and a flat top; an unlimited high-order scheme would ring around every edge. |
| [`deformation_2d.png`](docs/figures/a0/deformation_2d.png) | The swirl at four stages, the error after return, and a trace of peak amplitude and Σc² against time. Both fall and never recover — that is irreversible mixing — while mass holds to machine precision. |

## The Fortran reference

[`reference/fortran/`](reference/fortran) holds the CMAQ sources verbatim from
branch `5.5+` @ `b8e4303` — see [`reference/PROVENANCE.md`](reference/PROVENANCE.md).
These files are **never edited**; they are what the port is validated against.

Only `hppm.F` (477 lines, uniform-spacing PPM) and `vppm.F` (544 lines,
non-uniform-spacing PPM) are compiled, into a standalone golden harness. Their
dependency surface is small enough — `M3EXIT`, a few flags, and one cpp macro —
that no I/O API, netCDF, or MPI toolchain is needed.

> The advection sources are **identical** between `main` and `5.5+`;
> `git diff main 5.5+ -- CCTM/src/{hadv,vadv,couple,grid,spcs}` is empty. `5.5+`
> is used for provenance correctness, not because it changes the numerics.

## Deliberate deviations from the Fortran

Recorded so no numerical difference is ever silent.

| Deviation | Reason |
|---|---|
| float64 default (Fortran is float32) | Accuracy. float32 is a fully supported compute path, not just a comparison target: `GridConfig.dtype` selects it and `hadv_step` casts to it. **Every golden comparison runs in both precisions.** Native float32 often agrees with the Fortran *more* closely than float64-then-downcast, since it does the same arithmetic in the same precision. |
| `FBLN` blend term dropped | Hard-set to `1.0` upstream (`zadvppmwrf.F:249`, sigmoid commented out) — the term is identically a no-op. |
| Fixed-count loops replace unbounded `GO TO` loops | Data-dependent trip counts don't jit. Non-convergence is reported as a residual instead of `M3EXIT`. |
| ISAM / DDM-3D / IPR-budget branches omitted | Out of scope; JAX autodiff supersedes DDM-3D. |
| MPI halo exchange removed, halo *abstraction* kept | Single-device today; `shard_map` drops in later without touching the kernels. |

## Testing

```bash
python scripts/generate_goldens.py          # build Fortran + emit goldens
python scripts/generate_goldens.py --check  # drift detection (CI)

pytest tests/unit tests/regression -v       # kernel agreement with Fortran
pytest tests/properties -v                  # monotonicity, positivity, conservation
pytest tests/differentiability -v           # jax.grad through advection
ruff check . && mypy --strict src/
```

Validation is two-track: **goldens** from the real Fortran pin the numerics, and
**property tests** pin the scheme's mathematical guarantees — monotonicity,
positivity, mass conservation, and constancy preservation under a divergent
wind. The last of these is the CMAQ-specific one, and the most likely to catch a
coupling error.

## Development

See [`CLAUDE.md`](CLAUDE.md) for the working agreement: plan hierarchy, one
chunk per commit, constants live in `config.py`, and the vendored Fortran is
read-only.
