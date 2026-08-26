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

**Early alpha — scaffolding.** Repository created, Fortran reference vendored,
plan hierarchy in place. No public API yet; modules land chunk by chunk.

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
| `cmaq_jax.ppm` | 1-D PPM, non-uniform spacing (vertical) | A0.6 | not started |
| `cmaq_jax.bc` | `zfdbc` outflow BC, width-3 halo fill | A1.1–A1.2 | not started |
| `cmaq_jax.velocity` | contravariant velocity from `UHAT_JD` / `DENSA_J` | A1.3 | not started |
| `cmaq_jax.hadv` | axis-generic sweep, layer grouping, X-Y/Y-X alternation | A1.4–A1.5 | not started |
| `cmaq_jax.vadv` | flux/velocity diagnosis, velocity adjustment, CFL sub-stepping | A2.1–A2.4 | not started |
| `cmaq_jax.advstep` | CFL analysis → sync step + per-layer `ASTEP` | A3.1 | not started |
| `cmaq_jax.api` | `advect_step(state, met, cfg)` | A3.2 | not started |
| `cmaq_jax.io_mcip` | MCIP/IOAPI meteorology reader | A3.5 | blocked on met data |

Tracked in [`docs/plans/PLAN-advection.md`](docs/plans/PLAN-advection.md) as
chunks `A0.1` … `A3.7`.

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
| float64 default (Fortran is float32) | Accuracy; float32 selectable via `GridConfig.dtype`. Golden comparison downcasts via `atmos_jax_common.real4`. |
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
