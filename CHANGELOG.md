# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A0.1** — Repository scaffold: package layout, `pyproject.toml` (ruff +
  mypy strict + pytest), CI workflow, MIT license, `CLAUDE.md` working
  agreement.
- **A0.1** — CMAQ Fortran reference vendored verbatim from branch `5.5+`
  @ `b8e4303` under `reference/fortran/`, with per-file sha256 in
  `reference/PROVENANCE.md` and a `scripts/vendor_reference.sh` to re-vendor.
- **A0.1** — Plan hierarchy: `docs/ULTRAPLAN.md` (the CMAQ→JAX arc),
  `docs/plans/PLAN-advection.md`, and phase subplans `A0`–`A3`.
- **A0.2** — `cmaq_jax.config`: `PPMConstants` and `GridConfig`, every field
  citing the Fortran file and line it came from. Includes
  `sigma_layer_thickness()` porting `DS(L) = ABS(X3FACE_GD(L) - X3FACE_GD(L-1))`.
  Constants used only by `advstep` are deliberately deferred to A3.1 rather than
  guessed.

- **A0.3** — Fortran golden harness: `reference/harness/stubs.f90` supplies the
  handful of CMAQ symbols the kernels need (`M3EXIT`, `BUDGET_HPPM`,
  `SUBST_HI_LO_BND_PE`, the `CGRID_SPCS` counts), so `hppm.F` and `vppm.F`
  compile **unmodified** with no I/O API, netCDF, or MPI.
- **A0.4** — `scripts/generate_goldens.py` with a `--check` drift mode, and 17
  committed goldens under `data/goldens/` (10 HPPM, 7 VPPM). Each case runs in
  its own process to avoid the kernels' `SAVE`d first-call array sizing.
  `tests/regression/test_goldens.py` guards the harness itself: shapes,
  positivity, monotonicity, an untouched halo, and constancy preservation.
- **A0.5/A0.7** — `cmaq_jax.ppm`: the uniform-spacing PPM sweep
  (`ppm_parabola_uniform`, `ppm_flux_uniform`, `ppm_advect_uniform`), porting
  `hppm.F:283-445`. Written as whole-array slices with the sweep axis first, so
  a sweep is one fused kernel over all rows/layers/species; every Fortran `IF`
  is a branchless `jnp.where`. Matches all 10 HPPM goldens to a worst-case
  2.1e-7 relative (~1.7 float32 ULPs), with three cases bit-identical.
- **A0.5** — `jax_enable_x64` is set on package import, since float64 is the
  documented working precision and forgetting the flag silently halves it.

### Changed

- mypy `python_version` set to 3.12: numpy >= 2.5 ships PEP 695 `type`
  statements in its stubs that mypy cannot parse under 3.11. Runtime support for
  3.11 is unaffected and stays covered by the pytest matrix.
