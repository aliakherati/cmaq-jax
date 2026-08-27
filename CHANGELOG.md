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
- **A0.6** — `cmaq_jax.ppm`: the non-uniform-spacing reconstruction
  (`nonuniform_mesh`, `ppm_parabola_nonuniform`), porting `vppm.F:450-541`.
  Mesh coefficients are precomputed from `ds`, mirroring the Fortran's `SAVE`.
- **A0.6** — `reference/harness/harness_ppm_coeffs.f90` calls `vppm.F`'s inner
  `PPM` subroutine directly, pinning the parabola independently of the velocity
  adjustment. Adds 14 `coeffs_*` goldens (7 profiles x uniform/stretched grids).
- **A0.8** — `tests/properties/`: the guarantees the golden comparison cannot
  check — ~2nd-order convergence against an analytic Gaussian, exact mass
  conservation for a compact feature, no new extrema across a discontinuity,
  positivity under a divergent wind, and constancy preservation over 80 steps.
- **A0.9** — `scripts/make_a0_figures.py` and five figures under
  `docs/figures/a0/`.
- **A0.9** — `scripts/make_a0_flow_figures.py`: solid-body rotation (Zalesak)
  and deformational swirl (LeVeque) as static panels, a 3-D surface view, and
  two animated GIFs. Measured on the rotation: phase error 0.003 cells, slot
  23% filled, cone peak −8%, mass conserved to 7e-13, undershoot 1e-32.
- **A1.1** — `cmaq_jax.bc.zfdbc`: branchless zero-flux-divergence outflow
  boundary condition, porting `zfdbc.f`. Validated against a new Fortran harness
  over 2,726 cases covering all three branches (198+1 small-wind, 1,330
  diverging, 1,197 extrapolating, of which 296 hit the zero clamp).
- **A1.2** — `cmaq_jax.bc.fill_halo`: ghost-cell fill, BCON on inflow and
  `zfdbc` on outflow per edge, porting `x_ppm.F:418-441`. The halo stays a
  first-class array region so `shard_map` can later swap the local fill for a
  collective permute without touching the kernels.
- **A1** — The horizontal-advection *driver* is now golden-tested. A new
  `harness_hadv` runs `hadvppm.F` -> `x_ppm.F`/`y_ppm.F` -> `hcontvel.F` ->
  `hppm.F` unmodified, with only the data *sources* replaced: a stub
  `interpolate_var` reads a table the harness fills, and a stub `RDBCON` returns
  a preloaded boundary field. 8 `hadv_*` goldens cover per-layer sub-stepping,
  the X-Y/Y-X alternation, all-inflow and all-outflow boundaries, and constancy
  under a divergent wind.
- **A1** — Vendored CMAQ's serial no-op stencil-exchange layer
  (`reference/stenex/`) and the `INCLUDE SUBST_*` header files
  (`reference/include/`), so halo-exchange calls resolve to real upstream code
  rather than hand-written stubs.
- **A1.3** — `cmaq_jax.velocity`: face velocity for a sweep. The C-staggered
  path returns the wind unchanged (`hcontvel.F` RETURNs early when `CSTAGUV` is
  true, the default since MCIP v3.5); the pre-2009 density-weighted fallback is
  available by passing `rhoj`.
- **A1.4** — `cmaq_jax.hadv.sweep`: one axis-generic PPM sweep of the whole
  grid, replacing the two near-duplicate 660-line files `x_ppm.F`/`y_ppm.F`.
- **A1.5** — `cmaq_jax.hadv.hadv`: the driver. Per-layer sub-stepping and the
  X-Y/Y-X alternation, with layers statically grouped by
  `(sub-step count, starting order)` so each group is a plain loop of sweeps —
  no masking, no `lax.while_loop`. Matches all 8 driver goldens to under one
  float32 ULP (worst case 1.05e-7).
- **A1.6** — `tests/properties/test_hadv_properties.py`: scheme properties
  through the real driver rather than the bare kernel — solid-body rotation,
  inflow/outflow behaviour, constancy over 60 steps, and layer independence
  under mixed `ASTEP`.
- **A1.7** — `scripts/make_a1_figures.py` and `docs/figures/a1/`: the rotation
  benchmark re-run through `hadv_step`, plus a side-by-side against the A0
  periodic-halo version showing they are bit-identical at first and diverge to
  2.5e-8 over a full turn.
- **A2.2/A2.3** — `cmaq_jax.vadv`: the vertical column solve and its
  flux-matching velocity adjustment, porting `vppm.F`. Matches all 7 `vppm`
  goldens — concentrations to 0.4 float32 ULPs, adjusted velocities to 1.9 —
  in both precisions.
- **A2.1/A2.4** — `cmaq_jax.vadv`: the flux diagnosis from the rho*J budget and
  the per-column CFL sub-stepping, porting `zadvppmwrf.F`. A new `harness_zadv`
  golden-tests the whole vertical chain; 7 `zadv_*` goldens cover both CFL
  regimes (Courant 0.08 to 2.25, one to three sub-steps). Agreement with the
  Fortran: worst case 1.6 float32 ULPs, in both precisions.
- **A2.5/A2.6** — vertical property tests and figures. Column mass is **exactly**
  conserved: the column is closed at both ends, the top because
  `FLX(top) = DRJ*(1 - sum(ds))` and the sigma thicknesses sum to one.
- **A3.2** — `cmaq_jax.api.advect_step`: the `HADV -> ZADV` pair as `sciproc.F`
  invokes it, with a `Meteorology` bundle and diagnostics carrying the
  alternation state and vertical convergence.
- **A3.4** — `tests/differentiability/`: `jax.grad` through the full 3-D
  operator, checked against central finite differences (worst 5.4e-6, which is
  the finite-difference truncation error). Required fixing four sites where a
  masked branch still poisoned the reverse pass.
- **A3.1** — `cmaq_jax.advstep`: the CFL and divergence limits that choose the
  sync step and each layer's advection step, including the `SIGMA_SYNC_TOP`
  split that keeps a jet aloft from slowing the whole model.
- **A3.3** — `tests/properties/test_end_to_end.py`: the two operators together,
  over many sync steps, on a schedule `advstep` chose.
- **A3.6** — `python -m cmaq_jax.bench`: timing at benchmark resolution, with
  the vertical sub-step cap broken out as the main cost lever.

### Changed

- **Precision** — `docs/figures/a1/precision.png`: agreement with the Fortran in
  both precisions across all four golden families, showing float32 at or below
  float64 nearly everywhere and bit-identical on several coefficient cases.
- **Precision** — `GridConfig.dtype` is now wired up. It was declared in A0.2
  and documented in the README, but nothing read it: the port was float64-only
  and float32 appeared solely as a comparison target. `hadv_step` now casts its
  array inputs to `cfg.dtype`, and every golden comparison is parametrized over
  both precisions (`f32`/`f64` test ids). Agreement with the Fortran, worst case
  in float32 ULPs: hppm 0.4, zfdbc 0.8, hadv driver 0.9, PPM coefficients 4.8
  (the `c6` cancellation). Invariants — positivity, monotonicity, ρ·J, and
  constancy — are checked in float32 as well, where constancy holds to ~7 ULP
  over 40 steps.

- `hadv` split into `hadv_step` (pure, jittable) and `advance_xyfirst`
  (host-side flag bookkeeping). Returning both from one function was what
  blocked `jax.jit`; wrapping the split version gives a **1134x** speed-up
  (66.5 ms/step to 0.06 ms/step), the whole difference being dispatch overhead.
  `xyfirst` is now a `tuple[bool, ...]` rather than an ndarray, since it is
  control state rather than data.

- mypy `python_version` set to 3.12: numpy >= 2.5 ships PEP 695 `type`
  statements in its stubs that mypy cannot parse under 3.11. Runtime support for
  3.11 is unaffected and stays covered by the pytest matrix.
