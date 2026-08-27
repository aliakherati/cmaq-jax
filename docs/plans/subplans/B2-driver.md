# B2 — Driver

Parent: [`../PLAN-hdiff.md`](../PLAN-hdiff.md) · Depends on B1

Assemble the operator, prove its properties, and put it where `sciproc.F` does.

**Gate:** `hdiff_step` matches the Fortran driver golden, conserves mass,
preserves a uniform mixing ratio, and jits and differentiates.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **B2.1** | `hdiff.py`: `hdiff_step` — halo seed, sub-step loop, update (`hdiff.F:455-530`) | Matches the driver golden; halo frozen across sub-steps, as the Fortran does | `pytest tests/regression -k hdiff_driver` |
| **B2.2** | Property tests: mass conservation, uniformity, positivity, symmetry | Uniform field unchanged; total mass exactly conserved; no new extrema | `pytest tests/properties -k hdiff` |
| **B2.3** | `api.py`: `transport_step` = HADV → ZADV → HDIFF | Whole step under one `jit` | `pytest tests/unit/test_api.py` |
| **B2.4** | `tests/differentiability/`: `jax.grad` through `hdiff_step` | Matches central differences to 1e-6 relative | `pytest tests/differentiability -k hdiff` |
| **B2.5** | Figures `docs/figures/b2/` + `scripts/make_b2_figures.py` | Diffusion of a spike vs the analytic Gaussian; deformation field on a shear flow | `python scripts/make_b2_figures.py` |

## Notes

**Mass conservation is exact here, unlike advection.** The update is a
difference of face fluxes, so interior fluxes cancel telescopically and only the
boundary terms survive. With the seeded halo those are zero on the first
sub-step, so a single-sub-step run conserves mass to rounding — a sharp test.
Over several sub-steps the frozen halo lets a little through, and the test must
say which regime it is asserting rather than picking a tolerance that hides the
difference.

**Uniformity is the analogue of advection's constancy preservation**: a uniform
mixing ratio has zero gradient everywhere, so every flux vanishes and the field
must come back bit-identical. It catches a `rho*J` coupling error immediately —
and note the coupling differs from advection's, so intuition from `A2` is
actively misleading here.

**`max_substeps` again.** `NSTEPS` is data-dependent (`int(DTSEC/DT) + 1`), so
as in `A2` it becomes a fixed-count `lax.fori_loop` with a cap, and the cap is
a cost lever worth measuring.
