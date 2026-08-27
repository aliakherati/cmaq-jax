# B2 — Driver

Parent: [`../PLAN-hdiff.md`](../PLAN-hdiff.md) · Depends on B1

Assemble the operator, prove its properties, and put it where `sciproc.F` does.

**Gate: B2.1/B2.2/B2.4 passed.** `hdiff_step` matches the Fortran driver to a
worst of 1.8 float32 ULPs across four cases forcing 1, 2, 66 and 155 sub-steps,
in both precisions; it jits, and `jax.grad` matches central differences to 1e-6.
`transport_step` composes HADV → ZADV → HDIFF under one `jit`, and the figures
are in `docs/figures/b2/`.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **B2.1** ✅ | `hdiff.py`: `hdiff_step` — halo seed, sub-step loop, update (`hdiff.F:455-530`) | Matches the driver golden; halo frozen across sub-steps, as the Fortran does | `pytest tests/regression -k hdiff_driver` |
| **B2.2** ✅ | Property tests: mass conservation, uniformity, positivity, symmetry | Uniform field unchanged; total mass exactly conserved; no new extrema | `pytest tests/properties -k hdiff` |
| **B2.3** ✅ | `api.py`: `transport_step` = HADV → ZADV → HDIFF | Whole step under one `jit` | `pytest tests/unit/test_api.py` |
| **B2.4** ✅ | `tests/differentiability/`: `jax.grad` through `hdiff_step` | Matches central differences to 1e-6 relative | `pytest tests/differentiability -k hdiff` |
| **B2.5** ✅ | Figures `docs/figures/b2/` + `scripts/make_b2_figures.py` | Diffusion of a spike vs the analytic Gaussian; deformation field on a shear flow | `python scripts/make_b2_figures.py` |

## Notes

**Mass conservation is exact on the first sub-step, and then it is not.** The
update is a difference of face fluxes, so interior terms cancel telescopically
and only the boundary survives; the seeded halo makes that boundary gradient
exactly zero on the first pass. Afterwards the halo is stale and mass crosses
the domain edge. Measured directly from the Fortran goldens:

| sub-steps | mass drift |
|---|---|
| 1 | `+1.9e-10` (exact, to rounding) |
| 2 | `-3.0e-04` |
| 66 | `-1.85e-02` |
| 155 | `-2.23e-02` |

So an hour of deep sub-stepping loses about **2%** of the tracer, consistently
outward: diffusion lifts the edge cells above the frozen halo value, the
gradient there points out of the domain, and mass follows. This is CMAQ's
behaviour, reproduced rather than corrected — the port agrees with the Fortran
to under 2 ULPs on the 155-sub-step case. A single loose tolerance across both
regimes would have hidden it, so the two are asserted separately.

Note it is only reachable on a fine grid. On the 12 km benchmark domain the
stable step is ~2e5 s and `NSTEPS` is always 1, so the leak never appears there
— which is presumably why it has gone unremarked.

**Uniformity is the analogue of advection's constancy preservation**: a uniform
mixing ratio has zero gradient everywhere, so every flux vanishes and the field
must come back bit-identical. It catches a `rho*J` coupling error immediately —
and note the coupling differs from advection's, so intuition from `A2` is
actively misleading here.

**`max_substeps` again.** `NSTEPS` is data-dependent (`int(DTSEC/DT) + 1`), so
as in `A2` it becomes a fixed-count `lax.fori_loop` with a cap, and the cap is
a cost lever worth measuring.

**Uniformity does not survive the full chain exactly, and that is structural.**
`hdiff.F:309` takes its density from `RHO_J`, which reads `DENSA_J` from the
meteorology file — *not* the advected rho*J that advection has just been
transporting in the last `CGRID` slot. So diffusion divides the coupled
concentration by one density while it is coupled to another, and the two agree
only to the extent ZADV has closed the gap.

Measured with the meteorology held fixed across steps so the gap grows: an 18-26%
density mismatch comes with a 1.2-2.2% uniformity error, a ratio near 0.08 that
is stable over ten steps. So the error is *controlled by* the mismatch rather
than accumulating on its own. Feeding diffusion the density the state is actually
coupled to makes the error vanish identically, which identifies the cause rather
than leaving it as a plausible story — both are asserted in
`tests/properties/test_transport_step.py`.
