# C3 — Integration

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md) · Depends on C2

**Gate: C3.1/C3.2/C3.4 passed.** `science_step` runs VDIFF ahead of the coupled
transport block, `jax.grad` through `vdiff_step` matches central differences to
1e-5 relative — including with respect to the diffusivity — and the figures are
in `docs/figures/c2/`. C3.3 (benchmark) remains.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C3.1** ✅ | `api.py`: `science_step` = VDIFF → HADV → ZADV → HDIFF | Whole step under one `jit` | `pytest tests/unit/test_api.py` |
| **C3.2** ✅ | `tests/differentiability/`: `jax.grad` through `vdiff_step` | Matches central differences to 1e-6 | `pytest tests/differentiability -k vdiff` |
| **C3.3** | Benchmark: the sequential solves at benchmark resolution | Cost of the layer-scan recorded, CPU and GPU | `python -m cmaq_jax.bench` |
| **C3.4** ✅ | Figures `docs/figures/c2/` + README | Kz profile; convective vs stable evolution; well-mixed approach | `python scripts/make_c2_figures.py` |

## Notes

**Differentiability through a linear solve is the interesting case.** Advection
and diffusion were explicit updates; this is the first operator where the
gradient goes through a solve. JAX differentiates the scan directly, which is
correct but not necessarily what we want — the adjoint of a tridiagonal solve is
another tridiagonal solve, and if the scan-based gradient is slow, a custom VJP
is the fix. Measure before deciding.

**The layer scan is the GPU question.** `NLAYS ≈ 35` sequential steps per solve,
two solves per sub-step, several sub-steps per sync step. On CPU that is
nothing; on GPU a short sequential scan can dominate. If it does, the answer is
a parallel cyclic reduction, which is a real change and belongs in its own
chunk rather than being smuggled into C1.

**Order matters and is easy to get wrong.** `sciproc.F` runs VDIFF *first*, on
uncoupled concentrations, before `COUPLE`. It is not part of the coupled
transport block that `transport_step` implements, so `science_step` has to
apply it outside — not by appending it to the existing chain.
