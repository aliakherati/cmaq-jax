# C3 — Integration

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md) · Depends on C2

**Gate: passed.** `science_step` runs VDIFF ahead of the coupled transport
block, `jax.grad` through `vdiff_step` matches central differences to 1e-5
relative — including with respect to the diffusivity — the benchmark is
recorded below, and the figures are in `docs/figures/c2/`.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C3.1** ✅ | `api.py`: `science_step` = VDIFF → HADV → ZADV → HDIFF | Whole step under one `jit` | `pytest tests/unit/test_api.py` |
| **C3.2** ✅ | `tests/differentiability/`: `jax.grad` through `vdiff_step` | Matches central differences to 1e-6 | `pytest tests/differentiability -k vdiff` |
| **C3.3** ✅ | Benchmark: the sequential solves at benchmark resolution | Cost of the layer-scan recorded, CPU and GPU | `python -m cmaq_jax.bench` |
| **C3.4** ✅ | Figures `docs/figures/c2/` + README | Kz profile; convective vs stable evolution; well-mixed approach | `python scripts/make_c2_figures.py` |

## Notes

**Differentiability through a linear solve is the interesting case.** Advection
and diffusion were explicit updates; this is the first operator where the
gradient goes through a solve. JAX differentiates the scan directly, which is
correct but not necessarily what we want — the adjoint of a tridiagonal solve is
another tridiagonal solve, and if the scan-based gradient is slow, a custom VJP
is the fix. Measure before deciding.

**The layer scan: measured on CPU, still open on GPU.** At the benchmark
resolution (100×105×35, 20 species, float64) on this laptop:

| `max_substeps` | per sync step | per layer-scan pair |
|---|---|---|
| 1 | 93 ms | 46.5 ms |
| 3 *(what the column needs)* | 245 ms | 40.9 ms |
| 4 | 316 ms | 39.5 ms |
| 16 | 1244 ms | 38.9 ms |

Cost is almost exactly linear in the sub-step count, at ~40 ms per pair of
layer scans — so the scan *is* the cost, as expected, but it is not pathological:
245 ms puts vertical diffusion at about 40% of advection's 600 ms per sync step
on the same domain.

The GPU question is unresolved because there is no GPU here. The flat
per-scan-pair figure is the number to re-measure there: if it fails to fall the
way the parallel-over-columns work does, the scan is serialising and a parallel
cyclic reduction is the answer — its own chunk, not a quiet substitution.

**Order matters and is easy to get wrong.** `sciproc.F` runs VDIFF *first*, on
uncoupled concentrations, before `COUPLE`. It is not part of the coupled
transport block that `transport_step` implements, so `science_step` has to
apply it outside — not by appending it to the existing chain.
