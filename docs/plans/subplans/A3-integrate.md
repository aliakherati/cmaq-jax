# A3 — Integration

Parent: [`../PLAN-advection.md`](../PLAN-advection.md) · Depends on A1, A2

Assemble the full operator, prove it differentiates, and measure it.

**Gate:** `advect_step` jits end-to-end, `jax.grad` through it matches finite
differences, and CPU/GPU timings are recorded.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **A3.1** | `advstep.py`: CFL analysis → sync step + per-layer `ASTEP` (`advstep.F`) | Reproduces Fortran `ASTEP` for a given wind field, incl. the `HDIV_LIM = 0.9` divergence limit | `pytest tests/unit/test_advstep.py` |
| **A3.2** | `api.py`: `advect_step(state, met, cfg)` = HADV → ZADV | Whole step under one `jit`; no host sync in the hot path | `pytest tests/unit/test_api.py` |
| **A3.3** | End-to-end property tests on synthetic 3-D fields | Global mass conserved; constancy preserved through both operators | `pytest tests/properties -k end_to_end` |
| **A3.4** | `tests/differentiability/`: `jax.grad` w.r.t. wind and initial field | Gradients match central finite differences to 1e-6 relative | `pytest tests/differentiability` |
| **A3.5** | `io_mcip.py`: read MCIP/IOAPI meteorology via `xarray` | Loads `UHAT_JD`, `VHAT_JD`, `DENSA_J`, `JACOBM`, `ZF`, `MSFX2`; transposes to `(col,row,lay,spc)` | *blocked on real met data* |
| **A3.6** | `cmaq_jax.bench` — CPU/GPU timing at benchmark resolution | Timings recorded for 100×105×35, ~80 species | `python -m cmaq_jax.bench` |
| **A3.7** | Figures `docs/figures/a3/` + README status table final | Scaling plot; gradient-check plot | `python scripts/make_a3_figures.py` |

## Notes

**A3.5 is the one chunk gated on something outside the repo.** No benchmark
meteorology is available locally; CMAQ ships run scripts but not data, and
`$CMAQ_DATA` must be downloaded separately (`DOCS/CMAQ_Data.md`). The reader is
written against variable names already confirmed in `hcontvel.F` and `x_ppm.F`,
but stays untested until data exists. Everything else in A3 runs on synthetic
fields.

**Differentiability is the payoff**, mirroring `som-jax`'s S1.16. If `jax.grad`
doesn't work through the operator, the port has not delivered its main
advantage over a CUDA-Fortran rewrite. Note the `stop_gradient` switch on the
A2.2 velocity adjustment — document which mode the gradient tests use and why.

**Multi-device is deliberately *not* here.** The halo abstraction from A1.2 is
what makes it a later, additive change. Do not attempt `shard_map` before the
single-device version is green.
