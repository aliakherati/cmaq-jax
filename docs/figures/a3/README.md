# A3 figures — the assembled operator

Regenerate with:

```bash
python scripts/make_a3_figures.py
```

---

## `adjoint_footprint.png`

**What differentiability is for.** `jax.grad` of a receptor concentration with
respect to the initial field is a source-receptor footprint: *which upwind cells
did the pollution here come from?* One reverse pass, no adjoint written by hand.

CMAQ answers the same question through DDM-3D, a separate and partially
maintained model configuration. Here it falls out of the forward code.

The wind is westerly and veering, so the footprint reaches upwind and tilts
across rows. It concentrates in the surface layer (77% of total sensitivity) but
reaches higher, because the faster flow aloft draws from further back.

### The finite-difference check, and how to get it wrong

The right-hand panel checks the same gradient against central differences. That
check took three corrections, **none of them to the gradient**:

1. **A zero background is invalid.** PPM's limiter and `zfdbc` both clamp at
   zero, so a tracer sitting exactly there is at a corner of the operator — the
   gradient is a one-sided derivative and a central difference straddles it.
2. **Pointwise relative error is the wrong metric.** A cell whose sensitivity is
   a millionth of the peak has a finite difference dominated by cancellation;
   dividing by it manufactured a ratio of 7.4e+05 from a healthy gradient.
3. **The residual was cancellation, not kinks.** I assumed limiter branches. An
   eps sweep says otherwise: on cells above 20% of peak the agreement is
   **1e-12 to 5e-11**, and the error *grows* as eps shrinks — the signature of
   floating-point cancellation in the difference, not truncation.

Worst `|grad − fd|` as a fraction of peak sensitivity, measured:

| background | 1 step | 4 steps | 8 steps |
|---|---|---|---|
| 0.0 | 0.0 | 8.4e-2 | 2.9e-1 |
| 1.0 | 9.2e-12 | 6.0e-11 | 1.1e-10 |

One step from an all-zero field agrees *exactly* — the perturbation has not yet
spread far enough to flip a limiter branch. Both rows are pinned by tests in
`tests/differentiability/`, including one asserting the zero-background
disagreement **persists**: if it ever vanishes, the limiter has stopped
clamping.

---

## `scaling.png`

Cost of one sync step on CPU, against domain size, species count and the
vertical sub-step cap.

- **Domain size** — close to linear in cell-species, which is what a flux-form
  scheme should give.
- **Species count** — throughput *improves* with more species (11.8 → 17.2 M
  cell-species/s), because fixed per-step overhead amortises. That is the right
  direction for a GPU, where the arithmetic-to-dispatch ratio is what matters.
- **`max_substeps`** — CMAQ's error limit of 30 is also this loop's fixed trip
  count, so a column pays for all of it even when it needs two. Lowering it is
  safe: the residual diagnostic reports any column that ran out, and at a cap
  of 1 it correctly reports non-convergence rather than returning wrong numbers.

Benchmark-resolution timings (100×105×35, CPU, float64) are in
`python -m cmaq_jax.bench`.
