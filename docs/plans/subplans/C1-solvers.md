# C1 — Solvers and diffusivity

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md) · Depends on C0

**Gate: passed.** Both solvers match in both precisions (worst 0.98 float32
ULPs across 9 cases) and satisfy `‖Ax−b‖` at float64 machine precision.
`eddy_diffusivity` matches across 10 cases, worst 3.0 ULPs, and reproduces the
neutral surface-layer relation `κ·u*·z·(1−z/h)²` in closed form.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C1.1** ✅ | `vdiff.py`: `solve_tridiagonal` (`tri.F`) | Matches goldens; residual `‖Ax−b‖` at machine precision | `pytest tests/regression -k tri` |
| **C1.2** ✅ | `vdiff.py`: `solve_acm1` (`matrix1.F`) | Matches goldens; residual small on the first-column matrix | `pytest tests/regression -k matrix1` |
| **C1.3** ✅ | `vdiff.py`: `eddy_diffusivity` (`eddyx.F`) | Reproduces neutral/stable/unstable limits; `≥ KZMIN`; zero above the PBL | `pytest tests/unit -k eddy` |

## Notes

**Test the solvers by residual, not only by golden.** `A·x = b` is checkable
directly: assemble the matrix, multiply back, compare. That is a stronger
statement than agreeing with one Fortran run, and it is the test that would
catch a transposed sub/super-diagonal — which a symmetric test matrix would
hide. Use a deliberately asymmetric matrix.

**One matrix, many species.** Both routines take `B(:,:)` — species by layer —
and solve them all against the same `L/D/U`. In JAX that is a single scan whose
carry is species-shaped, not a `vmap` over species; the factorisation is shared
and doing it per species would be both slower and a different computation to
read.

**`eddyx` limits are the real test.** Its analytic behaviour is known:
`PHIH → 1` as `z/L → 0` (neutral), `EDYZ → 0` at the PBL top by the
`(1 − z/h)²` factor, and `EDYZ ≥ KZMIN` everywhere below it. Those pin the
parameterization far better than one field of numbers.
