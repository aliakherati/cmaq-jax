# B1 — Kernels

Parent: [`../PLAN-hdiff.md`](../PLAN-hdiff.md) · Depends on B0

The three quantities the update needs, each matched to its golden.

**Gate:** deformation, eddy diffusivity and the face-averaged coefficients all
match the Fortran in both float32 and float64.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **B1.1** | `hdiff.py`: `deformation` (`deform.F:352-432`) | Matches goldens; zero for solid-body translation; invariant under frame rotation | `pytest tests/regression -k deform` |
| **B1.2** | `hdiff.py`: `eddy_diffusivity` (`hcdiff3d.F:180-200`) | Matches goldens; `>= KHMIN`-derived floor; saturates at `KHA` as deformation grows | `pytest tests/unit -k diffusivity` |
| **B1.3** | `hdiff.py`: `face_coefficients` → `K11BAR`/`K22BAR`, and `stable_timestep` | Matches goldens incl. the zeroed last row/column | `pytest tests/regression -k coefficients` |

## Notes

**Deformation is a rotational invariant** — `sqrt(DF1^2 + DF2^2)` is the second
invariant of the strain-rate tensor, so rotating the wind field must leave it
unchanged. That is a stronger test than the golden (which only pins one field)
and catches a swapped `DF1`/`DF2` or a sign error that a single case can miss.

Confirmed against the harness, and it holds **exactly** — but only on the strict
interior. `deform.F` zeroes `DUDY` at rows 1 and `NROWS` and `DVDX` at columns 1
and `NCOLS` (`deform.F:420-421`), and those are *different* edges, so a slice
that keeps one of them breaks the symmetry and the test fails against correct
code. Measured: over `COL 2..NCOLS-1, ROW 2..NROWS-1` the rotated and unrotated
fields agree bit-for-bit and match the analytic value; extend the slice by one
column and the discrepancy is 40% of the signal. Scope the test to the strict
interior and say why.

**Solid-body translation has zero deformation** but non-zero wind, which
separates "reads the wind" from "computes the gradient correctly".

**The blend saturates.** `KHA*KHD/(KHA+KHD) -> KHA` as `KHD -> inf`. Worth a
test at large deformation: a cell in strong shear must not get an unbounded
diffusivity, and getting the blend upside-down would still look plausible on a
mild field.
