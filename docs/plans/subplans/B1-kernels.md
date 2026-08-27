# B1 — Kernels

Parent: [`../PLAN-hdiff.md`](../PLAN-hdiff.md) · Depends on B0

The three quantities the update needs, each matched to its golden.

**Gate: passed.** All three match the Fortran in both precisions, worst 5.4
float32 ULPs across 8 cases — and that worst case is `variable_density`'s face
coefficients, where the density division amplifies rounding. Deformation itself
is bit-exact on 7 of 8 cases and within 0.17 ULPs on the eighth, so it is pinned
separately at a 1-ULP budget rather than being allowed to hide inside the
chain's looser tolerance.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **B1.1** ✅ | `hdiff.py`: `deformation` (`deform.F:352-432`) | Matches goldens; zero for solid-body translation; invariant under frame rotation | `pytest tests/regression -k deform` |
| **B1.2** ✅ | `hdiff.py`: `eddy_diffusivity` (`hcdiff3d.F:180-200`) | Matches goldens; `>= KHMIN`-derived floor; saturates at `KHA` as deformation grows | `pytest tests/unit -k diffusivity` |
| **B1.3** ✅ | `hdiff.py`: `face_coefficients` → `K11BAR`/`K22BAR`, and `stable_timestep` | Matches goldens incl. the zeroed last row/column | `pytest tests/regression -k coefficients` |

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

**`halo_density` and `contravariant_winds` were not in the original chunk list.**
`deform.F` recovers the wind by dividing `UHAT_JD` by a density that includes a
one-cell halo ring read separately from the boundary file
(`deform.F:250-332`). That is part of the port, not test plumbing: substituting
a zero-gradient extrapolation for the ring changes the answer wherever the
density varies along the boundary, which is why the `variable_density` and
`smooth_random` golden cases exist.
