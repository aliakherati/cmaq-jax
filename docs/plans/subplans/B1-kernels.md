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

**Solid-body translation has zero deformation** but non-zero wind, which
separates "reads the wind" from "computes the gradient correctly".

**The blend saturates.** `KHA*KHD/(KHA+KHD) -> KHA` as `KHD -> inf`. Worth a
test at large deformation: a cell in strong shear must not get an unbounded
diffusivity, and getting the blend upside-down would still look plausible on a
mild field.
