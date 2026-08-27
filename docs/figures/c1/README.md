# C1 figures — the ACM2 solvers and eddy diffusivity

Regenerate with `python scripts/make_c1_figures.py`.

## `matrix_structure.png`

Why ACM2 needs two solvers, which is the least obvious thing about it.

The **local stage** is ordinary vertical diffusion: each layer exchanges with
its neighbours, so the matrix is tridiagonal and the Thomas algorithm applies.

The **convective stage** is not. The non-local plume carries mass from the
surface layer directly into every layer of the convective boundary layer, so
every row couples to column 1 — the matrix is tridiagonal *plus a full first
column*, and a Thomas sweep would be solving a different system. `matrix1.F`
exists for exactly this shape.

The dashed line is the CBL top, `KL`. Rows above it take no part. `KL` varies
per column in a real run, which is what makes the convective stage awkward to
vectorise: the mask has to exclude those rows without making the matrix
singular.

## `eddy_diffusivity.png`

**Left — three stability regimes.** The same column at `1/L` negative, zero and
positive. Unstable mixes roughly an order of magnitude more than stable at the
same height. Above the PBL the surface term switches off and only the small
Richardson-number term survives, which is the flat tail all three share.

**Middle — the neutral case against its closed form.** With `1/L = 0` the
stability function is exactly 1 and the surface-layer diffusivity reduces to

```
Kz = κ · u* · z · (1 − z/h)²
```

with no free parameters. The port sits on that curve. This is the check that
pins the implementation to the *scheme* rather than merely to one Fortran run —
a golden says "the same as CMAQ", this says "the same as Monin–Obukhov
similarity".

**Right — the stability function.** `Kz ∝ u*/φ_h`, so `φ_h` is what the three
regimes actually differ by. Note the kink at `z/L = 1`: the stable branch
changes formula there, and a mild stable case never reaches it. That is why
there is a `very_stable` golden as well as a `stable` one.
