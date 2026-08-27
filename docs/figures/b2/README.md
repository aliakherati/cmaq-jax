# B2 figures — horizontal diffusion

Regenerate with `python scripts/make_b2_figures.py`.

## `diffusivity_field.png`

Where the scheme actually diffuses, on a 1 km grid with a shear band and a
swirl.

The point the third panel makes is the one most easily got wrong: **the
diffusivity is never zero, even where the deformation is.** `KHD = max(KHMIN,
ACOEF·deformation)` puts a floor under it, so a cell in perfectly uniform flow
still diffuses at `KHA·KHMIN/(KHA+KHMIN)` — 199 m²/s here. Reading the
deformation map as the diffusivity map looks right and is wrong.

The blend `KHA·KHD/(KHA+KHD)` also saturates: however sheared a cell is, its
diffusivity cannot exceed `KHA`. The middle panel is on a log scale because the
field spans two orders of magnitude between those two limits, and a linear scale
collapses everything outside the shear band into one colour.

## `spike_spreading.png`

A single cell at 100, one hour later. The cross-section is bracketed by two
constant-`K` Gaussians: one built from `K` near the source, one from the domain
mean. The numerical solution tracks the *local* one closely, which is both a
check against theory and a demonstration that the spread follows `K` where the
plume actually is rather than any domain average.

Run at `r = 0.20`, which is **not** CMAQ's own sub-step count — see below.

## `substep_stability.png`

CMAQ's diffusion sub-step sits just past the stability limit of the scheme it
is stepping.

`hcdiff3d.F:253` sets `DT = CFC·dx1·dx2/max(K)` with `CFC = 0.300`
(`hcdiff3d.F:115`). For an explicit five-point Laplacian, von Neumann analysis
of the grid-scale mode requires

```
r = K·dt/dx² ≤ 0.25
```

and `NSTEPS = int(DTSEC/DT) + 1` means that once sub-stepping engages at all,
`dt` converges to `DT` and `r` converges to `CFC = 0.300`. The grid-scale mode
then grows by `|1 − 8r| = 1.4` per sub-step. The left panel shows `r` crossing
the limit as the grid refines; the right shows what it does to a point source on
a 1 km grid — ±19627 from an initial 100.

Three things worth stating precisely, because the finding is easy to overstate:

- **It is not a port artefact.** `hdiff.F` compiled unmodified produces the same
  blow-up on the same inputs, to the same values.
- **It does not affect CMAQ's benchmark configurations.** At 12 km the stable
  step is ~2×10⁵ s, `NSTEPS` is 1, and `dt` is the sync step — three orders of
  magnitude below the limit. `r` only approaches `CFC` once sub-stepping
  engages, which needs a fine grid.
- **It also needs an extended region of near-maximal `K`.** The golden cases at
  1 km and 500 m are formally past the limit (`r = 0.299`) yet stay bounded,
  because a 7×6 domain with one hot spot gives the unstable mode nowhere to
  develop.

## `halo_mass_leak.png`

The consequence of the frozen halo, measured.

`hdiff.F` seeds the halo once before the `DO 344` sub-step loop and reloads only
the interior on each pass. On the first sub-step the halo equals its neighbour,
the boundary gradient is zero and no flux crosses. Afterwards it is a fixed value
the interior has moved away from — a **Dirichlet condition pinned at t = 0**,
not the no-flux condition `hdiff.F:25` describes — and mass flows toward it.

Direction depends on the field, which is why the figure plots two rather than
quoting one number: a smooth interior field loses 0.12%, while tracer banked
against a wall *gains* 8.9%, because the halo holds the high initial edge value
while the interior drains.

That the drift is the halo and not something else was checked directly: running
the same golden cases with the halo refreshed each sub-step gives a mass drift of
exactly zero, against −1.85% and −2.23% frozen.
