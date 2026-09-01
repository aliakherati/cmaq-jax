# California NARR smoke test

The original visual demonstration: two inert Central Valley tracers pushed
through a month of NARR winds by `cmaq_jax.hadv.hadv_step`, CMAQ's PPM
horizontal advection.

This is a **smoke test, not physical validation**.  It runs on five pressure
levels, omits vertical advection, and resets atmospheric mass every step.  The
physical validation lives in [`examples/conus404/`](../conus404/README.md) —
native 4 km, 50 layers, real fire emissions, HADV→ZADV together, mass never
reset — and in [`examples/epa_2023/`](../epa_2023/README.md) at full CONUS
scale.  What this example is good for is watching the scheme move a plume
across a real domain under real winds.

## What it does and does not run

The grid **is** the NARR grid.  NARR is Lambert conformal, which is what CMAQ
runs on, so the reanalysis cells are used directly as model cells — nothing
regrids the meteorology, and the winds are exactly what the reanalysis says
they were.  At 32 km it carries the synoptic flow and the regional sea breeze;
it does not resolve the San Joaquin Valley itself, which is about two and a
half cells wide.

Vertical advection is deliberately left out.  `zadv`'s flux diagnosis assumes a
column closed at both ends — sigma thicknesses summing to one — and this domain
is a five-layer slab through the lower troposphere, not a whole atmosphere.
NARR's horizontal winds are divergent, though, and with no vertical operator to
absorb that divergence the advected ρ·J wanders (measured: 1.0 → 0.20–2.81 over
two days).  So `restore_density()` in `run_advection.py` puts ρ·J back each step
and rescales the coupled concentrations with it, leaving the mixing ratio — the
quantity plotted — untouched.  What that does *not* reproduce is the vertical
redistribution `zadv` would also do: there is no vertical exchange between the
five layers at all.

The two tracers are tracers.  Inert species with plausible source *patterns* —
urban centres for traffic, a band along the valley floor for agriculture — at
arbitrary units.  Nothing here is an emissions inventory, and with no chemistry,
deposition or vertical mixing the concentrations are transport only.

## Reproduce

Install the I/O and figure dependencies:

```bash
uv pip install -e ".[dev,io]"
```

Fetch a month of NARR winds on five pressure levels (~40 MB, into the
git-ignored `data/`), and optionally the coastline used to draw the maps:

```bash
.venv/bin/python examples/california/download_met.py --year 2018 --month 7
.venv/bin/python examples/california/fetch_coastline.py
```

Advect, then render:

```bash
.venv/bin/python examples/california/run_advection.py   # --days shortens it
.venv/bin/python examples/california/make_animation.py
```

## Output

| File | What it shows |
|---|---|
| [`california_july2018.gif`](figures/california_july2018.gif) | The month, three-hourly, both tracers with the 1000 mb wind field over them. |
| [`california_summary_201807.png`](figures/california_summary_201807.png) | Where the tracers enter, the July mean of each, and three snapshots of the agricultural tracer. |
