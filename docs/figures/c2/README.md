# C2 figures — the ACM2 operator

Regenerate with `python scripts/make_c2_figures.py`.

## `acm2_vs_local.png`

A surface release over one hour, in a convective column and a stable one. The
only difference between the two runs is the `CONVCT` flag.

The stable column (left) shows a diffusion front creeping upward layer by layer
— the classic square-root-of-time spread. The convective column (middle) fills
the whole boundary layer almost immediately, because the non-local plume carries
mass from the surface layer *directly* to every layer of the CBL rather than
passing it along the chain. That is what ACM2 is for, and it is the behaviour a
purely local scheme cannot reproduce at any diffusivity: local diffusion cannot
put mass at 1200 m before it has passed through 600 m.

The right panel is the same thing as profiles. Note the convective case is far
closer to uniform across the CBL at both times.

## `model_top_leak.png`

The local stage's top boundary is closed only because the diffusivity happens to
be zero there.

`vdiffacmx.F:675` sets `BB2(L) = 1 - CC(L) - EE2(L)` for every layer including
the last, so the top row's diagonal carries an upward-flux term
`EE2(NLAYS) = -DFSP(NLAYS)·EDDY(NLAYS)`. The right-hand side at that layer has
no matching `LFAC3` term (`vdiffacmx.F:1007-1008`). The flux is therefore
one-sided: mass leaves through the model top into nothing.

The left panel measures it. A perfectly uniform column — which should not change
at all — loses mass in exact proportion to the top-layer `Kz`, and exactly zero
when that is zero. The right panel shows the loss is confined to the top few
layers.

**It never bites in CMAQ**, because `eddyx.F` returns `EDDYV = 0` for the top
layer: `Kz` lives on layer interfaces and the top layer has none above it. The
two facts are load-bearing together — the scheme is conservative because the
diffusivity vanishes exactly where the matrix would otherwise leak, not because
the boundary is closed.

Verified against unmodified Fortran, which leaks identically: a uniform column
of 7.0 with `Kz(top) = 20` loses 0.48% of its mass in one 300 s step. A test in
`tests/properties/test_vdiff.py` guards the assumption by asserting that
`eddy_diffusivity` really does return zero there — if that ever changed, the
operator would start leaking and nothing else would notice.
