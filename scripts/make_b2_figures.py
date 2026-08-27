#!/usr/bin/env python3
"""B2 figures: horizontal diffusion.

    python scripts/make_b2_figures.py

Three figures, each answering a question the numbers alone do not:

* ``diffusivity_field`` -- where the scheme actually diffuses. Deformation, the
  diffusivity it produces, and the saturating blend that connects them, so the
  KHMIN floor and the KHA ceiling are both visible.
* ``spike_spreading`` -- the operator doing its job, against the analytic
  Gaussian a constant-coefficient Laplacian would give. Run with enough
  sub-steps to be *stable*, which is more than CMAQ asks for; see below.
* ``substep_stability`` -- why that caveat is needed. CMAQ picks its diffusion
  sub-step as ``CFC*dx^2/Kmax`` with ``CFC = 0.300``, but explicit 2-D diffusion
  needs ``dt <= 0.25*dx^2/Kmax``. Whenever the sub-stepping actually engages the
  scheme sits just past its own stability limit.
* ``halo_mass_leak`` -- the consequence of the frozen halo, measured. The halo
  is seeded from the initial edge value and never refreshed, so after the first
  sub-step it acts as a *Dirichlet* condition pinned at t=0 rather than the
  no-flux condition ``hdiff.F:25`` describes. Mass therefore flows toward that
  pinned value -- outward where the interior has risen above it, inward where it
  has fallen below. The figure plots both, because a single number would suggest
  a bias that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from cmaq_jax.config import DEFAULT_HDIFF, GridConfig, sigma_layer_thickness
from cmaq_jax.hdiff import (
    contravariant_winds,
    deformation,
    eddy_diffusivity,
    face_coefficients,
    halo_density,
    hdiff_step,
    stable_timestep,
    substep_count,
)

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures" / "b2"
NCOLS, NROWS, NLAYS = 48, 40, 1


def problem(dx: float, *, spike: str = "", seed: int = 20260827):
    """A domain with a sheared, swirling wind and either noise or a spike."""
    rng = np.random.default_rng(seed)
    rows = np.arange(NROWS + 1, dtype=np.float64)[None, :, None]
    cols = np.arange(NCOLS + 1, dtype=np.float64)[:, None, None]

    rho = np.full((NCOLS, NROWS, NLAYS), 2.0)
    ring = np.full((2 * (NCOLS + NROWS + 2), NLAYS), 2.0)

    # A shear band across the middle plus a swirl, so the deformation field has
    # structure rather than being flat.
    centre_r, centre_c = NROWS / 2.0, NCOLS / 2.0
    band = np.tanh((rows - centre_r) / 3.0)
    swirl = np.exp(-(((cols - centre_c) / 8.0) ** 2 + ((rows - centre_r) / 8.0) ** 2))
    u = (60.0 * band + 40.0 * swirl * (rows - centre_r)) * 2.0
    v = (30.0 * swirl * (cols - centre_c)) * 2.0

    if spike == "centre":
        q = np.zeros((NCOLS, NROWS, NLAYS, 1))
        q[NCOLS // 2, NROWS // 2, 0, 0] = 100.0
    elif spike == "edge":
        # Material banked against the west wall, so the stale halo sits next to
        # a real gradient rather than a flat field.
        q = np.zeros((NCOLS, NROWS, NLAYS, 1))
        q[:3, :, 0, 0] = 100.0
    else:
        q = 1.0 + rng.random((NCOLS, NROWS, NLAYS, 1))
    state = np.concatenate([q * rho[..., None], rho[..., None]], axis=-1)

    densj = halo_density(rho, ring)
    uu, vv = contravariant_winds(u, v, densj)
    deform = deformation(uu, vv, dx1=dx, dx2=dx)
    eddyh = eddy_diffusivity(deform, np.ones((NCOLS + 1, NROWS + 1)), dx1=dx, dx2=dx)
    k11, k22 = face_coefficients(eddyh)
    cfg = GridConfig(
        ncols=NCOLS,
        nrows=NROWS,
        ds=sigma_layer_thickness(np.linspace(1.0, 0.0, NLAYS + 1)),
        dx1=dx,
        dx2=dx,
        nspc_adv=2,
    )
    return cfg, state, densj, np.asarray(deform), np.asarray(eddyh), k11, k22


def diffusivity_field() -> None:
    dx = 1000.0
    _, _, _, deform, eddyh, _, _ = problem(dx)
    kha = DEFAULT_HDIFF.base_diffusivity(dx, dx)
    acoef = DEFAULT_HDIFF.deformation_coefficient(dx, dx)
    floor = kha * DEFAULT_HDIFF.khmin / (kha + DEFAULT_HDIFF.khmin)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    interior = (slice(0, NCOLS), slice(0, NROWS), 0)
    im = axes[0].imshow(deform[interior].T, origin="lower", cmap="magma")
    axes[0].set_title("wind deformation  $\\sqrt{DF_1^2 + DF_2^2}$  (s$^{-1}$)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # Log scale: the field runs from the KHMIN floor to near the KHA ceiling,
    # two orders of magnitude, and on a linear scale everything outside the
    # shear band collapses to a single colour -- hiding the point being made.
    im = axes[1].imshow(
        eddyh[interior].T,
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=floor, vmax=kha),
    )
    axes[1].set_title("eddy diffusivity $K_H$ (m$^2$ s$^{-1}$, log scale)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes[:2]:
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    # The blend, with the two limits the field above sits between.
    d = np.logspace(-8, 2, 400)
    khd = np.maximum(DEFAULT_HDIFF.khmin, acoef * d)
    axes[2].loglog(d, kha * khd / (kha + khd), lw=2, color="C0", label="$K_H$")
    axes[2].axhline(kha, ls="--", color="C3", label=f"ceiling $K_{{HA}}$ = {kha:.0f}")
    axes[2].axhline(floor, ls=":", color="C2", label=f"floor (from $K_{{HMIN}}$) = {floor:.0f}")
    observed = deform[interior]
    axes[2].axvspan(
        max(observed.min(), 1e-8),
        observed.max(),
        color="0.85",
        zorder=0,
        label="range in this field",
    )
    axes[2].set_xlabel("deformation (s$^{-1}$)")
    axes[2].set_ylabel("$K_H$ (m$^2$ s$^{-1}$)")
    axes[2].set_title("the saturating blend $K_{HA}K_{HD}/(K_{HA}+K_{HD})$")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Deformation-dependent horizontal diffusivity, {dx:.0f} m grid — "
        "note the diffusivity is never zero, even where deformation is",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "diffusivity_field.png", dpi=140)
    plt.close(fig)


def spike_spreading() -> None:
    dx = 1000.0
    cfg, state, densj, _, eddyh, k11, k22 = problem(dx, spike="centre")
    sync = 3600.0
    # Deliberately *not* CMAQ's substep_count here: at this spacing its own
    # criterion lands at r = 0.30, past the 0.25 stability limit, and the field
    # fills with a growing checkerboard instead of a plume. See
    # substep_stability() for that. Here we want the physics, so take enough
    # sub-steps to sit at r = 0.20.
    kmax = float(max(np.asarray(k11).max(), np.asarray(k22).max()))
    nsteps = int(np.ceil(sync * kmax / (0.20 * dx * dx)))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    initial = np.asarray(state)[..., 0, 0] / 2.0
    axes[0].imshow(initial.T, origin="lower", cmap="inferno")
    axes[0].set_title("initial: a single cell at 100")

    out = np.asarray(hdiff_step(state, densj, k11, k22, cfg=cfg, sync_seconds=sync, nsteps=nsteps))
    final = out[..., 0, 0] / 2.0
    im = axes[1].imshow(final.T, origin="lower", cmap="inferno")
    axes[1].set_title(f"after 1 h ({nsteps} sub-steps, $r$ = 0.20)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes[:2]:
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    # Against the analytic answer for a *constant* coefficient, which is what
    # the real thing is not -- the gap is the deformation dependence.
    row = NROWS // 2
    x = (np.arange(NCOLS) - NCOLS // 2) * dx
    axes[2].plot(x / 1000.0, final[:, row], "o-", ms=3, label="cmaq-jax")
    # Compare against constant-K solutions built from K *where the plume is* and
    # from the domain mean: the plume sits in the swirl where K is highest, so a
    # mean alone reads as the scheme being wrong rather than as K varying.
    half = 6
    local = np.asarray(eddyh)[
        NCOLS // 2 - half : NCOLS // 2 + half, NROWS // 2 - half : NROWS // 2 + half, 0
    ]
    for k_eff, style, label in (
        (float(local.mean()), "--", "near the source"),
        (float(np.asarray(eddyh)[:NCOLS, :NROWS, 0].mean()), ":", "domain mean"),
    ):
        var = 2.0 * k_eff * sync
        analytic = 100.0 * dx**2 / (4.0 * np.pi * k_eff * sync) * np.exp(-(x**2) / (2 * var))
        axes[2].plot(
            x / 1000.0,
            analytic,
            style,
            color="C3",
            label=f"Gaussian, $K$={k_eff:.0f} ({label})",
        )
    axes[2].set_xlabel("distance from source (km)")
    axes[2].set_ylabel("mixing ratio")
    axes[2].set_title("cross-section through the source")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "A point release spreading, bracketed by constant-$K$ Gaussians built from the "
        "local and domain-mean diffusivity —\nthe spread follows $K$ where the plume "
        "actually is. Run at $r$ = 0.20; CMAQ's own sub-step count is unstable here.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "spike_spreading.png", dpi=140)
    plt.close(fig)


def halo_mass_leak() -> None:
    """The frozen halo, measured.

    ``hdiff.F`` seeds the halo once before the sub-step loop and reloads only
    the interior each pass. On the first sub-step the halo equals its neighbour,
    so the gradient is zero and no flux crosses. Afterwards it is a fixed value
    the interior has moved away from -- a Dirichlet condition pinned at t=0 --
    and mass flows toward it. Direction depends on the field: a smooth interior
    field pushes a little out, while tracer banked against a wall pulls a lot
    in, because the halo holds the high initial edge value while the interior
    drains.

    None of it is visible on the benchmark grid, where the stable step is ~2e5 s
    and NSTEPS is always 1.
    """
    sync = 3600.0
    spacings = [12000.0, 8000.0, 4000.0, 2000.0, 1000.0, 700.0, 500.0]
    fields = {
        "smooth field, nothing at the wall": "",
        "tracer banked against the west wall": "edge",
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for (label, kind), colour in zip(fields.items(), ("C0", "C3"), strict=True):
        counts, drifts = [], []
        for dx in spacings:
            cfg, state, densj, _, _, k11, k22 = problem(dx, spike=kind)
            nsteps = substep_count(sync, float(stable_timestep(k11, k22, dx1=dx, dx2=dx)))
            out = np.asarray(
                hdiff_step(state, densj, k11, k22, cfg=cfg, sync_seconds=sync, nsteps=nsteps)
            )
            before = np.asarray(state)[..., 0].sum()
            counts.append(nsteps)
            drifts.append(100.0 * (out[..., 0].sum() - before) / before)
        ax.semilogx(counts, drifts, "o-", color=colour, label=label)
        ax.annotate(
            f"{drifts[-1]:.2f}%",
            (counts[-1], drifts[-1]),
            textcoords="offset points",
            xytext=(-38, -4),
            fontsize=9,
            color=colour,
        )

    ax.axhline(0.0, color="0.4", lw=0.8)
    top = ax.get_ylim()[1]
    for dx, n in zip(spacings, counts, strict=True):
        ax.annotate(
            f"{dx / 1000:g} km",
            (n, top),
            textcoords="offset points",
            xytext=(0, -34),
            rotation=90,
            fontsize=7,
            color="0.45",
            ha="center",
            va="top",
        )
    ax.set_xlabel("diffusion sub-steps in one hour  (labelled by grid spacing)")
    ax.set_ylabel("tracer mass change (%)")
    ax.set_title(
        "The frozen halo is a Dirichlet condition, not a no-flux one\n"
        "mass is exact at one sub-step, then flows toward the pinned edge value"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "halo_mass_leak.png", dpi=140)
    plt.close(fig)


def substep_stability() -> None:
    """CMAQ's diffusion sub-step sits just past the 2-D stability limit.

    ``hcdiff3d.F:253`` sets ``DT = CFC * dx1 * dx2 / max(K)`` with ``CFC = 0.300``
    (``hcdiff3d.F:115``). For an explicit five-point Laplacian the von Neumann
    condition on the grid-scale mode is ``r = K*dt/dx^2 <= 0.25``, so the choice
    is stable only while ``dt`` is set by something *other* than this criterion.

    ``NSTEPS = int(DTSEC/DT) + 1`` means that once sub-stepping engages at all,
    ``dt`` converges to ``DT`` and ``r`` converges to ``CFC`` -- and the
    grid-scale mode grows by ``|1 - 8r| = 1.4`` per sub-step.

    It does not bite on CMAQ's benchmark grids: at 12 km the stable step is
    ~2e5 s, ``NSTEPS`` is 1, and ``dt`` is the sync step, far below the limit.
    Reaching it needs a fine grid *and* an extended region of near-maximal K --
    a small domain with one hot cell stays bounded because the unstable mode has
    nowhere to develop.

    Verified against the Fortran: on the unstable configuration below,
    ``hdiff.F`` produces the same blow-up, to the same values.
    """
    sync = 3600.0
    spacings = [12000.0, 8000.0, 4000.0, 2000.0, 1500.0, 1000.0, 700.0, 500.0]
    ratios, counts = [], []
    for dx in spacings:
        _, _, _, _, _, k11, k22 = problem(dx)
        nsteps = substep_count(sync, float(stable_timestep(k11, k22, dx1=dx, dx2=dx)))
        kmax = float(max(np.asarray(k11).max(), np.asarray(k22).max()))
        counts.append(nsteps)
        ratios.append(kmax * (sync / nsteps) / (dx * dx))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    ax.semilogx(spacings, ratios, "o-", color="C0", label="$r = K\\,dt/dx^2$ as CMAQ runs it")
    ax.axhline(0.25, ls="--", color="C3", label="2-D stability limit, $r$ = 0.25")
    ax.axhline(DEFAULT_HDIFF.cfc, ls=":", color="C1", label=f"CFC = {DEFAULT_HDIFF.cfc}")
    ax.fill_between([min(spacings), max(spacings)], 0.25, 0.45, color="C3", alpha=0.10)
    for dx, r, n in zip(spacings, ratios, counts, strict=True):
        ax.annotate(
            f"{n}",
            (dx, r),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            ha="center",
            color="0.35",
        )
    ax.set_xlabel("grid spacing (m)   [labelled with sub-step count]")
    ax.set_ylabel("$r = K_{max}\\,dt/dx^2$")
    ax.set_title(
        "Once sub-stepping engages, $r$ converges to CFC = 0.300\nwhich is past the 0.25 limit"
    )
    ax.set_ylim(0.0, 0.45)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # What that looks like: CMAQ's own sub-step count on a fine grid.
    dx = 1000.0
    cfg, state, densj, _, _, k11, k22 = problem(dx, spike="centre")
    nsteps = substep_count(sync, float(stable_timestep(k11, k22, dx1=dx, dx2=dx)))
    out = np.asarray(hdiff_step(state, densj, k11, k22, cfg=cfg, sync_seconds=sync, nsteps=nsteps))
    field = out[..., 0, 0] / 2.0
    limit = float(np.abs(field).max())
    im = axes[1].imshow(field.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set_title(
        f"1 km grid, CMAQ's own {nsteps} sub-steps\n"
        f"a point source at 100 becomes ±{limit:.0f} — the Fortran agrees"
    )
    axes[1].set_xlabel("column")
    axes[1].set_ylabel("row")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(FIGURES / "substep_stability.png", dpi=140)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    diffusivity_field()
    spike_spreading()
    substep_stability()
    halo_mass_leak()
    print(f"wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
