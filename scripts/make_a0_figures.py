#!/usr/bin/env python3
"""Regenerate the A0 figures into docs/figures/a0/.

    python scripts/make_a0_figures.py

Four figures, each demonstrating a claim made in the README or the subplan:

* ``fortran_agreement`` -- the port matches CMAQ to within float32 rounding.
* ``convergence`` -- it converges at the order PPM should.
* ``limiter_action`` -- the monotonicity limiter fires where it must.
* ``transport`` -- a Gaussian and a square wave after advection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.config import DEFAULT_PPM
from cmaq_jax.ppm import (
    nonuniform_mesh,
    ppm_advect_uniform,
    ppm_parabola_nonuniform,
    ppm_parabola_uniform,
)

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "data" / "goldens"
FIGURES = REPO / "docs" / "figures" / "a0"
SWP = DEFAULT_PPM.halo_width
EPS32 = float(np.finfo(np.float32).eps)

PORT = "#1b6ca8"
REFERENCE = "#d1495b"
MUTED = "#8d99ae"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches=None if fig.get_layout_engine() else "tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


def _periodic(interior: np.ndarray) -> np.ndarray:
    return np.concatenate([interior[-SWP:], interior, interior[:SWP]])


def _advect(interior: np.ndarray, u: float, dt: float, ds: float, nsteps: int) -> np.ndarray:
    field = interior
    vel = np.full((field.shape[0] + 1, 1), u)
    for _ in range(nsteps):
        field = np.asarray(ppm_advect_uniform(_periodic(field), vel, dt, ds))[SWP:-SWP]
    return field


def figure_fortran_agreement() -> None:
    """Per-case disagreement with the Fortran, against the float32 noise floor."""
    names, errors = [], []

    for path in sorted(GOLDENS.glob("hppm_*.npz")):
        with np.load(path) as g:
            con = np.asarray(g["con_in"], dtype=np.float64)
            vel = np.asarray(g["vel_in"], dtype=np.float64)[:, None]
            got = downcast_to_real4(
                np.asarray(ppm_advect_uniform(con, vel, float(g["dt"]), float(g["ds"])))
            )
            expected = np.asarray(g["con_out"], dtype=np.float64)
            scale = max(float(np.abs(expected).max()), 1.0)
            names.append(path.stem.removeprefix("hppm_"))
            errors.append(float(np.abs(got - expected).max()) / scale)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(names))
    # Bit-identical cases have no bar to draw on a log axis; floor them so they
    # are still visible and labelled.
    floor = 1e-9
    plotted = [max(e, floor) for e in errors]
    colours = [MUTED if e == 0.0 else PORT for e in errors]
    ax.bar(positions, plotted, color=colours)

    ax.axhline(EPS32, color=REFERENCE, ls="--", lw=1.2, label=f"float32 eps = {EPS32:.1e}")
    ax.axhline(1e-6, color="k", ls=":", lw=1.2, label="test tolerance = 1e-6")

    for pos, err in zip(positions, errors, strict=True):
        if err == 0.0:
            ax.text(pos, floor * 1.4, "exact", ha="center", fontsize=7, color=MUTED, rotation=90)

    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, 3e-6)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("max relative difference")
    nonzero = [e for e in errors if e > 0.0]
    ax.set_title(
        "JAX port vs CMAQ hppm.F, per golden case\n"
        f"worst case {max(errors):.1e} ({max(errors) / EPS32:.1f} float32 ULPs); "
        f"{len(errors) - len(nonzero)} of {len(errors)} bit-identical",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    _save(fig, "fortran_agreement")


def figure_convergence() -> None:
    """L1 error against grid refinement, with a second-order reference slope."""
    courant, width, distance = 0.5, 0.08, 0.25
    resolutions = [32, 64, 128, 256, 512, 1024]
    errors = []

    for n in resolutions:
        ds = 1.0 / n
        dt = courant * ds
        nsteps = round(distance / dt)
        x = (np.arange(n) + 0.5) / n

        def gaussian(centre: float, grid: np.ndarray = x) -> np.ndarray:
            offset = (grid - centre + 0.5) % 1.0 - 0.5
            return np.exp(-(offset**2) / (2.0 * width**2))

        final = _advect(gaussian(0.5)[:, None], 1.0, dt, ds, nsteps)
        errors.append(float(np.abs(final[:, 0] - gaussian(0.5 + dt * nsteps)).mean()))

    cells = np.array(resolutions, dtype=float)
    errs = np.array(errors)
    orders = np.log2(errs[:-1] / errs[1:])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(cells, errs, "o-", color=PORT, lw=2, label="cmaq-jax PPM")
    ax.loglog(
        cells,
        errs[0] * (cells / cells[0]) ** -2.0,
        "--",
        color=REFERENCE,
        lw=1.2,
        label="second order",
    )
    ax.loglog(
        cells,
        errs[0] * (cells / cells[0]) ** -3.0,
        ":",
        color=MUTED,
        lw=1.2,
        label="third order (unlimited PPM)",
    )

    for n, err, order in zip(cells[1:], errs[1:], orders, strict=True):
        ax.annotate(f"{order:.2f}", (n, err), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("cells across the domain")
    ax.set_ylabel("mean absolute error")
    ax.set_title(
        "Gaussian advected a quarter domain, Courant 0.5\n"
        "~2nd order: the limiter clips the peak, costing PPM its 3rd order there",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, "convergence")


def figure_limiter_action() -> None:
    """Where the limiter collapses the parabola to a constant, and why."""
    n = 40
    idx = np.arange(n)
    # Each profile is the same smooth background plus one difficult feature, so
    # the panels show the limiter acting *selectively* rather than a flat field
    # being trivially flat.
    smooth = 3.0 + np.sin(2.0 * np.pi * idx / n)
    profiles = {
        "smooth": smooth,
        "smooth + step": smooth + np.where(idx >= n // 2, 3.0, 0.0),
        "smooth + spike": smooth + np.where(idx == n // 2, 4.0, 0.0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, (name, cn) in zip(axes, profiles.items(), strict=True):
        padded = np.concatenate([np.repeat(cn[:1], SWP), cn, np.repeat(cn[-1:], SWP)])
        p = ppm_parabola_uniform(padded)
        cl = np.asarray(p.cl)[SWP : SWP + n]
        cr = np.asarray(p.cr)[SWP : SWP + n]

        # Render each cell's parabola across its own width.
        for cell in range(n):
            xi = np.linspace(0.0, 1.0, 12)
            dc, c6 = cr[cell] - cl[cell], 6.0 * (cn[cell] - 0.5 * (cl[cell] + cr[cell]))
            ax.plot(
                cell + xi,
                cl[cell] + xi * (dc + c6 * (1.0 - xi)),
                color=PORT,
                lw=1.0,
                zorder=2,
            )

        ax.step(idx, cn, where="mid", color=MUTED, lw=1.0, alpha=0.9, zorder=1)
        ax.plot(idx + 0.5, cn, ".", color=REFERENCE, ms=4, zorder=3)

        flat = np.isclose(cl, cr)
        ax.plot(
            idx[flat] + 0.5,
            cn[flat],
            "o",
            mfc="none",
            mec="k",
            ms=9,
            lw=1.0,
            zorder=4,
            label="limiter -> constant",
        )
        ax.set_title(f"{name}  ({flat.sum()}/{n} cells clipped flat)", fontsize=10)
        ax.set_xlabel("cell")
        if name == "smooth":
            ax.set_ylabel("concentration")
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "PPM reconstruction (eq. 1.4) per cell. Smooth regions keep the full parabola; "
        "a discontinuity or an\nisolated extremum collapses only its own neighbourhood to "
        "piecewise-constant, so the scheme cannot overshoot.",
        fontsize=10,
    )
    _save(fig, "limiter_action")


def figure_transport() -> None:
    """A Gaussian and a square wave after one full revolution."""
    n, courant = 200, 0.5
    ds = 1.0 / n
    dt = courant * ds
    nsteps = round(1.0 / dt)
    x = (np.arange(n) + 0.5) / n

    offset = (x - 0.5 + 0.5) % 1.0 - 0.5
    gaussian = np.exp(-(offset**2) / (2.0 * 0.06**2))
    square = np.where((x > 0.35) & (x < 0.65), 1.0, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, (name, initial) in zip(
        axes, {"Gaussian": gaussian, "square wave": square}.items(), strict=True
    ):
        final = _advect(initial[:, None], 1.0, dt, ds, nsteps)[:, 0]
        ax.plot(x, initial, color=MUTED, lw=2.0, label="initial / exact")
        ax.plot(x, final, color=PORT, lw=1.6, label=f"after {nsteps} steps (1 revolution)")
        ax.set_title(
            f"{name}: peak {final.max():.4f} vs {initial.max():.4f}, min {final.min():.2e}",
            fontsize=10,
        )
        ax.set_xlabel("position")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("concentration")

    fig.suptitle(
        "Solid-body advection, 200 cells, Courant 0.5. The square wave loses amplitude to "
        "numerical diffusion\nbut develops no oscillation -- no undershoot below zero, "
        "no overshoot above one.",
        fontsize=10,
    )
    _save(fig, "transport")


def figure_vertical_stretching() -> None:
    """The non-uniform reconstruction on a CMAQ-like stretched sigma grid."""
    nlays = 35
    # CMAQ sigma runs 1.0 at the ground to 0.0 at the model top, and layers are
    # thin near the surface to resolve the boundary layer. An exponent below 1
    # produces that; above 1 would put the thickest layer at the ground.
    faces = np.linspace(1.0, 0.0, nlays + 1) ** 0.625
    ds = np.abs(np.diff(faces))
    mesh = nonuniform_mesh(ds)

    z = np.arange(nlays)
    profile = 2.0 + np.exp(-((z - 8.0) ** 2) / (2.0 * 3.0**2)) * 4.0
    p = ppm_parabola_nonuniform(profile, mesh)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")

    axes[0].barh(z, ds, color=PORT, height=0.8)
    axes[0].set_xlabel("layer thickness (sigma)")
    axes[0].set_ylabel("layer")
    axes[0].set_title(
        f"stretched sigma grid: {ds.max() / ds.min():.0f}x thicker aloft than at the surface",
        fontsize=10,
    )

    axes[1].plot(profile, z, "o-", color=MUTED, ms=4, lw=1.2, label="cell mean")
    axes[1].plot(np.asarray(p.cl), z, ".", color=PORT, label="left edge")
    axes[1].plot(np.asarray(p.cr), z, ".", color=REFERENCE, label="right edge")
    axes[1].set_xlabel("concentration")
    axes[1].set_title("reconstruction (vppm.F:472-541)", fontsize=10)
    axes[1].legend(fontsize=8)

    axes[2].plot(np.asarray(p.dc), z, "-", color=PORT, label="dc (slope)")
    axes[2].plot(np.asarray(p.c6), z, "-", color=REFERENCE, label="c6 (curvature)")
    axes[2].axvline(0.0, color="k", lw=0.8, alpha=0.4)
    axes[2].set_xlabel("coefficient")
    axes[2].set_title("slope and curvature", fontsize=10)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Non-uniform PPM on 35 CMAQ-like sigma layers. The ds-weighted eq. (1.7)/(1.6) forms "
        "keep the\nreconstruction bounded even where neighbouring layers differ several-fold "
        "in thickness.",
        fontsize=10,
    )
    _save(fig, "vertical_stretching")


def main() -> None:
    figure_fortran_agreement()
    figure_convergence()
    figure_limiter_action()
    figure_transport()
    figure_vertical_stretching()


if __name__ == "__main__":
    main()
