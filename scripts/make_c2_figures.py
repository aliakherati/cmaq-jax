#!/usr/bin/env python3
"""C2 figures: the ACM2 operator in action.

    python scripts/make_c2_figures.py

Two figures:

* ``acm2_vs_local`` -- what the non-local plume buys. The same surface release
  in a convective and a stable column, side by side, and the vertical profile
  over time.
* ``model_top_leak`` -- the local stage's top boundary is closed only because
  the diffusivity is zero there. Measured, and reproduced by the Fortran.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cmaq_jax.vdiff import ColumnState, SurfaceExchange, substep_counts, vdiff_step

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures" / "c2"
NLAYS = 24
LPBL = 14


def column(*, convective: bool, seddy: float = 40.0, top: float = 2400.0):
    face = np.linspace(50.0, top, NLAYS)
    shape2, shape3 = (1, 1), (1, 1, NLAYS)
    kz = np.full(shape3, seddy)
    kz[..., -1] = 0.0  # what eddyx.F returns; see model_top_leak()

    state = ColumnState(
        seddy=kz,
        zf=np.broadcast_to(face, shape3) + 0.0,
        zh=np.broadcast_to(face - 25.0, shape3) + 0.0,
        pbl=np.full(shape2, float(face[LPBL - 1])),
        lpbl=np.full(shape2, LPBL, dtype=np.int32),
        hol=np.full(shape2, -4.0 if convective else 6.0),
        dens1=np.full(shape2, 1.2),
        rdepvht=np.full(shape2, 1.0 / face[0]),
        convective=np.full(shape2, convective),
    )
    surface = SurfaceExchange(
        depv=np.full((1, 1, 1), 1.0e-12),
        pldv=np.zeros((1, 1, 1)),
        emis=np.zeros((1, 1, NLAYS, 1)),
    )
    conc = np.zeros((1, 1, NLAYS, 1))
    conc[0, 0, 0, 0] = 100.0
    dzh = np.concatenate([[face[0]], np.diff(face)])
    return conc, state, surface, face, dzh


def evolve(convective: bool, *, steps: int, dtsec: float = 300.0):
    conc, state, surface, face, _dzh = column(convective=convective)
    bound = int(np.asarray(substep_counts(state, dtsec)).max())
    history = [conc[0, 0, :, 0].copy()]
    for _ in range(steps):
        conc, _ = vdiff_step(conc, state, surface, dtsec=dtsec, max_substeps=bound)
        conc = np.asarray(conc)
        history.append(conc[0, 0, :, 0].copy())
    return np.array(history), face, _dzh


def acm2_vs_local() -> None:
    steps = 12
    convective, face, _dzh = evolve(True, steps=steps)
    stable, _, _ = evolve(False, steps=steps)
    pbl_height = face[LPBL - 1]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.4))
    hours = np.arange(steps + 1) * 300.0 / 3600.0

    for ax, data, title in (
        (axes[0], stable.T, "stable — local diffusion only"),
        (axes[1], convective.T, "convective — ACM2, with the non-local plume"),
    ):
        im = ax.pcolormesh(hours, face, data, shading="nearest", cmap="inferno", vmin=0, vmax=25)
        ax.axhline(pbl_height, color="w", ls="--", lw=1.2)
        ax.annotate(
            "PBL top", (hours[-1], pbl_height), color="w", fontsize=8, ha="right", va="bottom"
        )
        ax.set_xlabel("hours")
        ax.set_ylabel("height (m)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    for data, label, colour in (
        (convective, "convective (ACM2)", "C3"),
        (stable, "stable (local only)", "C0"),
    ):
        ax.plot(data[4], face, "o-", ms=3, color=colour, label=f"{label}, t = 20 min")
        ax.plot(data[-1], face, "s--", ms=3, color=colour, alpha=0.55, label=f"{label}, t = 1 h")
    ax.axhline(pbl_height, color="0.4", ls="--", lw=1.0)
    ax.set_xlabel("concentration")
    ax.set_ylabel("height (m)")
    ax.set_title("profiles", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "A surface release, one hour. The non-local plume carries mass from the surface layer "
        "directly to the\ntop of the convective boundary layer, rather than diffusing through "
        "every layer in between.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "acm2_vs_local.png", dpi=140)
    plt.close(fig)


def model_top_leak() -> None:
    """The top boundary is closed only by the diffusivity vanishing there."""
    conc, state, surface, face, dzh = column(convective=False)
    # A perfectly uniform column: with a closed top, nothing should happen at all.
    conc = np.full_like(conc, 7.0)
    initial = float((conc[0, 0, :, 0] * dzh).sum())

    tops = np.array([0.0, 0.01, 0.1, 1.0, 5.0, 20.0])
    drifts = []
    for ktop in tops:
        kz = state.seddy.copy()
        kz[..., -1] = ktop
        leaky = state._replace(seddy=kz)
        bound = int(np.asarray(substep_counts(leaky, 300.0)).max())
        out, _ = vdiff_step(conc, leaky, surface, dtsec=300.0, max_substeps=bound)
        final = float((np.asarray(out)[0, 0, :, 0] * dzh).sum())
        drifts.append(100.0 * (final - initial) / initial)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    ax.plot(tops, drifts, "o-", color="C3")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.annotate(
        "what `eddyx.F` actually\nreturns for the top layer",
        xy=(0.0, drifts[0]),
        xytext=(5.0, 0.45 * drifts[-1]),
        fontsize=8,
        ha="left",
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_xlabel("$K_z$ in the top layer (m$^2$ s$^{-1}$)")
    ax.set_ylabel("mass change in one 300 s step (%)")
    ax.set_title("a uniform column should not change at all", fontsize=10, pad=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for ktop, colour in ((0.0, "C0"), (20.0, "C3")):
        kz = state.seddy.copy()
        kz[..., -1] = ktop
        leaky = state._replace(seddy=kz)
        bound = int(np.asarray(substep_counts(leaky, 300.0)).max())
        out, _ = vdiff_step(conc, leaky, surface, dtsec=300.0, max_substeps=bound)
        ax.plot(
            np.asarray(out)[0, 0, :, 0],
            face,
            "o-",
            ms=4,
            color=colour,
            label=f"$K_z$(top) = {ktop:g}",
        )
    ax.axvline(7.0, color="0.4", ls="--", lw=1.0, label="initial (uniform)")
    ax.set_xlabel("concentration")
    ax.set_ylabel("height (m)")
    ax.set_title("the loss is entirely at the model top", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "`BB2(NLAYS)` carries an upward-flux term with no matching right-hand side, so the top "
        "row is a one-sided sink.\nIt never bites because `eddyx.F` returns zero there — the "
        "scheme is conservative by coincidence of the two.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "model_top_leak.png", dpi=140)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    acm2_vs_local()
    model_top_leak()
    print(f"wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
