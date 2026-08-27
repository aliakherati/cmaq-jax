#!/usr/bin/env python3
"""C1 figures: the ACM2 solvers and the eddy diffusivity.

    python scripts/make_c1_figures.py

Two figures:

* ``matrix_structure`` -- why ACM2 needs two solvers. The local stage is
  tridiagonal; the convective stage is tridiagonal *plus a full first column*,
  because the non-local plume connects the surface layer to every layer in the
  convective boundary layer at once.
* ``eddy_diffusivity`` -- the Kz parameterization across its three regimes,
  with the neutral surface-layer profile checked against its closed form.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from cmaq_jax.config import DEFAULT_ACM2
from cmaq_jax.vdiff import VerticalMeteorology, eddy_diffusivity

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures" / "c1"


def matrix_structure() -> None:
    """The two matrices, side by side.

    The picture is the argument for `matrix1.F` existing at all: a Thomas solver
    assumes a banded matrix, and the ACM2 convective stage is not one.
    """
    n, kl = 14, 9
    local = np.zeros((n, n))
    for k in range(n):
        local[k, k] = 2.0
        if k > 0:
            local[k, k - 1] = 1.0
        if k < n - 1:
            local[k, k + 1] = 1.0

    convective = np.zeros((n, n))
    convective[0, 0] = 2.0
    convective[0, 1] = 1.0
    for k in range(1, kl):
        convective[k, 0] = 1.5
        convective[k, k] = 2.0
        if k < kl - 1:
            convective[k, k + 1] = 1.0
    # Layers above the CBL top take no part; the identity keeps them untouched.
    for k in range(kl, n):
        convective[k, k] = 2.0

    cmap = ListedColormap(["#f7f7f7", "#9ecae1", "#08519c", "#e6550d"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))

    for ax, matrix, title in (
        (axes[0], local, "local stage — tridiagonal\nsolved by `tri.F` (Thomas)"),
        (
            axes[1],
            convective,
            "convective stage — tridiagonal + first column\n"
            f"solved by `matrix1.F`, over rows 1..KL (KL={kl})",
        ),
    ):
        shown = np.where(
            matrix == 0.0, 0, np.where(matrix == 1.5, 3, np.where(matrix == 2.0, 2, 1))
        )
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks(range(0, n, 2))
        ax.set_yticks(range(0, n, 2))
        ax.set_xlabel("column (layer)")
        ax.set_ylabel("row (layer)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)

    axes[1].axhline(kl - 0.5, color="k", lw=1.2, ls="--")
    axes[1].annotate(
        "CBL top: rows above are untouched\n(and KL varies per column)",
        xy=(n - 1, kl - 0.5),
        xytext=(n - 1, kl + 2.6),
        fontsize=8,
        ha="right",
    )
    axes[1].annotate(
        "the non-local plume:\nevery CBL layer exchanges\ndirectly with the surface",
        xy=(0, kl - 2),
        xytext=(3.2, kl - 1.2),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    fig.suptitle(
        "Why ACM2 needs two solvers. A Thomas sweep assumes a banded matrix; "
        "the convective stage is not banded.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "matrix_structure.png", dpi=140)
    plt.close(fig)


def column(moli: float, *, pbl: float = 1200.0, ustar: float = 0.35, nlays: int = 40):
    """A single column, uniform in the horizontal, at one stability."""
    ncols = nrows = 3
    face = np.linspace(20.0, 2600.0, nlays)
    middle = face - 0.5 * (face[1] - face[0])
    shape2, shape3 = (ncols, nrows), (ncols, nrows, nlays)
    dot = (ncols + 1, nrows + 1, nlays)
    across = np.linspace(0.0, 1.0, ncols + 1)[:, None, None]
    return VerticalMeteorology(
        pbl=np.full(shape2, pbl),
        ustar=np.full(shape2, ustar),
        moli=np.full(shape2, moli),
        zf=np.broadcast_to(face, shape3) + 0.0,
        zh=np.broadcast_to(middle, shape3) + 0.0,
        kzmin=np.full(shape3, 0.5),
        thetav=np.broadcast_to(300.0 + 0.4 * np.arange(nlays), shape3) + 0.0,
        ta=np.full(shape3, 288.0),
        qv=np.full(shape3, 0.006),
        qc=np.zeros(shape3),
        uwind=np.broadcast_to(0.8 * np.arange(nlays), dot) * (1.0 + across),
        vwind=np.zeros(dot),
    ), face


def eddy_profile() -> None:
    pbl = 1200.0
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4))

    # --- the three regimes ------------------------------------------------
    ax = axes[0]
    for moli, label, colour in (
        (-0.02, "unstable  ($1/L = -0.02$)", "C3"),
        (0.0, "neutral  ($1/L = 0$)", "C0"),
        (0.01, "stable  ($1/L = +0.01$)", "C1"),
    ):
        met, face = column(moli, pbl=pbl)
        kz = np.asarray(eddy_diffusivity(met))[1, 1, :-1]
        ax.plot(kz, face[:-1], "o-", ms=3, color=colour, label=label)
    ax.axhline(pbl, color="0.4", ls="--", lw=1.0)
    ax.annotate("PBL top", (ax.get_xlim()[1], pbl), fontsize=8, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("$K_z$ (m$^2$ s$^{-1}$)")
    ax.set_ylabel("height (m)")
    ax.set_title("Three stability regimes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # --- the neutral case against its closed form -------------------------
    ax = axes[1]
    met, face = column(0.0, pbl=pbl)
    kz = np.asarray(eddy_diffusivity(met))[1, 1, :-1]
    z = face[:-1]
    analytic = DEFAULT_ACM2.karman * 0.35 * z * (1.0 - z / pbl) ** 2
    ax.plot(kz, z, "o", ms=4, color="C0", label="cmaq-jax")
    ax.plot(
        np.where(z < pbl, analytic, np.nan),
        z,
        "-",
        color="C3",
        lw=2,
        label=r"$\kappa u_* z (1 - z/h)^2$",
    )
    ax.axhline(pbl, color="0.4", ls="--", lw=1.0)
    ax.set_xlabel("$K_z$ (m$^2$ s$^{-1}$)")
    ax.set_ylabel("height (m)")
    ax.set_title(
        "Neutral: exact in closed form\n"
        "no free parameters — this is what pins the port to the scheme",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- the stability function -------------------------------------------
    ax = axes[2]
    zol = np.linspace(-2.0, 3.0, 800)
    phih = np.where(
        zol < 0.0,
        1.0 / np.sqrt(1.0 - DEFAULT_ACM2.gamah * np.minimum(zol, 0.0)),
        np.where(zol < 1.0, 1.0 + DEFAULT_ACM2.betah * zol, DEFAULT_ACM2.betah + zol),
    )
    ax.plot(zol, phih, lw=2, color="C0")
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.axvline(1.0, color="C1", ls=":", lw=1.2)
    ax.annotate(
        "branch at $z/L = 1$\n(a mild case never reaches it —\nhence the very_stable golden)",
        xy=(1.0, DEFAULT_ACM2.betah + 1.0),
        xytext=(1.15, 3.0),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.annotate("unstable", (-1.4, 0.45), fontsize=9, color="C3")
    ax.annotate("stable", (1.8, 1.2), fontsize=9, color="C1")
    ax.set_xlabel("$z/L$")
    ax.set_ylabel(r"$\phi_h$")
    ax.set_title(r"The stability function. $K_z \propto u_*/\phi_h$", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Vertical eddy diffusivity (`eddyx.F`). Matched to the Fortran across 10 cases, "
        "worst 3.0 float32 ULPs.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "eddy_diffusivity.png", dpi=140)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    matrix_structure()
    eddy_profile()
    print(f"wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
