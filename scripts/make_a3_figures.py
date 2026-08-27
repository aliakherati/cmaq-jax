#!/usr/bin/env python3
"""A3 figures: the assembled operator.

    python scripts/make_a3_figures.py

Two figures:

* ``adjoint_footprint`` -- what differentiability is *for*. One reverse pass
  answers "which upwind cells did the pollution at this receptor come from?",
  and the same pass is checked against finite differences.
* ``scaling`` -- cost against domain size, species count and the vertical
  sub-step cap.
"""

from __future__ import annotations

import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from cmaq_jax.advstep import DEFAULT_LIMITS, advstep, sync_top_layer, wind_index
from cmaq_jax.api import Meteorology, advect_step
from cmaq_jax.config import DEFAULT_PPM, GridConfig, sigma_layer_thickness
from cmaq_jax.hadv import BoundaryConditions, advance_xyfirst
from cmaq_jax.ppm import nonuniform_mesh

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "figures" / "a3"

FIELD_CMAP = "magma"
PORT = "#1b6ca8"
ACCENT = "#d1495b"
MUTED = "#8d99ae"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches=None if fig.get_layout_engine() else "tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


def build(ncols: int, nrows: int, nlays: int, nspc: int, *, substeps: int = 4):
    """A domain with a westerly, slightly veering wind."""
    faces = np.linspace(1.0, 0.0, nlays + 1) ** 0.625
    ds = sigma_layer_thickness(faces)
    cfg = GridConfig(
        ncols=ncols,
        nrows=nrows,
        ds=ds,
        dx1=12000.0,
        dx2=12000.0,
        nspc_adv=nspc,
        ppm=replace(DEFAULT_PPM, max_substeps=substeps),
    )
    rng = np.random.default_rng(20260913)
    rhoj = 1.5 + 0.4 * rng.random((ncols, nrows, nlays))
    # A positive background, not zero. PPM's limiter and the outflow condition
    # both clamp at zero, so a field sitting exactly there is a kink: a central
    # difference straddles it and the check is meaningless. See the note in
    # figure_adjoint_footprint.
    state = np.stack([np.ones_like(rhoj)] * (nspc - 1) + [rhoj], axis=-1)

    profile = np.linspace(1.0, 2.0, nlays)
    uhat = 10.0 * profile[None, None, :] * np.ones((ncols + 1, nrows, 1))
    vhat = 4.0 * profile[None, None, :] * np.ones((ncols, nrows + 1, 1))

    edge = np.ones(nspc)
    edge[-1] = 2.0
    bcon = BoundaryConditions(
        *(np.broadcast_to(edge, (n, nlays, nspc)) for n in (nrows, nrows, ncols, ncols))
    )
    met = Meteorology(uhat=uhat, vhat=vhat, rhoj_met=rhoj * 1.01, bcon=bcon)

    schedule = advstep(
        wind_index(uhat, vhat, cfg.dx1, cfg.dx2),
        np.zeros(nlays),
        3600,
        DEFAULT_LIMITS,
        sync_layers=sync_top_layer(faces, DEFAULT_LIMITS.sigma_sync_top),
    )
    return cfg, nonuniform_mesh(ds), jnp.asarray(state), met, schedule


def figure_adjoint_footprint() -> None:
    """Where did the pollution at one receptor come from?

    ``jax.grad`` of a receptor concentration with respect to the initial field
    is a source-receptor footprint: one reverse pass, no adjoint code written
    by hand. CMAQ reaches the same question through DDM-3D, a separate model
    configuration; here it falls out of the forward code.

    The background is deliberately positive. PPM's limiter and the outflow
    condition both clamp at zero, so a tracer sitting exactly there is at a
    kink in the operator: the gradient is a one-sided derivative and a central
    finite difference straddles the corner. Checked on a zero background the
    disagreement is 48% of peak; on a positive one it is 6e-5. The gradient was
    never the problem -- the check was.
    """
    ncols, nrows, nlays, nspc = 48, 40, 6, 2
    nsteps = 12
    cfg, mesh, state, met, schedule = build(ncols, nrows, nlays, nspc)

    receptor = (36, 22, 0)

    def concentration_at_receptor(initial):
        current = initial
        xyfirst = (True,) * nlays
        for _ in range(nsteps):
            current, _ = advect_step(
                current,
                met,
                mesh,
                cfg=cfg,
                astep_seconds=schedule.astep_seconds,
                sync_seconds=schedule.sync_seconds,
                xyfirst=xyfirst,
            )
            xyfirst = advance_xyfirst(xyfirst, schedule.astep_seconds, schedule.sync_seconds)
        return current[(*receptor, 0)]

    footprint = np.asarray(jax.grad(concentration_at_receptor)(state))[..., 0]

    # Verify a sample of the footprint against finite differences.
    rng = np.random.default_rng(4)
    candidates = np.argwhere(np.abs(footprint) > 1e-2 * np.abs(footprint).max())
    sample = candidates[rng.choice(len(candidates), size=min(40, len(candidates)), replace=False)]
    eps = 1e-4
    analytic, numeric = [], []
    for cell in sample:
        index = (int(cell[0]), int(cell[1]), int(cell[2]), 0)
        analytic.append(float(footprint[tuple(cell)]))
        plus = concentration_at_receptor(state.at[index].add(eps))
        minus = concentration_at_receptor(state.at[index].add(-eps))
        numeric.append(float((plus - minus) / (2.0 * eps)))
    analytic_arr, numeric_arr = np.array(analytic), np.array(numeric)
    # Scaled by the peak sensitivity, not pointwise. A cell whose true
    # sensitivity is a millionth of the peak has a finite difference dominated
    # by rounding, and dividing by it manufactures an enormous ratio out of an
    # entirely healthy gradient.
    peak = float(np.abs(footprint).max())
    worst = float(np.max(np.abs(analytic_arr - numeric_arr))) / peak

    surface = footprint[:, :, 0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    positive = surface[surface > 0]
    floor = float(positive.max()) * 1e-4 if positive.size else 1e-12
    image = axes[0].pcolormesh(
        np.arange(ncols),
        np.arange(nrows),
        np.maximum(surface, floor).T,
        cmap=FIELD_CMAP,
        norm=LogNorm(vmin=floor, vmax=max(float(surface.max()), floor * 10)),
        shading="auto",
    )
    axes[0].plot(receptor[0], receptor[1], "o", mfc="none", mec="w", ms=12, mew=2)
    axes[0].annotate(
        "receptor",
        (receptor[0], receptor[1]),
        color="w",
        fontsize=9,
        xytext=(-52, 10),
        textcoords="offset points",
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")
    axes[0].set_title("source footprint, surface layer", fontsize=10)
    fig.colorbar(image, ax=axes[0], shrink=0.85, label="d(receptor) / d(initial)")

    by_layer = np.abs(footprint).sum(axis=(0, 1))
    axes[1].barh(np.arange(nlays), by_layer / by_layer.sum(), color=PORT)
    axes[1].set_xlabel("share of total sensitivity")
    axes[1].set_ylabel("layer")
    axes[1].set_title("which layers the receptor draws from", fontsize=10)
    axes[1].grid(axis="x", alpha=0.25)

    limit = max(float(np.abs(numeric_arr).max()), 1e-30)
    axes[2].plot(
        [-limit, limit], [-limit, limit], "-", color=MUTED, lw=1.2, label="exact agreement"
    )
    axes[2].plot(numeric_arr, analytic_arr, "o", color=ACCENT, ms=5, alpha=0.75)
    axes[2].set_xlabel("central finite difference")
    axes[2].set_ylabel("jax.grad")
    axes[2].set_title(
        f"{len(sample)} sampled cells\nworst disagreement {worst:.1e} of peak sensitivity",
        fontsize=10,
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "One reverse pass answers a source-receptor question: which upwind cells the concentration "
        "at a chosen cell came\nfrom, after "
        f"{nsteps} sync steps. No adjoint was written -- this is the forward code differentiated. "
        "The wind is westerly and\nveering, so the footprint reaches upwind and tilts across rows, "
        "and it spreads with height because the faster flow aloft\nreaches further back. The right "
        "panel checks the same gradient against central finite differences, on the cells\nwhere "
        "that check is itself reliable.",
        fontsize=10,
    )
    _save(fig, "adjoint_footprint")


def figure_scaling() -> None:
    """Cost against domain size, species count and the sub-step cap."""

    def measure(cfg, mesh, state, met, schedule, *, repeats: int = 3) -> float:
        fn = jax.jit(
            partial(
                advect_step,
                mesh=mesh,
                cfg=cfg,
                astep_seconds=schedule.astep_seconds,
                sync_seconds=schedule.sync_seconds,
                xyfirst=(True,) * cfg.nlays,
            )
        )
        warm = fn(state, met)
        jax.block_until_ready(jax.tree.leaves(warm)[0])
        start = time.perf_counter()
        for _ in range(repeats):
            out = fn(state, met)
        jax.block_until_ready(jax.tree.leaves(out)[0])
        return (time.perf_counter() - start) / repeats * 1e3

    sizes = [24, 40, 60, 80, 100]
    size_ms = [measure(*build(n, n, 20, 10)) for n in sizes]

    species = [2, 5, 10, 20, 40]
    species_ms = [measure(*build(60, 60, 20, s)) for s in species]

    caps = [1, 2, 4, 8, 16, 30]
    cap_ms = [measure(*build(60, 60, 20, 10, substeps=c)) for c in caps]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout="constrained")

    cells = np.array(sizes) ** 2 * 20 * 10
    axes[0].loglog(cells, size_ms, "o-", color=PORT, lw=1.8, ms=5, label="measured")
    axes[0].loglog(
        cells,
        size_ms[0] * cells / cells[0],
        "--",
        color=MUTED,
        lw=1.2,
        label="linear in cells",
    )
    axes[0].set_xlabel("cell-species in the domain")
    axes[0].set_ylabel("ms per sync step")
    axes[0].set_title("domain size (20 layers, 10 species)", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(species, species_ms, "o-", color=PORT, lw=1.8, ms=5)
    throughput = np.array(species) * 60 * 60 * 20 / np.array(species_ms) / 1e3
    twin = axes[1].twinx()
    twin.plot(species, throughput, "s--", color=ACCENT, lw=1.4, ms=4)
    twin.set_ylabel("M cell-species / s", color=ACCENT, fontsize=9)
    twin.tick_params(axis="y", colors=ACCENT, labelsize=8)
    axes[1].set_xlabel("advected species")
    axes[1].set_ylabel("ms per sync step", color=PORT)
    axes[1].set_title("species count: throughput improves\nas overhead amortises", fontsize=10)

    axes[2].plot(caps, cap_ms, "o-", color=PORT, lw=1.8, ms=5)
    axes[2].axvline(2, color=ACCENT, ls="--", lw=1.2, label="sub-steps actually needed")
    axes[2].set_xlabel("max_substeps")
    axes[2].set_ylabel("ms per sync step")
    axes[2].set_title("the vertical sub-step cap\nis paid for whether used or not", fontsize=10)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        "Cost of one sync step on CPU. Time is close to linear in the number of\n"
        "cell-species, which is what a flux-form scheme should give. The sub-step cap is a "
        "fixed loop trip count,\nso a column pays for all of it even when it needs two -- "
        "lowering it is safe because the residual\ndiagnostic reports any column that ran out.",
        fontsize=10,
    )
    _save(fig, "scaling")


def main() -> None:
    figure_adjoint_footprint()
    figure_scaling()


if __name__ == "__main__":
    main()
