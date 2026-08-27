#!/usr/bin/env python3
"""Build the CMAQ PPM kernels and emit reference ("golden") arrays.

    python scripts/generate_goldens.py            # regenerate data/goldens/
    python scripts/generate_goldens.py --check    # fail if committed goldens drift

Each case is run in its **own process**. That is not tidiness -- ``hppm.F`` and
``vppm.F`` allocate work arrays on their first call, size them from the
arguments, and ``SAVE`` them (``hppm.F:225-246``, ``vppm.F:174-177``,
``vppm.F:450-468``). A second call with a different ``NI``, ``NSPCS`` or ``DS``
silently reuses the first shape and returns wrong numbers, with no error.

Each ``.npz`` holds the case's **inputs and outputs**, so the regression tests
run without a Fortran toolchain.

All arrays are ``float32`` -- CMAQ's default ``REAL`` -- and stay that way. The
JAX port runs in float64 and downcasts for comparison via
``atmos_jax_common.real4``.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "reference"
BUILD = REFERENCE / "build"
GOLDENS = REPO / "data" / "goldens"

# Halo width; must match SWP in hppm.F:147 and the harness.
SWP = 3

F32 = np.dtype(np.float32)
I32 = np.dtype(np.int32)


# --------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HppmCase:
    """One call to HPPM: a 1-D row of cells swept by a face-velocity field."""

    name: str
    ni: int
    nspcs: int
    dt: float
    ds: float
    con: NDArray[np.float32]  # (ni + 2*SWP, nspcs), index 0 is cell 1-SWP
    vel: NDArray[np.float32]  # (ni + 1,)
    ori: str = "C"
    note: str = ""

    def __post_init__(self) -> None:
        assert self.con.shape == (self.ni + 2 * SWP, self.nspcs), self.name
        assert self.vel.shape == (self.ni + 1,), self.name
        assert self.ori in ("C", "R"), self.name


@dataclass(frozen=True)
class VppmCase:
    """One call to VPPM: a single vertical column.

    ``vel`` is modified in place by the kernel's flux-matching adjustment
    (``vppm.F:200-246``), so the adjusted velocity is part of the golden.
    """

    name: str
    ni: int
    nspcs: int
    dt: float
    ds: NDArray[np.float32]  # (ni,)   layer thicknesses, sigma
    flx: NDArray[np.float32]  # (ni+1,) face mass fluxes
    vel: NDArray[np.float32]  # (ni+1,) face velocities
    con: NDArray[np.float32]  # (ni, nspcs); slot nspcs-1 is rho*J
    note: str = ""

    def __post_init__(self) -> None:
        assert self.ds.shape == (self.ni,), self.name
        assert self.flx.shape == (self.ni + 1,), self.name
        assert self.vel.shape == (self.ni + 1,), self.name
        assert self.con.shape == (self.ni, self.nspcs), self.name


@dataclass(frozen=True)
class PpmCoeffCase:
    """One call to PPM, the non-uniform reconstruction inside ``vppm.F``.

    Pins the parabola on its own, without the flux-matching velocity
    adjustment that ``VPPM`` wraps around it.
    """

    name: str
    ni: int
    dt: float
    ds: NDArray[np.float32]  # (ni,)
    cn: NDArray[np.float32]  # (ni,)
    note: str = ""

    def __post_init__(self) -> None:
        assert self.ds.shape == (self.ni,), self.name
        assert self.cn.shape == (self.ni,), self.name


def _halo(profile: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pad an interior profile with SWP ghost cells by edge replication.

    A zeroth-order (constant) extrapolation, which is what CMAQ's outflow BC
    reduces to when the edge gradient is flat (``zfdbc.f:36``).
    """
    return np.concatenate([np.full(SWP, profile[0]), profile, np.full(SWP, profile[-1])])


def hppm_cases() -> list[HppmCase]:
    """Cases chosen to exercise every branch of the limiter and flux code."""
    ni, nspcs = 24, 3
    x = np.arange(ni)
    cases: list[HppmCase] = []

    def build(
        name: str,
        profiles: list[NDArray[np.float64]],
        vel: NDArray[np.float64],
        *,
        dt: float = 60.0,
        ds: float = 12000.0,
        ori: str = "C",
        note: str = "",
    ) -> HppmCase:
        con = np.stack([_halo(p) for p in profiles], axis=1)
        return HppmCase(
            name=name,
            ni=ni,
            nspcs=len(profiles),
            dt=dt,
            ds=ds,
            con=con.astype(F32),
            vel=vel.astype(F32),
            ori=ori,
            note=note,
        )

    rng = np.random.default_rng(20260826)
    smooth = [
        1.0 + 0.5 * np.sin(2 * np.pi * x / ni),
        2.0 + 0.3 * np.cos(4 * np.pi * x / ni),
        1.0 + 0.1 * rng.random(ni),
    ]
    # Courant = vel*dt/ds; 100 m/s * 60 s / 12000 m = 0.5
    u_pos = np.full(ni + 1, 100.0)
    u_neg = np.full(ni + 1, -100.0)

    cases.append(build("smooth_positive_wind", smooth, u_pos, note="generic interior case"))
    cases.append(
        build("smooth_negative_wind", smooth, u_neg, note="exercises the FM/left-face branch")
    )

    step = np.where(x < ni // 2, 1.0, 5.0)
    cases.append(
        build(
            "step_positive_wind",
            [step, step[::-1].copy(), np.full(ni, 3.0)],
            u_pos,
            note="discontinuity; limiter must not overshoot",
        )
    )

    spike = np.full(ni, 1.0)
    spike[ni // 2] = 10.0
    cases.append(
        build(
            "spike",
            [spike, np.full(ni, 1.0), spike[::-1].copy()],
            u_pos,
            note="isolated extremum; parabola collapses to a constant",
        )
    )

    # Constancy preservation, the CMAQ-specific invariant. The state must be in
    # COUPLED units -- slot s holds rho*J*q_s and the last slot holds rho*J --
    # and the wind must be divergent, or the test is trivially satisfied. After
    # advection every q_s = con[:, s] / con[:, -1] must come back unchanged.
    rhoj = 1.0 + 0.4 * np.sin(2 * np.pi * x / ni)
    divergent = 120.0 * np.sin(2 * np.pi * np.arange(ni + 1) / (ni + 1))
    cases.append(
        build(
            "constancy_divergent_wind",
            [0.75 * rhoj, 2.0 * rhoj, rhoj],
            divergent,
            note="coupled units, divergent wind; q = con[:,s]/con[:,-1] must be preserved",
        )
    )

    cases.append(
        build(
            "zero_wind",
            smooth,
            np.zeros(ni + 1),
            note="no transport: output must equal input exactly",
        )
    )

    # Sign change mid-domain: divergence at the reversal, convergence elsewhere.
    reversing = 100.0 * np.sin(2 * np.pi * np.arange(ni + 1) / (ni + 1))
    cases.append(build("reversing_wind", smooth, reversing, note="both flux branches in one call"))

    # Courant = 190*60/12000 = 0.95
    cases.append(
        build(
            "near_cfl_one",
            smooth,
            np.full(ni + 1, 190.0),
            note="Courant 0.95, just inside the stability limit",
        )
    )

    cases.append(
        build(
            "row_orientation",
            smooth,
            u_pos,
            ori="R",
            note="ORI='R' (y-sweep); must match the 'C' result",
        )
    )

    # A single species is a distinct code path in the S loops.
    cases.append(build("single_species", [smooth[0]], u_pos, note="nspcs = 1"))

    assert nspcs == 3  # the multi-species cases above
    return cases


def _upwind_velocity(flx: NDArray[np.float64], rhoj: NDArray[np.float64]) -> NDArray[np.float64]:
    """Face velocity from face flux, upwinded on the sign of the flux.

    Ports ``zadvppmwrf.F:373-381``: ``VEL(L) = FLX(L)/RJT(L-1)`` when the flux
    is non-negative, else ``FLX(L)/RJT(L)``. Keeps ``flx`` and ``vel``
    mutually consistent, which is what the kernel's adjustment loop expects --
    inconsistent inputs make it iterate to its cap and call ``M3EXIT``.
    """
    n = rhoj.size
    vel = np.zeros(n + 1)
    for lvl in range(1, n):  # faces 2..NI in Fortran indexing
        vel[lvl] = flx[lvl] / (rhoj[lvl - 1] if flx[lvl] >= 0.0 else rhoj[lvl])
    vel[n] = flx[n] / rhoj[n - 1]
    vel[0] = 0.0  # impermeable surface, zadvppmwrf.F:339
    return vel


def vppm_cases() -> list[VppmCase]:
    """Vertical cases, including the stretched sigma grid CMAQ actually uses."""
    nlays, nspcs = 12, 4
    cases: list[VppmCase] = []
    rng = np.random.default_rng(20260827)

    uniform_ds = np.full(nlays, 1.0 / nlays)
    # Thin near the surface, thick aloft -- the usual CMAQ stretching.
    stretched_ds = np.diff(np.linspace(0.0, 1.0, nlays + 1) ** 1.7)
    assert stretched_ds.size == nlays

    def build(
        name: str,
        ds: NDArray[np.float64],
        profiles: list[NDArray[np.float64]],
        flx_shape: NDArray[np.float64],
        *,
        dt: float = 60.0,
        note: str = "",
    ) -> VppmCase:
        rhoj = 1.0 + 0.5 * np.linspace(1.0, 0.2, nlays)  # decreasing with height
        con = np.stack([*profiles, rhoj], axis=1)
        flx = flx_shape.copy()
        flx[0] = 0.0  # impermeable surface
        vel = _upwind_velocity(flx, rhoj)
        return VppmCase(
            name=name,
            ni=nlays,
            nspcs=con.shape[1],
            dt=dt,
            ds=ds.astype(F32),
            flx=flx.astype(F32),
            vel=vel.astype(F32),
            con=con.astype(F32),
            note=note,
        )

    z = np.arange(nlays)
    smooth = [
        1.0 + 0.4 * np.sin(np.pi * z / nlays),
        3.0 - 0.1 * z,
        1.0 + 0.2 * rng.random(nlays),
    ]
    step = np.where(z < nlays // 2, 1.0, 4.0)
    spike = np.full(nlays, 1.0)
    spike[nlays // 2] = 8.0

    # Gentle upward flux; magnitude keeps the Courant number well under 1 on
    # the thinnest layer.
    gentle = 2.0e-4 * np.sin(np.pi * np.arange(nlays + 1) / nlays)

    cases.append(build("smooth_uniform_ds", uniform_ds, smooth, gentle, note="uniform layers"))
    cases.append(
        build(
            "smooth_stretched_ds",
            stretched_ds,
            smooth,
            gentle,
            note="non-uniform mesh coefficients exercised",
        )
    )
    cases.append(
        build(
            "step_stretched_ds",
            stretched_ds,
            [step, step[::-1].copy(), np.full(nlays, 2.0)],
            gentle,
            note="discontinuity on a stretched grid",
        )
    )
    cases.append(
        build(
            "spike_stretched_ds",
            stretched_ds,
            [spike, np.full(nlays, 1.0), spike[::-1].copy()],
            gentle,
            note="isolated extremum",
        )
    )
    # Constancy preservation in coupled units. `build` appends rho*J as the last
    # slot, so passing q_s * rhoj here makes every mixing ratio uniform; after
    # advection con[:, s] / con[:, -1] must return q_s. A uniform *coupled*
    # field would not test anything, since q would then vary with rho*J.
    rhoj_profile = 1.0 + 0.5 * np.linspace(1.0, 0.2, nlays)
    cases.append(
        build(
            "constancy_coupled",
            stretched_ds,
            [0.75 * rhoj_profile, 2.0 * rhoj_profile, 0.5 * rhoj_profile],
            gentle,
            note="coupled units; q = con[:,s]/con[:,-1] must be preserved",
        )
    )
    cases.append(
        build(
            "zero_flux",
            stretched_ds,
            smooth,
            np.zeros(nlays + 1),
            note="quiescent column: output must equal input exactly",
        )
    )
    cases.append(
        build(
            "downward_flux",
            stretched_ds,
            smooth,
            -gentle,
            note="subsidence; exercises the opposite upwind branch",
        )
    )

    assert nspcs == 4  # 3 tracers + rho*J
    return cases


def ppm_coeff_cases() -> list[PpmCoeffCase]:
    """Reconstruction cases, on both uniform and stretched vertical grids.

    ``PPM`` needs at least 5 layers for its interior loops (``I = 2, NI-2``) to
    be non-degenerate; all cases here use more.
    """
    nlays = 12
    rng = np.random.default_rng(20260828)
    uniform_ds = np.full(nlays, 1.0 / nlays)
    stretched_ds = np.diff(np.linspace(0.0, 1.0, nlays + 1) ** 1.7)

    z = np.arange(nlays)
    profiles: dict[str, NDArray[np.float64]] = {
        "smooth": 1.0 + 0.4 * np.sin(np.pi * z / nlays),
        "linear": 3.0 - 0.1 * z,
        "step": np.where(z < nlays // 2, 1.0, 4.0),
        "spike": np.where(z == nlays // 2, 8.0, 1.0),
        "constant": np.full(nlays, 2.0),
        "noisy": 1.0 + 0.3 * rng.random(nlays),
        "monotone": np.linspace(1.0, 5.0, nlays),
    }

    cases: list[PpmCoeffCase] = []
    for grid_name, ds in (("uniform", uniform_ds), ("stretched", stretched_ds)):
        for profile_name, cn in profiles.items():
            cases.append(
                PpmCoeffCase(
                    name=f"{profile_name}_{grid_name}",
                    ni=nlays,
                    dt=60.0,
                    ds=ds.astype(F32),
                    cn=cn.astype(F32),
                    note=f"{profile_name} profile on a {grid_name} grid",
                )
            )
    return cases


@dataclass(frozen=True)
class ZfdbcCase:
    """A batch of calls to ZFDBC, the zero-flux-divergence outflow condition."""

    name: str
    c1: NDArray[np.float32]
    c2: NDArray[np.float32]
    v1: NDArray[np.float32]
    v2: NDArray[np.float32]
    note: str = ""

    @property
    def ncase(self) -> int:
        return int(self.c1.size)

    def __post_init__(self) -> None:
        shape = self.c1.shape
        assert self.c2.shape == shape and self.v1.shape == shape, self.name
        assert self.v2.shape == shape, self.name


def zfdbc_cases() -> list[ZfdbcCase]:
    """Cover every branch of zfdbc.f, including both sides of its SMALL cutoff.

    The function has three outcomes (``zfdbc.f:32-40``):

    * ``|v1| < 1e-3``           -> pass the edge value through unchanged
    * ``v1*v2 <= 0`` (diverging) -> likewise unchanged
    * otherwise                  -> extrapolate, clamped at zero

    The clamp matters: without it an outflow boundary can manufacture negative
    concentrations, so cases below deliberately drive the extrapolation
    negative.
    """
    small = 1.0e-3
    rng = np.random.default_rng(20260830)

    # Structured corners: every sign combination, plus the cutoff itself.
    speeds = np.array(
        [-5.0, -1.0, -small * 2, -small, -small / 2, 0.0, small / 2, small, small * 2, 1.0, 5.0]
    )
    v1_grid, v2_grid = np.meshgrid(speeds, speeds, indexing="ij")
    v1_corner = v1_grid.ravel()
    v2_corner = v2_grid.ravel()
    # Gradients that extrapolate up, down, flat, and hard enough to go negative.
    gradients = np.array([0.0, 0.5, -0.5, 3.0, -3.0, 20.0])
    c1_corner = np.tile(np.full(v1_corner.size, 1.0), gradients.size)
    c2_corner = np.concatenate([np.full(v1_corner.size, 1.0) + g for g in gradients])
    v1_corner = np.tile(v1_corner, gradients.size)
    v2_corner = np.tile(v2_corner, gradients.size)

    n_random = 2000
    return [
        ZfdbcCase(
            name="branch_corners",
            c1=c1_corner.astype(F32),
            c2=c2_corner.astype(F32),
            v1=v1_corner.astype(F32),
            v2=v2_corner.astype(F32),
            note="every sign combination x gradient, incl. the SMALL cutoff exactly",
        ),
        ZfdbcCase(
            name="random_sweep",
            c1=(rng.random(n_random) * 10.0).astype(F32),
            c2=(rng.random(n_random) * 10.0).astype(F32),
            v1=((rng.random(n_random) - 0.5) * 4.0).astype(F32),
            v2=((rng.random(n_random) - 0.5) * 4.0).astype(F32),
            note="broad random coverage of the three-way branch",
        ),
    ]


@dataclass(frozen=True)
class HadvCase:
    """One or more calls to HADV, CMAQ's whole horizontal-advection driver.

    Unlike the kernel cases this exercises the driver chain end to end:
    per-layer sub-stepping, the X-Y/Y-X alternation, both boundary conditions
    and both sweeps.
    """

    name: str
    cgrid: NDArray[np.float32]  # (ncols, nrows, nlays, ntrns+1); last slot rho*J
    uwindc: NDArray[np.float32]  # (ncols+1, nrows, nlays)
    vwindc: NDArray[np.float32]  # (ncols, nrows+1, nlays)
    bcon: NDArray[np.float32]  # (nbndy, ntrns+1, nlays)
    astep: NDArray[np.int32]  # (nlays,) HHMMSS
    tstep: tuple[int, int, int] = (10000, 300, 0)
    ncalls: int = 1
    jdate: int = 2018182
    jtime: int = 120000
    xcell: float = 12000.0
    ycell: float = 12000.0
    note: str = ""

    @property
    def shape(self) -> tuple[int, int, int, int]:
        ncols, nrows, nlays, nspc = self.cgrid.shape
        return ncols, nrows, nlays, nspc

    def __post_init__(self) -> None:
        ncols, nrows, nlays, nspc = self.shape
        assert self.uwindc.shape == (ncols + 1, nrows, nlays), self.name
        assert self.vwindc.shape == (ncols, nrows + 1, nlays), self.name
        assert self.bcon.shape == (bndy_size(ncols, nrows), nspc, nlays), self.name
        assert self.astep.shape == (nlays,), self.name


def bndy_size(ncols: int, nrows: int, nthik: int = 1) -> int:
    """Length of CMAQ's boundary ring.

    ``HGRD_DEFN``: ``NBNDY = 2*NTHIK*(NCOLS + NROWS + 2*NTHIK)``.
    """
    return 2 * nthik * (ncols + nrows + 2 * nthik)


def bndy_slices(ncols: int, nrows: int) -> dict[str, slice]:
    """Where each edge lives in the boundary ring, as 0-based Python slices.

    CMAQ indexes the ring through per-edge offsets rather than named blocks:
    ``SFX = 0`` and ``NFX = NCOLS+NROWS+3`` (``y_ppm.F:203``), ``EFX = NCOLS+1``
    and ``WFX = 2*NCOLS+NROWS+4`` (``x_ppm.F:208``), each added to a 1-based
    column or row index. Converting once here keeps that arithmetic in one
    place.
    """
    return {
        "south": slice(0, ncols),
        "east": slice(ncols + 1, ncols + 1 + nrows),
        "north": slice(ncols + nrows + 3, 2 * ncols + nrows + 3),
        "west": slice(2 * ncols + nrows + 4, 2 * ncols + 2 * nrows + 4),
    }


def hadv_cases() -> list[HadvCase]:
    """Driver cases, chosen for what only the driver can exercise."""
    ncols, nrows = 8, 6
    rng = np.random.default_rng(20260831)
    cases: list[HadvCase] = []

    def build(
        name: str,
        *,
        nlays: int,
        ntrns: int,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        astep: list[int],
        ncalls: int = 1,
        coupled_q: list[float] | None = None,
        note: str = "",
    ) -> HadvCase:
        nspc = ntrns + 1
        rhoj = 1.5 + 0.4 * rng.random((ncols, nrows, nlays))
        if coupled_q is None:
            # Independent tracer fields, with a blob to make transport visible.
            layers = [1.0 + rng.random((ncols, nrows, nlays)) for _ in range(ntrns)]
            layers[0][ncols // 2, nrows // 2, :] += 6.0
            cgrid = np.stack([*layers, rhoj], axis=-1)
            bcon = np.zeros((bndy_size(ncols, nrows), nspc, nlays))
            bcon[:, :ntrns, :] = 1.0
            bcon[:, ntrns, :] = 2.0
        else:
            # Coupled units with a uniform mixing ratio, boundary included, so
            # constancy preservation is actually testable.
            cgrid = np.stack([q * rhoj for q in coupled_q] + [rhoj], axis=-1)
            rhoj_bndy = 2.0
            bcon = np.zeros((bndy_size(ncols, nrows), nspc, nlays))
            for idx, q in enumerate(coupled_q):
                bcon[:, idx, :] = q * rhoj_bndy
            bcon[:, ntrns, :] = rhoj_bndy

        return HadvCase(
            name=name,
            cgrid=cgrid.astype(F32),
            uwindc=u.astype(F32),
            vwindc=v.astype(F32),
            bcon=bcon.astype(F32),
            astep=np.array(astep, dtype=np.int32),
            ncalls=ncalls,
            note=note,
        )

    def uniform(nlays: int, speed_u: float, speed_v: float):
        return (
            np.full((ncols + 1, nrows, nlays), speed_u),
            np.full((ncols, nrows + 1, nlays), speed_v),
        )

    # Courant = speed * 180 s / 12000 m; 25 m/s gives 0.375.
    u1, v1 = uniform(1, 25.0, 18.0)
    cases.append(
        build(
            "uniform_wind",
            nlays=1,
            ntrns=2,
            u=u1,
            v=v1,
            astep=[300],
            note="both sweeps, inflow on west and south, outflow on east and north",
        )
    )

    cases.append(
        build(
            "outflow_all_edges",
            nlays=1,
            ntrns=2,
            u=np.concatenate(
                [
                    np.full((ncols // 2 + 1, nrows, 1), -20.0),
                    np.full((ncols - ncols // 2, nrows, 1), 20.0),
                ]
            ),
            v=np.concatenate(
                [
                    np.full((ncols, nrows // 2 + 1, 1), -15.0),
                    np.full((ncols, nrows - nrows // 2, 1), 15.0),
                ],
                axis=1,
            ),
            astep=[300],
            note="wind out of every edge: the zfdbc branch on all four sides",
        )
    )

    cases.append(
        build(
            "inflow_all_edges",
            nlays=1,
            ntrns=2,
            u=np.concatenate(
                [
                    np.full((ncols // 2 + 1, nrows, 1), 20.0),
                    np.full((ncols - ncols // 2, nrows, 1), -20.0),
                ]
            ),
            v=np.concatenate(
                [
                    np.full((ncols, nrows // 2 + 1, 1), 15.0),
                    np.full((ncols, nrows - nrows // 2, 1), -15.0),
                ],
                axis=1,
            ),
            astep=[300],
            note="wind into every edge: the BCON branch on all four sides",
        )
    )

    # Sub-stepping: ASTEP 130 is 90 s against a 180 s sync step, so two passes.
    u2, v2 = uniform(1, 25.0, 18.0)
    cases.append(
        build(
            "substepped",
            nlays=1,
            ntrns=2,
            u=u2,
            v=v2,
            astep=[130],
            note="two advection sub-steps per sync step",
        )
    )

    # Layer 0 sub-steps, layer 1 does not -- the layers must stay independent.
    u3, v3 = uniform(2, 25.0, 18.0)
    cases.append(
        build(
            "mixed_layer_astep",
            nlays=2,
            ntrns=2,
            u=u3,
            v=v3,
            astep=[130, 300],
            note="per-layer ASTEP: layer 0 sub-steps, layer 1 does not",
        )
    )

    # Two calls flip XYFIRST, so the second sweeps Y before X.
    u4, v4 = uniform(1, 25.0, 18.0)
    cases.append(
        build(
            "xy_alternation",
            nlays=1,
            ntrns=2,
            u=u4,
            v=v4,
            astep=[300],
            ncalls=2,
            note="X-Y on the first call, Y-X on the second (hadvppm.F XYFIRST)",
        )
    )

    # Divergent wind, uniform mixing ratio, boundary consistent.
    theta_u = np.linspace(0.0, 2.0 * np.pi, ncols + 1)
    theta_v = np.linspace(0.0, 2.0 * np.pi, nrows + 1)
    cases.append(
        build(
            "constancy_divergent",
            nlays=2,
            ntrns=2,
            u=20.0 * np.sin(theta_u)[:, None, None] * np.ones((1, nrows, 2)),
            v=15.0 * np.cos(theta_v)[None, :, None] * np.ones((ncols, 1, 2)),
            astep=[300, 300],
            coupled_q=[1.0, 3.0],
            note="coupled units, divergent wind; q must survive both sweeps",
        )
    )

    u5, v5 = uniform(1, 32.0, 0.0)
    cases.append(
        build(
            "single_species",
            nlays=1,
            ntrns=1,
            u=u5,
            v=v5,
            astep=[300],
            note="ntrns = 1: the smallest species layout the driver accepts",
        )
    )

    return cases


@dataclass(frozen=True)
class ZadvCase:
    """One call to ZADV, CMAQ's vertical-advection driver."""

    name: str
    faces: NDArray[np.float32]  # (nlays+1,) sigma face coordinates
    rhoj_met: NDArray[np.float32]  # (ncols, nrows, nlays) met density
    cgrid: NDArray[np.float32]  # (ncols, nrows, nlays, ntrns+1); last slot rho*J
    tstep: tuple[int, int, int] = (10000, 300, 0)
    jdate: int = 2018182
    jtime: int = 120000
    note: str = ""

    def __post_init__(self) -> None:
        ncols, nrows, nlays, _ = self.cgrid.shape
        assert self.faces.shape == (nlays + 1,), self.name
        assert self.rhoj_met.shape == (ncols, nrows, nlays), self.name


def zadv_cases() -> list[ZadvCase]:
    """Vertical cases, chosen for what only the driver reaches.

    The Courant number here is not set by a wind field -- it follows from how
    far the transported density has drifted from the meteorology, since that
    mismatch is what the diagnosed flux has to close. A larger mismatch means a
    larger flux, and past a Courant of one the driver has to sub-step.
    """
    ncols, nrows, nlays = 4, 3, 12
    rng = np.random.default_rng(20260904)
    stretched = (np.linspace(1.0, 0.0, nlays + 1) ** 0.625).astype(F32)
    uniform = np.linspace(1.0, 0.0, nlays + 1).astype(F32)
    cases: list[ZadvCase] = []

    def build(
        name: str,
        *,
        faces: NDArray[np.float32],
        mismatch: float,
        ntrns: int = 2,
        coupled_q: list[float] | None = None,
        note: str = "",
    ) -> ZadvCase:
        rhoj = 1.5 + 0.4 * rng.random((ncols, nrows, nlays))
        drift = mismatch * np.sin(np.linspace(0.0, 2.0 * np.pi, nlays))[None, None, :]
        rhoj_met = rhoj * (1.0 + drift)

        if coupled_q is None:
            tracers = [1.0 + rng.random((ncols, nrows, nlays)) for _ in range(ntrns)]
            tracers[0][:, :, nlays // 2] += 5.0
        else:
            tracers = [q * rhoj for q in coupled_q]

        return ZadvCase(
            name=name,
            faces=faces,
            rhoj_met=rhoj_met.astype(F32),
            cgrid=np.stack([*tracers, rhoj], axis=-1).astype(F32),
            note=note,
        )

    cases.append(
        build(
            "gentle_stretched",
            faces=stretched,
            mismatch=0.02,
            note="Courant well under 1: a single pass, no sub-stepping",
        )
    )
    cases.append(
        build(
            "gentle_uniform_layers",
            faces=uniform,
            mismatch=0.02,
            note="same, on evenly spaced layers",
        )
    )
    cases.append(
        build(
            "substepped",
            faces=stretched,
            mismatch=0.3,
            note="Courant above 1: the driver must split the sync step",
        )
    )
    cases.append(
        build(
            "heavily_substepped",
            faces=stretched,
            mismatch=0.6,
            note="Courant above 2: several sub-steps, each shorter than the last",
        )
    )
    cases.append(
        build(
            "constancy",
            faces=stretched,
            mismatch=0.15,
            coupled_q=[0.75, 3.0],
            note="coupled units; q must survive the column solve",
        )
    )
    cases.append(
        build(
            "no_mismatch",
            faces=stretched,
            mismatch=0.0,
            note="transported density already matches met: no flux, nothing moves",
        )
    )
    cases.append(
        build(
            "single_species",
            faces=stretched,
            mismatch=0.1,
            ntrns=1,
            note="the smallest species layout",
        )
    )
    return cases


# --------------------------------------------------------------------------
# Running the harness
# --------------------------------------------------------------------------


def build_harness() -> None:
    """Compile the kernels and harnesses. Raises on failure."""
    subprocess.run(["make", "-C", str(REFERENCE)], check=True, capture_output=True, text=True)


def _run(exe: Path, payload: bytes, workdir: Path) -> bytes:
    """Write `payload`, run `exe` in a fresh process, return its output bytes."""
    in_path = workdir / "in.bin"
    out_path = workdir / "out.bin"
    in_path.write_bytes(payload)
    # check=False on purpose: a non-zero exit usually means the kernel called
    # M3EXIT, and its stderr says why. Raising with that text beats a bare
    # CalledProcessError.
    result = subprocess.run(
        [str(exe), str(in_path), str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{exe.name} exited {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return out_path.read_bytes()


def _f(order: str = "F") -> dict[str, str]:
    return {"order": order}


def run_hppm(case: HppmCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    payload = (
        struct.pack("<4i", case.ni, case.ni, case.nspcs, 0 if case.ori == "C" else 1)
        + struct.pack("<2f", case.dt, case.ds)
        + case.con.astype(F32).tobytes(**_f())
        + case.vel.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_hppm", payload, workdir)

    n_con = (case.ni + 2 * SWP) * case.nspcs
    expected = (n_con + 4 * case.nspcs) * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")

    values = np.frombuffer(raw, dtype=F32)
    con = values[:n_con].reshape((case.ni + 2 * SWP, case.nspcs), order="F")
    rest = values[n_con:].reshape((4, case.nspcs))
    return {
        "con_out": con.copy(),
        "f_lo_in": rest[0].copy(),
        "f_lo_out": rest[1].copy(),
        "f_hi_in": rest[2].copy(),
        "f_hi_out": rest[3].copy(),
    }


def run_vppm(case: VppmCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    payload = (
        struct.pack("<2i", case.ni, case.nspcs)
        + struct.pack("<f", case.dt)
        + case.ds.astype(F32).tobytes(**_f())
        + case.flx.astype(F32).tobytes(**_f())
        + case.vel.astype(F32).tobytes(**_f())
        + case.con.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_vppm", payload, workdir)

    n_con = case.ni * case.nspcs
    expected = (n_con + case.ni + 1) * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")

    values = np.frombuffer(raw, dtype=F32)
    con = values[:n_con].reshape((case.ni, case.nspcs), order="F")
    return {"con_out": con.copy(), "vel_out": values[n_con:].copy()}


def run_ppm_coeffs(case: PpmCoeffCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    payload = (
        struct.pack("<i", case.ni)
        + struct.pack("<f", case.dt)
        + case.ds.astype(F32).tobytes(**_f())
        + case.cn.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_ppm_coeffs", payload, workdir)

    expected = 4 * case.ni * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")

    cr, cl, dc, c6 = np.frombuffer(raw, dtype=F32).reshape((4, case.ni))
    return {"cr": cr.copy(), "cl": cl.copy(), "dc": dc.copy(), "c6": c6.copy()}


def run_zfdbc(case: ZfdbcCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    payload = struct.pack("<i", case.ncase) + b"".join(
        arr.astype(F32).tobytes(**_f()) for arr in (case.c1, case.c2, case.v1, case.v2)
    )
    raw = _run(BUILD / "harness_zfdbc", payload, workdir)

    expected = case.ncase * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    return {"result": np.frombuffer(raw, dtype=F32).copy()}


def run_hadv(case: HadvCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    ncols, nrows, nlays, nspc = case.shape
    payload = (
        struct.pack("<4i", ncols, nrows, nlays, nspc - 1)
        + struct.pack("<i", case.ncalls)
        + struct.pack("<2i", case.jdate, case.jtime)
        + struct.pack("<3i", *case.tstep)
        + case.astep.astype(np.int32).tobytes(**_f())
        + struct.pack("<2f", case.xcell, case.ycell)
        + case.uwindc.astype(F32).tobytes(**_f())
        + case.vwindc.astype(F32).tobytes(**_f())
        + case.bcon.astype(F32).tobytes(**_f())
        + case.cgrid.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_hadv", payload, workdir)

    expected = case.cgrid.size * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    return {"cgrid_out": np.frombuffer(raw, dtype=F32).reshape(case.cgrid.shape, order="F").copy()}


def run_zadv(case: ZadvCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    ncols, nrows, nlays, nspc = case.cgrid.shape
    payload = (
        struct.pack("<4i", ncols, nrows, nlays, nspc - 1)
        + struct.pack("<2i", case.jdate, case.jtime)
        + struct.pack("<3i", *case.tstep)
        + case.faces.astype(F32).tobytes(**_f())
        + case.rhoj_met.astype(F32).tobytes(**_f())
        + case.rhoj_met.astype(F32).tobytes(**_f())  # end-of-step; unused, FBLN = 1
        + case.cgrid.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_zadv", payload, workdir)

    expected = case.cgrid.size * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    return {"cgrid_out": np.frombuffer(raw, dtype=F32).reshape(case.cgrid.shape, order="F").copy()}


# --------------------------------------------------------------------------
# Golden files
# --------------------------------------------------------------------------


def hppm_golden(case: HppmCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    out = run_hppm(case, workdir)
    return {
        "ni": np.int32(case.ni),
        "nspcs": np.int32(case.nspcs),
        "dt": np.float32(case.dt),
        "ds": np.float32(case.ds),
        "ori": np.array(case.ori),
        "swp": np.int32(SWP),
        "con_in": case.con,
        "vel_in": case.vel,
        **out,
    }


def vppm_golden(case: VppmCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    out = run_vppm(case, workdir)
    return {
        "ni": np.int32(case.ni),
        "nspcs": np.int32(case.nspcs),
        "dt": np.float32(case.dt),
        "ds": case.ds,
        "flx_in": case.flx,
        "vel_in": case.vel,
        "con_in": case.con,
        **out,
    }


def ppm_coeff_golden(case: PpmCoeffCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    out = run_ppm_coeffs(case, workdir)
    return {
        "ni": np.int32(case.ni),
        "dt": np.float32(case.dt),
        "ds": case.ds,
        "cn": case.cn,
        **out,
    }


def zfdbc_golden(case: ZfdbcCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "c1": case.c1,
        "c2": case.c2,
        "v1": case.v1,
        "v2": case.v2,
        **run_zfdbc(case, workdir),
    }


def hadv_golden(case: HadvCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "cgrid_in": case.cgrid,
        "uwindc": case.uwindc,
        "vwindc": case.vwindc,
        "bcon": case.bcon,
        "astep": case.astep,
        "tstep": np.array(case.tstep, dtype=np.int32),
        "ncalls": np.int32(case.ncalls),
        "xcell": np.float32(case.xcell),
        "ycell": np.float32(case.ycell),
        **run_hadv(case, workdir),
    }


@dataclass(frozen=True)
class HcdiffCase:
    """One call to HCDIFF3D, the horizontal eddy diffusivity.

    ``uhat_jd``/``vhat_jd`` are the contravariant velocity times Jacobian times
    density, on the dot grid, exactly as ``deform.F:299-301`` reads them.
    ``densa_j_bnd`` is the perimeter ring: ``deform.F`` takes the non-WINDOW
    path, which reads interior and boundary separately and reassembles them,
    and the halo density divides the wind at the domain-edge faces.
    """

    name: str
    uhat_jd: NDArray[np.float32]  # (ncols+1, nrows+1, nlays)
    vhat_jd: NDArray[np.float32]  # (ncols+1, nrows+1, nlays)
    densa_j: NDArray[np.float32]  # (ncols, nrows, nlays)
    densa_j_bnd: NDArray[np.float32]  # (nbndy, nlays), nbndy = 2*(ncols+nrows+2)
    msfd2: NDArray[np.float32]  # (ncols+1, nrows+1)
    dx: float = 12000.0
    jdate: int = 2018182
    jtime: int = 120000
    note: str = ""

    def __post_init__(self) -> None:
        ncols_p1, nrows_p1, nlays = self.uhat_jd.shape
        ncols, nrows = ncols_p1 - 1, nrows_p1 - 1
        assert self.vhat_jd.shape == self.uhat_jd.shape, self.name
        assert self.densa_j.shape == (ncols, nrows, nlays), self.name
        assert self.densa_j_bnd.shape == (2 * (ncols + nrows + 2), nlays), self.name
        assert self.msfd2.shape == (ncols_p1, nrows_p1), self.name


def hcdiff_cases() -> list[HcdiffCase]:
    """Diffusivity cases, chosen to separate the stages that can hide each other.

    The scheme composes a deformation, a saturating blend and a face average,
    and a smooth random field exercises all three at once without distinguishing
    them. Each analytic case below isolates one, so a failure says which stage.
    """
    ncols, nrows, nlays = 8, 7, 3
    nbndy = 2 * (ncols + nrows + 2)
    rng = np.random.default_rng(20260827)
    rows = np.arange(nrows + 1, dtype=np.float64)[None, :, None]
    cols = np.arange(ncols + 1, dtype=np.float64)[:, None, None]
    shape = (ncols + 1, nrows + 1, nlays)
    zero = np.zeros(shape)

    def build(
        name: str,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        *,
        rho: float | NDArray[np.float64] = 2.0,
        msfd2: float | NDArray[np.float64] = 1.0,
        note: str = "",
    ) -> HcdiffCase:
        density = np.broadcast_to(np.asarray(rho, dtype=np.float64), (ncols, nrows, nlays))
        # The wind arrives already multiplied by rho*J, so build it that way
        # rather than dividing later -- that is what is actually on the file.
        face_rho = float(np.mean(density))
        return HcdiffCase(
            name=name,
            uhat_jd=(np.broadcast_to(u, shape) * face_rho).astype(F32),
            vhat_jd=(np.broadcast_to(v, shape) * face_rho).astype(F32),
            densa_j=density.astype(F32),
            densa_j_bnd=np.full((nbndy, nlays), face_rho, dtype=F32),
            msfd2=np.broadcast_to(
                np.asarray(msfd2, dtype=np.float64), (ncols + 1, nrows + 1)
            ).astype(F32),
            note=note,
        )

    return [
        build(
            "uniform_wind",
            np.full(shape, 10.0),
            np.full(shape, -4.0),
            note="solid-body translation: deformation is identically zero, so the "
            "diffusivity sits on its KHMIN floor everywhere. Separates 'reads the "
            "wind' from 'differentiates it'.",
        ),
        build(
            "shear_dudy",
            3.0 * rows,
            zero,
            note="pure shear. deform = |du/dy| = 3/dx2 in the interior, and zero on "
            "rows 1 and NROWS where deform.F:420 zeroes DUDY.",
        ),
        build(
            "stretch_dudx",
            5.0 * cols,
            zero,
            note="pure stretching, the DF1 term, exact for a linear field.",
        ),
        build(
            "linear_both",
            3.0 * rows,
            2.0 * cols,
            note="both cross terms at once. Rotating this field to (-v, u) must "
            "leave the deformation unchanged on the strict interior -- it is the "
            "second invariant of the strain-rate tensor.",
        ),
        build(
            "saturating",
            1.0e4 * rows,
            zero,
            note="deformation large enough that KHA*KHD/(KHA+KHD) is within "
            "rounding of KHA. Catches an upside-down blend, which looks plausible "
            "on a mild field.",
        ),
        build(
            "map_factor",
            3.0 * rows,
            2.0 * cols,
            msfd2=1.0 + 0.25 * rng.random((ncols + 1, nrows + 1)),
            note="non-unit MSFD2. It is ~1 on the benchmark Lambert grid, so a "
            "dropped multiplication would pass every other case here.",
        ),
        build(
            "variable_density",
            3.0 * rows,
            2.0 * cols,
            rho=1.5 + 0.4 * rng.random((ncols, nrows, nlays)),
            note="density varying in space, so the wind recovered at each face "
            "depends on the two-cell average rather than a constant.",
        ),
        build(
            "smooth_random",
            rng.normal(0.0, 6.0, shape),
            rng.normal(0.0, 6.0, shape),
            rho=1.5 + 0.4 * rng.random((ncols, nrows, nlays)),
            note="no structure to exploit; the case that would catch anything the "
            "analytic ones agree on for the wrong reason.",
        ),
    ]


def run_hcdiff(case: HcdiffCase, workdir: Path) -> dict[str, NDArray[np.float32]]:
    ncols_p1, nrows_p1, nlays = case.uhat_jd.shape
    ncols, nrows = ncols_p1 - 1, nrows_p1 - 1
    payload = (
        struct.pack("<3i", ncols, nrows, nlays)
        + struct.pack("<2i", case.jdate, case.jtime)
        + struct.pack("<2d", case.dx, case.dx)
        + case.uhat_jd.astype(F32).tobytes(**_f())
        + case.vhat_jd.astype(F32).tobytes(**_f())
        + case.densa_j.astype(F32).tobytes(**_f())
        + case.densa_j_bnd.astype(F32).tobytes(**_f())
        + case.msfd2.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_hcdiff3d", payload, workdir)

    field = ncols_p1 * nrows_p1 * nlays
    expected = (3 * field + 1) * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    values = np.frombuffer(raw, dtype=F32)
    shape = (ncols_p1, nrows_p1, nlays)
    take = lambda i: values[i * field : (i + 1) * field].reshape(shape, order="F").copy()  # noqa: E731
    return {
        "deform": take(0),
        "k11bar": take(1),
        "k22bar": take(2),
        "dt": np.float32(values[3 * field]),
    }


def hcdiff_golden(case: HcdiffCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "uhat_jd": case.uhat_jd,
        "vhat_jd": case.vhat_jd,
        "densa_j": case.densa_j,
        "densa_j_bnd": case.densa_j_bnd,
        "msfd2": case.msfd2,
        **run_hcdiff(case, workdir),
    }


def zadv_golden(case: ZadvCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "faces": case.faces,
        "rhoj_met": case.rhoj_met,
        "cgrid_in": case.cgrid,
        "tstep": np.array(case.tstep, dtype=np.int32),
        **run_zadv(case, workdir),
    }


@dataclass(frozen=True)
class HdiffCase:
    """One call to HDIFF, CMAQ's horizontal-diffusion driver.

    ``cgrid`` is in coupled units with rho*J in the last slot, matching the
    advection cases -- but note that diffusion does **not** transport that slot,
    so it must come back unchanged. ``dx`` matters more here than anywhere else:
    the sub-step count is ``CFC*dx^2/max(K) `` against the sync step, and on a
    12 km grid the stable step is ~2e5 s, so nothing ever subdivides. Reaching
    the sub-stepping path at all requires a finer grid.
    """

    name: str
    uhat_jd: NDArray[np.float32]
    vhat_jd: NDArray[np.float32]
    densa_j: NDArray[np.float32]
    densa_j_bnd: NDArray[np.float32]
    msfd2: NDArray[np.float32]
    cgrid: NDArray[np.float32]
    dx: float = 12000.0
    sync_seconds: int = 180
    jdate: int = 2018182
    jtime: int = 120000
    note: str = ""


def _hhmmss(seconds: int) -> int:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return hours * 10000 + minutes * 100 + secs


def hdiff_cases() -> list[HdiffCase]:
    """Driver cases, chosen mainly for how many sub-steps they force.

    The two behaviours worth pinning are both sub-step-dependent: the frozen
    halo only drifts after the first pass, and a run that agrees over 147
    sub-steps is real evidence the halo is not being refreshed.
    """
    ncols, nrows, nlays, ntrns = 7, 6, 3, 2
    nbndy = 2 * (ncols + nrows + 2)
    shape = (ncols + 1, nrows + 1, nlays)
    rows = np.arange(nrows + 1, dtype=np.float64)[None, :, None]
    cols = np.arange(ncols + 1, dtype=np.float64)[:, None, None]

    def build(name: str, *, dx: float, sync_seconds: int, note: str) -> HdiffCase:
        rng = np.random.default_rng(20260827)
        rho = 1.5 + 0.4 * rng.random((ncols, nrows, nlays))
        mean_rho = float(rho.mean())
        u = 40.0 * rows + 15.0 * cols + rng.normal(0.0, 3.0, shape)
        v = 25.0 * cols + rng.normal(0.0, 3.0, shape)
        q = 1.0 + rng.random((ncols, nrows, nlays, ntrns))
        q[2, 2, 1, 0] += 8.0  # a spike, so there is a gradient worth diffusing
        return HdiffCase(
            name=name,
            uhat_jd=(u * mean_rho).astype(F32),
            vhat_jd=(v * mean_rho).astype(F32),
            densa_j=rho.astype(F32),
            densa_j_bnd=np.full((nbndy, nlays), mean_rho, dtype=F32),
            msfd2=np.ones((ncols + 1, nrows + 1), dtype=F32),
            cgrid=np.concatenate([q * rho[..., None], rho[..., None]], axis=-1).astype(F32),
            dx=dx,
            sync_seconds=sync_seconds,
            note=note,
        )

    return [
        build(
            "single_step",
            dx=12000.0,
            sync_seconds=180,
            note="the benchmark grid, where the stable step is ~2e5 s and NSTEPS "
            "is 1. The frozen halo is exact here, so this isolates the update "
            "itself from the sub-stepping.",
        ),
        build(
            "two_steps",
            dx=4000.0,
            sync_seconds=3600,
            note="NSTEPS = 2: the first case where the halo drift exists at all.",
        ),
        build(
            "many_steps",
            dx=1000.0,
            sync_seconds=3600,
            note="NSTEPS = 63. Enough passes that a halo refreshed each sub-step "
            "would diverge visibly from one held fixed.",
        ),
        build(
            "deep_substepping",
            dx=500.0,
            sync_seconds=3600,
            note="NSTEPS = 147, the sub-stepping path under real stress.",
        ),
    ]


def run_hdiff(case: HdiffCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    ncols, nrows, nlays, nspc = case.cgrid.shape
    payload = (
        struct.pack("<4i", ncols, nrows, nlays, nspc - 1)
        + struct.pack("<2i", case.jdate, case.jtime)
        + struct.pack("<3i", 10000, _hhmmss(case.sync_seconds), 0)
        + struct.pack("<2d", case.dx, case.dx)
        + case.uhat_jd.astype(F32).tobytes(**_f())
        + case.vhat_jd.astype(F32).tobytes(**_f())
        + case.densa_j.astype(F32).tobytes(**_f())
        + case.densa_j_bnd.astype(F32).tobytes(**_f())
        + case.msfd2.astype(F32).tobytes(**_f())
        + case.cgrid.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_hdiff", payload, workdir)

    expected = case.cgrid.size * 4 + 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    out = np.frombuffer(raw[:-4], dtype=F32).reshape(case.cgrid.shape, order="F").copy()
    return {"cgrid_out": out, "nsteps": np.int32(struct.unpack("<i", raw[-4:])[0])}


def hdiff_golden(case: HdiffCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "uhat_jd": case.uhat_jd,
        "vhat_jd": case.vhat_jd,
        "densa_j": case.densa_j,
        "densa_j_bnd": case.densa_j_bnd,
        "msfd2": case.msfd2,
        "cgrid_in": case.cgrid,
        "dx": np.float64(case.dx),
        "sync_seconds": np.int32(case.sync_seconds),
        **run_hdiff(case, workdir),
    }


#: Value written into matrix entries the solvers must never reference --
#: ``L(1)``/``U(NLAYS)`` for TRI, ``A(1)``/``E(1)`` for MATRIX1. Large enough
#: that a port which touches one disagrees loudly instead of subtly.
POISON = np.float32(-9.99e9)


@dataclass(frozen=True)
class TriCase:
    """One call to TRI, the Thomas solver for ACM2's local stage."""

    name: str
    sub: NDArray[np.float32]  # (nlays,) subdiagonal; entry 0 unused
    diag: NDArray[np.float32]  # (nlays,)
    sup: NDArray[np.float32]  # (nlays,) superdiagonal; last entry unused
    rhs: NDArray[np.float32]  # (nspcs, nlays)
    note: str = ""


def _tri_matrix(case: TriCase) -> NDArray[np.float64]:
    """The dense matrix TRI is solving, from CMAQ's storage (``tri.F:40-46``).

    Row ``k`` holds ``L(k)`` at ``k-1``, ``D(k)`` at ``k`` and ``U(k)`` at
    ``k+1``. Confirmed by residual against the compiled Fortran rather than read
    off the comment block.
    """
    nlays = case.diag.size
    matrix = np.zeros((nlays, nlays), dtype=np.float64)
    for k in range(nlays):
        matrix[k, k] = case.diag[k]
        if k > 0:
            matrix[k, k - 1] = case.sub[k]
        if k < nlays - 1:
            matrix[k, k + 1] = case.sup[k]
    return matrix


def tri_cases() -> list[TriCase]:
    """Solver cases. Diagonally dominant unless a case says otherwise, since
    that is what the ACM2 assembly actually produces."""
    rng = np.random.default_rng(20260828)
    cases: list[TriCase] = []

    def build(name: str, nlays: int, nspcs: int, *, dominance: float, note: str) -> TriCase:
        sub = np.concatenate([[POISON], rng.uniform(-1.0, -0.2, nlays - 1)])
        sup = np.concatenate([rng.uniform(-1.0, -0.2, nlays - 1), [POISON]])
        # Diagonal set relative to the off-diagonal row sum, so `dominance`
        # controls conditioning directly.
        off = np.zeros(nlays)
        off[1:] += np.abs(sub[1:])
        off[:-1] += np.abs(sup[:-1])
        diag = dominance * off + 0.05
        return TriCase(
            name=name,
            sub=sub.astype(F32),
            diag=diag.astype(F32),
            sup=sup.astype(F32),
            rhs=rng.normal(0.0, 1.0, (nspcs, nlays)).astype(F32),
            note=note,
        )

    cases.append(
        build(
            "well_conditioned",
            35,
            4,
            dominance=2.0,
            note="a benchmark-depth column, comfortably diagonally dominant -- "
            "what the Crank-Nicolson assembly gives at a normal sub-step.",
        )
    )
    cases.append(
        build(
            "barely_dominant",
            35,
            4,
            dominance=1.01,
            note="on the edge of diagonal dominance, where the Thomas algorithm "
            "is least accurate. A long sub-step with strong mixing approaches this.",
        )
    )
    cases.append(
        build(
            "shallow",
            4,
            2,
            dominance=2.0,
            note="few layers: catches an off-by-one in the sweep that a deep column would dilute.",
        )
    )

    # An asymmetric matrix, so a transposed sub/super-diagonal cannot pass.
    nlays, nspcs = 12, 3
    sub = np.concatenate([[POISON], np.full(nlays - 1, -0.20)])
    sup = np.concatenate([np.full(nlays - 1, -0.90), [POISON]])
    cases.append(
        TriCase(
            name="asymmetric",
            sub=sub.astype(F32),
            diag=np.full(nlays, 2.5, dtype=F32),
            sup=sup.astype(F32),
            rhs=rng.normal(0.0, 1.0, (nspcs, nlays)).astype(F32),
            note="sub- and super-diagonals deliberately unequal. A symmetric "
            "matrix would let a transposed pair pass, and that is the natural "
            "error when porting a banded solver.",
        )
    )
    return cases


def run_tri(case: TriCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    nspcs, nlays = case.rhs.shape
    payload = (
        struct.pack("<2i", nlays, nspcs)
        + case.sub.astype(F32).tobytes(**_f())
        + case.diag.astype(F32).tobytes(**_f())
        + case.sup.astype(F32).tobytes(**_f())
        + case.rhs.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_tri", payload, workdir)
    expected = case.rhs.size * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    return {"x": np.frombuffer(raw, dtype=F32).reshape(case.rhs.shape, order="F").copy()}


def tri_golden(case: TriCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "sub": case.sub,
        "diag": case.diag,
        "sup": case.sup,
        "rhs": case.rhs,
        **run_tri(case, workdir),
    }


@dataclass(frozen=True)
class Matrix1Case:
    """One call to MATRIX1, the ACM1 solver for the convective stage.

    The matrix is tridiagonal-plus-first-column: every layer in the convective
    boundary layer couples directly to the surface layer, which is what the
    non-local plume means. ``col`` is a *column*, not a subdiagonal.
    """

    name: str
    col: NDArray[np.float32]  # (nlays,) first column, entry 0 unused
    diag: NDArray[np.float32]  # (nlays,)
    sup: NDArray[np.float32]  # (nlays,) sup[L] sits in row L-1; entry 0 unused
    rhs: NDArray[np.float32]  # (nspcs, nlays)
    kl: int  # top of the convective boundary layer, 1-based
    note: str = ""


def _matrix1_matrix(case: Matrix1Case) -> NDArray[np.float64]:
    """The dense matrix MATRIX1 solves, over rows ``1..kl``.

    Row 1:        ``B(1)X(1) + E(2)X(2)``
    Rows 2..kl-1: ``A(L)X(1) + B(L)X(L) + E(L+1)X(L+1)``
    Row kl:       ``A(kl)X(1) + B(kl)X(kl)``

    Derived from the back-substitution at ``matrix1.F:96-106`` and confirmed by
    residual against the compiled Fortran.
    """
    kl = case.kl
    matrix = np.zeros((kl, kl), dtype=np.float64)
    matrix[0, 0] = case.diag[0]
    if kl > 1:
        matrix[0, 1] = case.sup[1]
    for k in range(1, kl):
        matrix[k, 0] += case.col[k]
        matrix[k, k] += case.diag[k]
        if k < kl - 1:
            matrix[k, k + 1] = case.sup[k + 1]
    return matrix


def matrix1_cases() -> list[Matrix1Case]:
    """Convective-stage cases, chosen around the ALPHA product.

    ``matrix1.F`` accumulates ``ALPHA = prod(-E/B)`` down the CBL and divides by
    a sum weighted with it. How small that gets is the question these cases
    exist to answer, so the ratio ``|E/B|`` and the CBL depth are what vary.
    """
    rng = np.random.default_rng(20260828)
    cases: list[Matrix1Case] = []

    def build(
        name: str, nlays: int, nspcs: int, kl: int, *, ratio: float, note: str
    ) -> Matrix1Case:
        diag = rng.uniform(2.0, 4.0, nlays)
        sup = np.concatenate([[POISON], -ratio * diag[1:]])
        col = np.concatenate([[POISON], rng.uniform(-0.6, -0.1, nlays - 1)])
        return Matrix1Case(
            name=name,
            col=col.astype(F32),
            diag=diag.astype(F32),
            sup=sup.astype(F32),
            rhs=rng.normal(0.0, 1.0, (nspcs, nlays)).astype(F32),
            kl=kl,
            note=note,
        )

    cases.append(
        build(
            "shallow_cbl",
            35,
            4,
            kl=6,
            ratio=0.30,
            note="a shallow convective boundary layer, ALPHA barely decaying.",
        )
    )
    cases.append(
        build(
            "deep_cbl",
            35,
            4,
            kl=28,
            ratio=0.30,
            note="a deep CBL: 27 factors of ~0.3, so ALPHA reaches ~1e-14.",
        )
    )
    cases.append(
        build(
            "alpha_underflow",
            35,
            4,
            kl=30,
            ratio=0.05,
            note="ALPHA driven as small as a realistic column allows -- 29 "
            "factors of ~0.05 is ~1e-38, at the edge of float32. GAMA is a "
            "denominator, so this is where the solver would break if it breaks.",
        )
    )
    cases.append(
        build(
            "whole_column",
            20,
            3,
            kl=20,
            ratio=0.40,
            note="CBL filling the entire column, the largest KL possible.",
        )
    )
    cases.append(
        build(
            "minimal_cbl",
            12,
            2,
            kl=2,
            ratio=0.35,
            note="KL = 2, the smallest convective stage that runs at all: the "
            "ALPHA loop executes exactly once.",
        )
    )
    return cases


def run_matrix1(case: Matrix1Case, workdir: Path) -> dict[str, NDArray[np.generic]]:
    nspcs, nlays = case.rhs.shape
    payload = (
        struct.pack("<3i", nlays, nspcs, case.kl)
        + case.col.astype(F32).tobytes(**_f())
        + case.diag.astype(F32).tobytes(**_f())
        + case.sup.astype(F32).tobytes(**_f())
        + case.rhs.astype(F32).tobytes(**_f())
    )
    raw = _run(BUILD / "harness_matrix1", payload, workdir)
    expected = case.rhs.size * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    return {"x": np.frombuffer(raw, dtype=F32).reshape(case.rhs.shape, order="F").copy()}


def matrix1_golden(case: Matrix1Case, workdir: Path) -> dict[str, NDArray[np.generic]]:
    return {
        "col": case.col,
        "diag": case.diag,
        "sup": case.sup,
        "rhs": case.rhs,
        "kl": np.int32(case.kl),
        **run_matrix1(case, workdir),
    }


#: How far a regenerated golden may differ from the committed one, in float32
#: ULPs, before ``--check`` calls it drift.
#:
#: Exact equality is the obvious rule and it is wrong. These arrays are float32
#: results of a Fortran kernel, and a different compiler, architecture or
#: optimisation level reassociates that arithmetic differently -- x86-64 and
#: arm64 disagree in the last bit or two on every case that is not trivially
#: exact. Requiring bit-identity would mean the goldens could only ever be
#: verified on the machine that produced them.
#:
#: What the check is actually for is catching an edit to ``reference/fortran/``
#: or a toolchain change that alters the numerics *materially*. Both move
#: answers by far more than a few ULPs. 20 leaves room for reassociation while
#: staying orders of magnitude below anything meaningful; the observed maximum
#: is printed on every run so creep is visible rather than silently absorbed.
#:
#: The binding case is ``coeffs_monotone_stretched`` at 24 ULPs between
#: macOS/arm64 and ubuntu/x86-64, and it is binding for a reason already known
#: from A0.6: on a linear profile ``c6 = 6*(cn - (cl+cr)/2)`` is a difference of
#: nearly-equal numbers, so its true value is ~0 while its rounding error is set
#: by ``|cn|``. The same cancellation is why that family's regression test
#: budgets 16 ULPs on one machine. Cross-platform reassociation roughly doubles
#: it, so 64 clears the measured worst case with room to spare while remaining
#: four orders of magnitude below a real change: editing the vendored Fortran or
#: altering the scheme moves answers by percent, which is ~1e5 ULPs.
DRIFT_TOLERANCE_ULP = 64.0

EPS32 = float(np.finfo(np.float32).eps)


def relative_drift(committed: NDArray[np.generic], fresh: NDArray[np.generic]) -> float:
    """Largest difference between two golden arrays, in float32 ULPs.

    Scaled by the array's own magnitude, so a field of concentrations and a
    field of fluxes are judged on the same footing.
    """
    if committed.shape != fresh.shape or committed.dtype != fresh.dtype:
        return float("inf")
    if not np.issubdtype(committed.dtype, np.floating):
        return 0.0 if np.array_equal(committed, fresh) else float("inf")

    left = committed.astype(np.float64)
    right = fresh.astype(np.float64)
    scale = max(float(np.abs(left).max()), 1.0)
    return float(np.abs(left - right).max()) / scale / EPS32


@dataclass
class Result:
    written: list[str] = field(default_factory=list)
    drifted: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    worst_ulp: float = 0.0
    worst_case: str = ""


def _check_one(
    name: str,
    path: Path,
    arrays: dict[str, NDArray[np.generic]],
    result: Result,
) -> None:
    """Compare one freshly generated golden against the committed copy."""
    if not path.exists():
        result.missing.append(name)
        return

    with np.load(path, allow_pickle=False) as committed:
        if set(committed.files) != set(arrays):
            result.drifted.append(f"{name} (different arrays)")
            return
        worst = max(relative_drift(committed[key], arrays[key]) for key in arrays)

    if worst > result.worst_ulp:
        result.worst_ulp, result.worst_case = worst, name
    if worst > DRIFT_TOLERANCE_ULP:
        result.drifted.append(f"{name} ({worst:.1f} float32 ULPs)")


@dataclass(frozen=True)
class EddyxCase:
    """One call to EDDYX, the vertical eddy diffusivity.

    Fields are named as ``Met_Data`` names them. ``moli`` is the *inverse*
    Monin-Obukhov length, so its sign is what selects the stability regime:
    negative unstable, zero neutral, positive stable.
    """

    name: str
    pbl: NDArray[np.float32]  # (ncols, nrows)
    ustar: NDArray[np.float32]
    moli: NDArray[np.float32]
    zf: NDArray[np.float32]  # (ncols, nrows, nlays)
    zh: NDArray[np.float32]
    kzmin: NDArray[np.float32]
    thetav: NDArray[np.float32]
    ta: NDArray[np.float32]
    qv: NDArray[np.float32]
    qc: NDArray[np.float32]
    uwind: NDArray[np.float32]  # (ncols+1, nrows+1, nlays)
    vwind: NDArray[np.float32]
    cstaguv: bool = True
    note: str = ""


def eddyx_cases() -> list[EddyxCase]:
    """Diffusivity cases, one per regime the parameterization switches between.

    A single random field exercises all three branches at once and distinguishes
    none of them, so each case here isolates one: surface-layer similarity below
    the PBL, Richardson-number mixing above it, and the moist correction.
    """
    ncols, nrows, nlays = 4, 3, 12
    rng = np.random.default_rng(20260828)
    face = np.linspace(80.0, 2400.0, nlays)
    middle = face - 40.0

    def build(
        name: str,
        *,
        moli: float,
        pbl: float = 1000.0,
        ustar: float = 0.3,
        kzmin: float = 0.5,
        shear: float = 0.0,
        lapse: float = 0.0,
        qc: float = 0.0,
        cstaguv: bool = True,
        jitter: bool = False,
        note: str = "",
    ) -> EddyxCase:
        shape2 = (ncols, nrows)
        shape3 = (ncols, nrows, nlays)
        dot = (ncols + 1, nrows + 1, nlays)

        def noise(shape: tuple[int, ...]) -> NDArray[np.float64]:
            return rng.normal(0.0, 0.02, shape) if jitter else np.zeros(shape)

        thetav = 300.0 + lapse * np.arange(nlays)
        # Winds vary across the domain as well as with height. A spatially
        # uniform profile makes the two wind-shear stencils algebraically
        # identical -- 0.25*(2du)^2 == (1/16)*(4du)^2 -- so the b_staggered case
        # would be bit-identical to the C-staggered one and test nothing.
        across = (
            np.linspace(0.0, 1.0, ncols + 1)[:, None, None]
            + 0.5 * np.linspace(0.0, 1.0, nrows + 1)[None, :, None]
        )
        wind = shear * np.arange(nlays)[None, None, :] * (1.0 + across)
        return EddyxCase(
            name=name,
            pbl=np.full(shape2, pbl, dtype=F32),
            ustar=np.full(shape2, ustar, dtype=F32),
            moli=np.full(shape2, moli, dtype=F32),
            zf=(np.broadcast_to(face, shape3) * (1.0 + noise(shape3))).astype(F32),
            zh=(np.broadcast_to(middle, shape3) * (1.0 + noise(shape3))).astype(F32),
            kzmin=np.full(shape3, kzmin, dtype=F32),
            thetav=(np.broadcast_to(thetav, shape3) + noise(shape3)).astype(F32),
            ta=np.full(shape3, 290.0, dtype=F32),
            qv=np.full(shape3, 0.005, dtype=F32),
            qc=np.full(shape3, qc, dtype=F32),
            uwind=(np.broadcast_to(wind, dot) + noise(dot)).astype(F32),
            vwind=noise(dot).astype(F32),
            cstaguv=cstaguv,
            note=note,
        )

    return [
        build(
            "neutral",
            moli=0.0,
            note="1/L = 0, so PHIH = 1 and the surface term is exactly "
            "KARMAN*USTAR*z*(1-z/h)^2 -- checkable in closed form.",
        ),
        build(
            "unstable",
            moli=-0.01,
            note="convective: PHIH = 1/sqrt(1 - GAMAH*z/L), capped at 0.1*PBL.",
        ),
        build(
            "stable",
            moli=0.01,
            note="PHIH = 1 + BETAH*z/L below z/L = 1, BETAH + z/L above -- the "
            "branch at z/L = 1 that a mild case never reaches.",
        ),
        build(
            "very_stable",
            moli=0.05,
            pbl=1500.0,
            note="drives z/L past 1 in the PBL, taking the second stable branch.",
        ),
        build(
            "sheared",
            moli=0.0,
            shear=1.5,
            lapse=0.6,
            note="wind shear and a stable lapse rate together, so the bulk "
            "Richardson number is finite and the free-atmosphere term is live "
            "rather than sitting on WS2's 1e-9 floor. Below the PBL the "
            "similarity term still dominates, so most of the difference from "
            "the neutral case shows up aloft.",
        ),
        build(
            "unstable_richardson",
            moli=0.0,
            shear=1.5,
            lapse=-0.4,
            note="negative RIB, which takes the other branch of the mixing-length "
            "formula -- sqrt(WS2*(1 - 25*RIB)) instead of the FH polynomial.",
        ),
        build(
            "cloudy",
            moli=0.0,
            shear=1.5,
            lapse=0.6,
            qc=1.0e-3,
            note="cloud water above the 0.01e-3 threshold, enabling the moist "
            "correction from HIRPBL. Omitting it entirely would pass every case "
            "above.",
        ),
        build(
            "kzmin_floor",
            moli=0.02,
            kzmin=4.0,
            note="a large KZMIN, so the floor binds instead of the similarity "
            "term. Catches a max() written as a min().",
        ),
        build(
            "b_staggered",
            moli=0.0,
            shear=1.5,
            lapse=0.6,
            cstaguv=False,
            note="the pre-MCIPv3.5 wind stencil: a 1/16 four-corner average "
            "rather than the 1/4 two-point one.",
        ),
        build(
            "realistic",
            moli=-0.005,
            shear=1.2,
            lapse=0.3,
            qc=0.5e-3,
            jitter=True,
            note="everything varying at once, with no structure to exploit -- "
            "the case that catches anything the analytic ones agree on for the "
            "wrong reason.",
        ),
    ]


def run_eddyx(case: EddyxCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    ncols, nrows, nlays = case.zf.shape
    payload = (
        struct.pack("<3i", ncols, nrows, nlays)
        + struct.pack("<i", 1 if case.cstaguv else 0)
        + b"".join(
            getattr(case, field).astype(F32).tobytes(**_f())
            for field in (
                "pbl",
                "ustar",
                "moli",
                "zf",
                "zh",
                "kzmin",
                "thetav",
                "ta",
                "qv",
                "qc",
                "uwind",
                "vwind",
            )
        )
    )
    raw = _run(BUILD / "harness_eddyx", payload, workdir)
    expected = ncols * nrows * nlays * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    return {"eddyv": np.frombuffer(raw, dtype=F32).reshape((ncols, nrows, nlays), order="F").copy()}


def eddyx_golden(case: EddyxCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    fields = {
        name: getattr(case, name)
        for name in (
            "pbl",
            "ustar",
            "moli",
            "zf",
            "zh",
            "kzmin",
            "thetav",
            "ta",
            "qv",
            "qc",
            "uwind",
            "vwind",
        )
    }
    return {
        **fields,
        "cstaguv": np.bool_(case.cstaguv),
        **run_eddyx(case, workdir),
    }


@dataclass(frozen=True)
class VdiffCase:
    """One call to VDIFFACMX, the ACM2 driver.

    Deposition velocities and emission fluxes are inputs, which is this port's
    scope boundary. ``depv`` must be strictly positive: the surface exchange
    computes ``POL = PLDV/DEPV`` (``vdiffacmx.F:625``), so a zero deposition
    velocity gives 0/0 and the whole column becomes NaN. A "no deposition" case
    therefore uses a negligible value, not zero.
    """

    name: str
    dtsec: float
    convct: NDArray[np.int32]  # (ncols, nrows), 1 = convective
    lpbl: NDArray[np.int32]  # (ncols, nrows), layer index of the PBL top
    pbl: NDArray[np.float32]
    hol: NDArray[np.float32]  # PBL height / Monin-Obukhov length; < 0 convective
    dens1: NDArray[np.float32]
    rdepvht: NDArray[np.float32]
    zf: NDArray[np.float32]  # (ncols, nrows, nlays)
    zh: NDArray[np.float32]
    seddy: NDArray[np.float32]  # (nlays, ncols, nrows) -- layer-first, as CMAQ passes it
    depv: NDArray[np.float32]  # (nspc, ncols, nrows)
    pldv: NDArray[np.float32]
    vdemis: NDArray[np.float32]  # (nspc, nlays, ncols, nrows)
    cngrd: NDArray[np.float32]  # (nspc, nlays, ncols, nrows)
    note: str = ""


def vdiff_cases() -> list[VdiffCase]:
    """Driver cases: one per code path the two-stage split can take."""
    ncols, nrows, nlays, nspc = 3, 2, 14, 3
    rng = np.random.default_rng(20260828)
    face = np.linspace(40.0, 2600.0, nlays)

    def build(
        name: str,
        *,
        convective: bool,
        lpbl: int = 5,
        seddy: float = 10.0,
        depv: float = 1.0e-9,
        pldv: float = 0.0,
        emis: float = 0.0,
        dtsec: float = 300.0,
        surface_pulse: bool = True,
        note: str = "",
    ) -> VdiffCase:
        shape2 = (ncols, nrows)
        shape3 = (ncols, nrows, nlays)

        if surface_pulse:
            conc = np.zeros((nspc, nlays, ncols, nrows))
            conc[0, 0] = 100.0
            conc[1] = 1.0 + rng.random((nlays, ncols, nrows))
            conc[2, nlays // 2] = 40.0
        else:
            conc = 1.0 + rng.random((nspc, nlays, ncols, nrows))

        return VdiffCase(
            name=name,
            dtsec=dtsec,
            convct=np.full(shape2, 1 if convective else 0, dtype=np.int32),
            lpbl=np.full(shape2, lpbl, dtype=np.int32),
            pbl=np.full(shape2, float(face[lpbl - 1]), dtype=F32),
            hol=np.full(shape2, -2.0 if convective else 5.0, dtype=F32),
            dens1=np.full(shape2, 1.2, dtype=F32),
            rdepvht=np.full(shape2, 0.02, dtype=F32),
            zf=np.broadcast_to(face, shape3).astype(F32),
            zh=(np.broadcast_to(face, shape3) - 20.0).astype(F32),
            seddy=np.full((nlays, ncols, nrows), seddy, dtype=F32),
            depv=np.full((nspc, ncols, nrows), depv, dtype=F32),
            pldv=np.full((nspc, ncols, nrows), pldv, dtype=F32),
            vdemis=np.concatenate(
                [
                    np.full((nspc, 1, ncols, nrows), emis),
                    np.zeros((nspc, nlays - 1, ncols, nrows)),
                ],
                axis=1,
            ).astype(F32),
            cngrd=conc.astype(F32),
            note=note,
        )

    return [
        build(
            "stable",
            convective=False,
            note="CONVCT false, so the convective stage is skipped entirely and "
            "this is Crank-Nicolson diffusion with a surface flux. The sharper "
            "first target: if this does not match, debugging the ACM2 terms is "
            "premature.",
        ),
        build(
            "convective",
            convective=True,
            lpbl=6,
            note="the non-local stage active. SEDDY is scaled by (1 - FNL) "
            "inside the CBL and the removed fraction is carried by the plume "
            "instead -- the asymmetry ACM2 is named for.",
        ),
        build(
            "deep_cbl",
            convective=True,
            lpbl=11,
            note="a CBL filling most of the column, so the first-column matrix "
            "is large and the ALPHA product runs long.",
        ),
        build(
            "shallow_cbl",
            convective=True,
            lpbl=2,
            note="the smallest convective stage that runs: LCBL = 2.",
        ),
        build(
            "with_deposition",
            convective=False,
            depv=0.01,
            note="a real deposition velocity, so the surface layer relaxes "
            "toward PLDV/DEPV and the dry-deposition accumulator is exercised.",
        ),
        build(
            "with_emissions",
            convective=False,
            depv=0.005,
            pldv=0.02,
            emis=0.5,
            note="emissions in both forms: PLDV at the surface, which sets the "
            "relaxation target, and VDEMIS as a layer source in the local "
            "stage's right-hand side.",
        ),
        build(
            "substepped",
            convective=True,
            lpbl=8,
            seddy=150.0,
            dtsec=1800.0,
            note="strong mixing over a long step, so NLP is well above 1 and the "
            "sub-step loop runs many times.",
        ),
        build(
            "smooth_random",
            convective=True,
            lpbl=7,
            depv=0.008,
            pldv=0.01,
            emis=0.2,
            surface_pulse=False,
            note="everything on at once with no structure to exploit.",
        ),
    ]


def run_vdiff(case: VdiffCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    nspc, nlays, ncols, nrows = case.cngrd.shape
    payload = (
        struct.pack("<4i", ncols, nrows, nlays, nspc)
        + struct.pack("<f", case.dtsec)
        + case.convct.astype(np.int32).tobytes(**_f())
        + case.lpbl.astype(np.int32).tobytes(**_f())
        + b"".join(
            getattr(case, name).astype(F32).tobytes(**_f())
            for name in (
                "pbl",
                "hol",
                "dens1",
                "rdepvht",
                "zf",
                "zh",
                "seddy",
                "depv",
                "pldv",
                "vdemis",
                "cngrd",
            )
        )
    )
    raw = _run(BUILD / "harness_vdiff", payload, workdir)

    n_conc = case.cngrd.size
    n_dep = nspc * ncols * nrows
    n_sed = nlays * ncols * nrows
    expected = (n_conc + n_dep + n_sed) * 4
    if len(raw) != expected:
        raise RuntimeError(f"{case.name}: got {len(raw)} output bytes, expected {expected}")
    values = np.frombuffer(raw, dtype=F32)
    return {
        "cngrd_out": values[:n_conc].reshape(case.cngrd.shape, order="F").copy(),
        "ddep": values[n_conc : n_conc + n_dep].reshape((nspc, ncols, nrows), order="F").copy(),
        "seddy_out": values[n_conc + n_dep :].reshape((nlays, ncols, nrows), order="F").copy(),
    }


def vdiff_golden(case: VdiffCase, workdir: Path) -> dict[str, NDArray[np.generic]]:
    fields = {
        name: getattr(case, name)
        for name in (
            "convct",
            "lpbl",
            "pbl",
            "hol",
            "dens1",
            "rdepvht",
            "zf",
            "zh",
            "seddy",
            "depv",
            "pldv",
            "vdemis",
            "cngrd",
        )
    }
    return {**fields, "dtsec": np.float32(case.dtsec), **run_vdiff(case, workdir)}


#: Every golden family: name prefix, the case list, and how to run one.
#:
#: A table rather than a run of loops because it is the thing that grows with
#: every operator, and a straight-line version had already outgrown ruff's
#: branch limit by the time vertical diffusion was added.
FAMILIES: list[tuple[str, Callable[[], list[Any]], Callable[[Any, Path], dict[str, Any]]]] = [
    ("hppm", hppm_cases, hppm_golden),
    ("vppm", vppm_cases, vppm_golden),
    ("coeffs", ppm_coeff_cases, ppm_coeff_golden),
    ("zfdbc", zfdbc_cases, zfdbc_golden),
    ("hadv", hadv_cases, hadv_golden),
    ("zadv", zadv_cases, zadv_golden),
    ("hcdiff", hcdiff_cases, hcdiff_golden),
    ("hdiff", hdiff_cases, hdiff_golden),
    ("tri", tri_cases, tri_golden),
    ("matrix1", matrix1_cases, matrix1_golden),
    ("eddyx", eddyx_cases, eddyx_golden),
    ("vdiff", vdiff_cases, vdiff_golden),
]


def generate(check: bool) -> Result:
    build_harness()
    GOLDENS.mkdir(parents=True, exist_ok=True)
    result = Result()

    work: list[tuple[str, dict[str, NDArray[np.generic]]]] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for prefix, cases, golden in FAMILIES:
            for case in cases():
                work.append((f"{prefix}_{case.name}", golden(case, workdir)))

    for name, arrays in work:
        path = GOLDENS / f"{name}.npz"
        if check:
            _check_one(name, path, arrays, result)
        else:
            np.savez_compressed(path, **arrays)
            result.written.append(name)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against committed goldens instead of overwriting them",
    )
    args = parser.parse_args(argv)

    result = generate(check=args.check)

    if not args.check:
        print(f"wrote {len(result.written)} goldens to {GOLDENS.relative_to(REPO)}")
        return 0

    if result.missing or result.drifted:
        for name in result.missing:
            print(f"MISSING: {name}")
        for name in result.drifted:
            print(f"DRIFTED: {name}")
        print(
            f"\nCommitted goldens differ from a fresh Fortran run by more than "
            f"{DRIFT_TOLERANCE_ULP:.0f} float32 ULPs. Either reference/fortran/ was edited "
            "(it must not be -- see reference/PROVENANCE.md) or the toolchain changed "
            "in a way that moves the numerics."
        )
        return 1

    print(
        f"goldens up to date (worst drift {result.worst_ulp:.2f} float32 ULPs"
        + (f", in {result.worst_case}" if result.worst_case else "")
        + f"; tolerance {DRIFT_TOLERANCE_ULP:.0f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
