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
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class Result:
    written: list[str] = field(default_factory=list)
    drifted: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def generate(check: bool) -> Result:
    build_harness()
    GOLDENS.mkdir(parents=True, exist_ok=True)
    result = Result()

    work: list[tuple[str, dict[str, NDArray[np.generic]]]] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for case in hppm_cases():
            work.append((f"hppm_{case.name}", hppm_golden(case, workdir)))
        for vcase in vppm_cases():
            work.append((f"vppm_{vcase.name}", vppm_golden(vcase, workdir)))

    for name, arrays in work:
        path = GOLDENS / f"{name}.npz"
        if check:
            if not path.exists():
                result.missing.append(name)
                continue
            with np.load(path, allow_pickle=False) as committed:
                same = set(committed.files) == set(arrays) and all(
                    np.array_equal(committed[k], arrays[k]) for k in arrays
                )
            if not same:
                result.drifted.append(name)
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
            "\nCommitted goldens do not match a fresh Fortran run. Either "
            "reference/fortran/ was edited (it must not be -- see "
            "reference/PROVENANCE.md) or the toolchain changed."
        )
        return 1

    print("goldens up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
