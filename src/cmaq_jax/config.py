"""Configuration objects holding every constant used by the advection kernels.

No kernel in this package reads a global or hard-codes a numeric bound; they all
take one of the frozen dataclasses defined here. Each field records the Fortran
file and line it came from, so any value can be traced upstream to
``reference/fortran/``.

Constants needed only by ``advstep`` (the CFL/sync-step analysis, chunk A3.1)
are deliberately absent — they land with that module rather than being guessed
now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_HDIFF",
    "DEFAULT_PPM",
    "GridConfig",
    "HDiffConstants",
    "PPMConstants",
    "sigma_layer_thickness",
]

DType = Literal["float32", "float64"]


@dataclass(frozen=True)
class PPMConstants:
    """Numeric constants of the piecewise parabolic method and its drivers.

    Defaults reproduce CMAQ v5.5 exactly. The iteration caps are the one place
    we deliberately differ in *kind*: Fortran loops until convergence and calls
    ``M3EXIT`` on failure, while we run a fixed count and report the residual
    (see ``docs/plans/subplans/A2-zadv.md``).
    """

    # --- PPM parabola and flux integration ---------------------------------
    two_thirds: float = 2.0 / 3.0
    """Colella & Woodward eq. (1.12) flux integral coefficient. ``hppm.F:169``."""

    sixth: float = 1.0 / 6.0
    """Colella & Woodward eq. (1.6) edge-value coefficient. ``hppm.F:170``."""

    halo_width: int = 3
    """Ghost cells per side (``SWP``). ``hppm.F:147``.

    Retained as an explicit array region even though it is filled locally on a
    single device — it is the swap point for a collective permute under
    ``shard_map``.
    """

    # --- Boundary conditions ------------------------------------------------
    zfdbc_small_wind: float = 1.0e-3
    """Wind speed (m/s) below which ZFDBC leaves the edge value alone.
    ``zfdbc.f:29`` (``SMALL``)."""

    # --- Vertical velocity adjustment (VPPM) --------------------------------
    velocity_flux_tolerance: float = 1.0e-3
    """Relative agreement required between the PPM flux of the rho*J column and
    the donor-cell flux. ``vppm.F:145`` (``EPSF``)."""

    velocity_adjust_iterations: int = 8
    """Fixed sqrt-Newton iterations for the face-velocity adjustment.

    Fortran loops to convergence with a cap of 50 (``vppm.F:166``, ``MAXCNT``).
    The update ``vel <- vel * sqrt(F_target / F_ppm)`` converges quadratically,
    so 8 is ample at a 1e-3 tolerance; the post-loop residual is reported rather
    than raising.
    """

    # --- Vertical CFL sub-stepping (ZADV) -----------------------------------
    cfl_safety: float = 0.9
    """Fraction of the CFL-limited step actually taken.
    ``zadvppmwrf.F:393`` (``DTNEW = 0.9 * DELT / CC``)."""

    min_substep_seconds: float = 1.0
    """Floor on a vertical sub-step. ``zadvppmwrf.F:426`` (``MAX(DTNEW, 1.0)``)."""

    max_substeps: int = 30
    """Cap on vertical CFL sub-steps per column. ``zadvppmwrf.F:126`` (``MAXITER``).

    Upstream this is an *error* limit — CMAQ calls ``M3EXIT`` on reaching it —
    but here it is also the trip count of a fixed-length loop, so every column
    pays for all of it whether or not it needs the steps. Measured, each
    iteration costs about the same as one column solve, which makes this the
    single largest lever on the cost of a step.

    Lowering it is safe: :class:`~cmaq_jax.vadv.ZadvDiagnostics` reports an
    infinite residual for any column that ran out of sub-steps, so a caller can
    set it to what a given CFL regime actually needs and check rather than
    assume. 30 is kept as the default because it is what CMAQ permits.
    """

    def __post_init__(self) -> None:
        if self.halo_width < 3:
            # The PPM stencil reads two cells beyond the parabola it builds;
            # hppm.F sizes its lattice arrays on SWP = 3 throughout.
            raise ValueError(f"halo_width must be at least 3, got {self.halo_width}")
        if self.velocity_adjust_iterations < 1:
            raise ValueError("velocity_adjust_iterations must be >= 1")
        if self.max_substeps < 1:
            raise ValueError("max_substeps must be >= 1")
        if not 0.0 < self.cfl_safety <= 1.0:
            raise ValueError(f"cfl_safety must be in (0, 1], got {self.cfl_safety}")


DEFAULT_PPM = PPMConstants()
"""CMAQ v5.5 defaults. Kernels take a ``PPMConstants``; this is the usual one."""


@dataclass(frozen=True)
class HDiffConstants:
    """Constants of the deformation-dependent horizontal eddy diffusivity.

    All from ``hcdiff3d.F:104-115``. Note that ``kh`` and ``khmin`` are tuning
    values chosen for a particular resolution, not physical constants -- the
    file carries two commented-out alternatives for ``kh`` (``3.3e4`` from Brost
    et al. 1988, and ``50.0`` for a 12 km run) and the active one is labelled
    for a 4 km SARMAP simulation. ``dxb`` is the grid spacing they were tuned
    at, and :meth:`base_diffusivity` rescales away from it.
    """

    kh: float = 2000.0
    """Base horizontal eddy diffusivity, m^2/s, at spacing :attr:`dxb`.
    ``hcdiff3d.F:107``."""

    khmin: float = 200.0
    """Floor on the deformation-induced diffusivity, m^2/s. ``hcdiff3d.F:109``."""

    dxb: float = 4000.0
    """Grid spacing :attr:`kh` was tuned at, m. ``hcdiff3d.F:110``."""

    alp: float = 0.28
    """Deformation coefficient; enters squared. ``hcdiff3d.F:111``."""

    cfc: float = 0.300
    """Stability factor for the diffusion sub-step. ``hcdiff3d.F:115``.

    Commented in the source as "99%(1/sqrt(2))", which is 0.700 -- the value on
    the line above, disabled. The active 0.300 is well inside that, so the
    sub-step is conservative rather than marginal.
    """

    max_substeps: int = 64
    """Cap on diffusion sub-steps per sync step.

    CMAQ computes ``NSTEPS = int(DTSEC/DT) + 1`` and loops (``hdiff.F:337``),
    with no cap. A traced loop needs a fixed trip count, so this bounds it; the
    step reports how many were actually needed, and exceeding the cap is
    reported rather than silently truncated -- the same arrangement as
    ``PPMConstants.max_substeps``.
    """

    def __post_init__(self) -> None:
        if self.kh <= 0.0 or self.khmin <= 0.0:
            raise ValueError(f"kh and khmin must be positive, got {self.kh}, {self.khmin}")
        if self.dxb <= 0.0:
            raise ValueError(f"dxb must be positive, got {self.dxb}")
        if self.cfc <= 0.0:
            raise ValueError(f"cfc must be positive, got {self.cfc}")
        if self.max_substeps < 1:
            raise ValueError(f"max_substeps must be >= 1, got {self.max_substeps}")

    def base_diffusivity(self, dx1: float, dx2: float) -> float:
        """``KHA`` -- :attr:`kh` rescaled to this grid. ``hcdiff3d.F:188``.

        ``(dxb^2)/(dx1*dx2) * kh``: coarser cells get *less* diffusivity, because
        a coarse grid already represents less of the sub-grid mixing that this
        term stands in for.
        """
        return (self.dxb * self.dxb) / (dx1 * dx2) * self.kh

    def deformation_coefficient(self, dx1: float, dx2: float) -> float:
        """``ACOEF = alp^2 * dx1 * dx2``. ``hcdiff3d.F:190``."""
        return self.alp * self.alp * (dx1 * dx2)


DEFAULT_HDIFF = HDiffConstants()
"""CMAQ v5.5 defaults for horizontal diffusion."""


def sigma_layer_thickness(x3face: NDArray[np.float64]) -> NDArray[np.float64]:
    """Layer thicknesses from sigma-level face coordinates.

    Ports ``DS(LVL) = ABS(X3FACE_GD(LVL) - X3FACE_GD(LVL-1))``
    (``zadvppmwrf.F:246``). ``x3face`` is the full face array including the
    surface, i.e. ``nlays + 1`` values, matching CMAQ's ``X3FACE_GD(0:NLAYS)``.

    The result is dimensionless (sigma coordinate) and constant in space and
    time, which is why CMAQ can ``SAVE`` the PPM mesh coefficients derived from
    it and why we can precompute them here.
    """
    faces = np.asarray(x3face, dtype=np.float64)
    if faces.ndim != 1 or faces.size < 2:
        raise ValueError(f"x3face must be 1-D with >= 2 entries, got shape {faces.shape}")
    return np.abs(np.diff(faces))


@dataclass(frozen=True)
class GridConfig:
    """Domain geometry and the species layout of the advected state.

    The advected state is ``c[ncols, nrows, nlays, nspc_adv]``, matching
    Fortran's ``CGRID(COL, ROW, LAY, SPC)`` so a meteorology reader is a
    straight transpose.
    """

    ncols: int
    nrows: int
    ds: NDArray[np.float64]
    """Sigma layer thicknesses, shape ``(nlays,)``. Build with
    :func:`sigma_layer_thickness`."""

    dx1: float
    """East-west cell width in metres. ``x_ppm.F:215`` (``XCELL_GD``, or the
    ``DG2M``-scaled value on a lat-lon grid)."""

    dx2: float
    """North-south cell width in metres. ``y_ppm.F`` (``YCELL_GD``)."""

    nspc_adv: int
    """Number of advected slots, *including* the rho*J slot.

    Fortran: ``N_SPC_ADV = N_GC_TRNS + N_AE_TRNS + N_NR_TRNS + N_TR_ADV + 1``
    (``hadvppm.F:162``) — the ``+ 1`` is rho*J.
    """

    dtype: DType = "float64"
    """Working precision.

    CMAQ runs in ``float32``. We default to ``float64`` for accuracy, but
    ``float32`` is a supported compute path, not just a comparison target:
    :func:`cmaq_jax.hadv.hadv_step` casts its array inputs to this, and the
    kernels are dtype-transparent from there. Both are tested against the
    goldens.
    """

    ppm: PPMConstants = field(default_factory=PPMConstants)

    def __post_init__(self) -> None:
        if self.ncols < 1 or self.nrows < 1:
            raise ValueError(f"ncols and nrows must be >= 1, got {self.ncols}, {self.nrows}")
        if self.dx1 <= 0.0 or self.dx2 <= 0.0:
            raise ValueError(f"dx1 and dx2 must be positive, got {self.dx1}, {self.dx2}")
        if self.nspc_adv < 1:
            raise ValueError(
                "nspc_adv must be >= 1; slot nspc_adv - 1 carries rho*J, which is "
                "CMAQ's mass-conservation mechanism and is never optional"
            )
        ds = np.asarray(self.ds, dtype=np.float64)
        if ds.ndim != 1 or ds.size < 1:
            raise ValueError(f"ds must be 1-D with >= 1 entry, got shape {ds.shape}")
        if not np.all(ds > 0.0):
            raise ValueError("all layer thicknesses in ds must be positive")
        object.__setattr__(self, "ds", ds)

    @property
    def numpy_dtype(self) -> np.dtype[np.floating]:
        """:attr:`dtype` as a concrete numpy dtype."""
        return np.dtype(self.dtype)

    @property
    def nlays(self) -> int:
        """Number of vertical layers, taken from ``ds``."""
        return int(self.ds.size)

    @property
    def rhoj_index(self) -> int:
        """Index of the rho*J slot in the species axis.

        Fortran advects rho*J as the last species,
        ``ADV_MAP(N_SPC_ADV) = RHOJ_LOC`` (``x_ppm.F:312``). Advecting it with
        the same scheme as everything else is how CMAQ conserves mass.
        """
        return self.nspc_adv - 1

    @property
    def uniform_ds(self) -> bool:
        """Whether the vertical grid is uniform.

        When true the non-uniform PPM reconstruction must reduce to the uniform
        one — a useful cross-check between the two implementations.
        """
        return bool(np.allclose(self.ds, self.ds[0]))
