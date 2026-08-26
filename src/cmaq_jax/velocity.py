"""Face velocities for the horizontal sweeps.

Ports ``hcontvel.F``. The name is slightly misleading in modern CMAQ: on the
default path there is no contravariant transformation left to do.

``hcontvel.F`` has two branches, chosen once at startup by whether the
dot-point meteorology file carries C-staggered winds:

* **C-staggered** (``CSTAGUV = .TRUE.``, the default since MCIP v3.5, Fall
  2009) -- ``UWINDC``/``VWINDC`` are already on the flux faces, and
  ``hcontvel.F:245-260`` returns them **directly, with an early RETURN**. No
  density, no Jacobian.
* **B-staggered fallback** -- when ``UWINDC`` is absent, the routine reads
  ``UHAT_JD``/``VHAT_JD`` (contravariant velocity times Jacobian times air
  density) and divides by the density interpolated to the face
  (``hcontvel.F:329-351``).

The early RETURN is easy to miss, and taking the fallback for the default case
would divide by a density that should not be there. Both are provided here;
:func:`face_velocity` selects between them.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["face_velocity", "face_velocity_from_flux"]


def face_velocity_from_flux(wind_jd: Array, rhoj: Array, axis: int) -> Array:
    """Recover face velocity from the density-weighted flux (legacy path).

    Ports ``hcontvel.F:329-351``. ``wind_jd`` is ``UHAT_JD`` or ``VHAT_JD`` on
    the faces of ``axis``; ``rhoj`` is ``DENSA_J`` at cell centres. The density
    is averaged onto each face, matching
    ``DJ = 0.5*(DENSJ(COL,ROW) + DENSJ(COL-1,ROW))``.

    The domain-edge faces have only one neighbouring cell. CMAQ fills the other
    side from the halo, which off-domain is the boundary density; here the edge
    cell's own value is used, i.e. a zero-gradient extrapolation.
    """
    wind_jd = jnp.asarray(wind_jd)
    rhoj = jnp.asarray(rhoj)

    moved = jnp.moveaxis(rhoj, axis, 0)
    padded = jnp.concatenate([moved[:1], moved, moved[-1:]])
    face_density = 0.5 * (padded[:-1] + padded[1:])
    face_density = jnp.moveaxis(face_density, 0, axis)

    if face_density.shape != wind_jd.shape:
        raise ValueError(
            f"face density has shape {face_density.shape} but wind has {wind_jd.shape}; "
            f"wind must be on the faces of axis {axis}"
        )
    return wind_jd / face_density


def face_velocity(
    wind: Array,
    axis: int,
    rhoj: Array | None = None,
) -> Array:
    """Face velocity for a sweep along ``axis``.

    With ``rhoj`` omitted this is the C-staggered path: ``wind`` is
    ``UWINDC``/``VWINDC`` and is returned unchanged, exactly as
    ``hcontvel.F`` does when ``CSTAGUV`` is true. Passing ``rhoj`` selects the
    pre-MCIPv3.5 fallback, where ``wind`` is ``UHAT_JD``/``VHAT_JD``.

    ``wind`` must be dimensioned on the faces of ``axis`` -- one longer than
    the cell count along it.
    """
    wind = jnp.asarray(wind)
    if rhoj is None:
        return wind
    return face_velocity_from_flux(wind, rhoj, axis)
