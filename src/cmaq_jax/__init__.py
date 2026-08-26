"""Differentiable JAX port of the CMAQ advection core.

Ports the piecewise parabolic method (PPM) transport operators from the
Community Multiscale Air Quality model — horizontal (``HADV``) and vertical
(``ZADV``) advection — following Colella & Woodward (1984).

The Fortran being ported is vendored verbatim under ``reference/fortran``; see
``reference/PROVENANCE.md``. Module docstrings cite the file and line range they
implement.
"""

import jax

# JAX defaults to float32. This package documents float64 as its working
# precision (see README, "Deliberate deviations from the Fortran"), so the flag
# is set on import rather than left to every caller to remember -- getting it
# wrong silently halves the precision instead of raising. It has to happen
# before the first array is created, which importing the package guarantees.
# The ignore is for jax.config.update being unannotated upstream.
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

__version__ = "0.1.0"

__all__ = ["__version__"]
