"""Differentiable JAX port of the CMAQ advection core.

Ports the piecewise parabolic method (PPM) transport operators from the
Community Multiscale Air Quality model — horizontal (``HADV``) and vertical
(``ZADV``) advection — following Colella & Woodward (1984).

The Fortran being ported is vendored verbatim under ``reference/fortran``; see
``reference/PROVENANCE.md``. Module docstrings cite the file and line range they
implement.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
