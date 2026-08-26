"""Smoke test: the package imports and reports a version."""

import cmaq_jax


def test_version_present() -> None:
    assert cmaq_jax.__version__
