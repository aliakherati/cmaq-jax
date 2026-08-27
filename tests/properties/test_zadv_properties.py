"""A2.5 — scheme properties of the vertical operator.

Stated in terms of the physics rather than against CMAQ, so they would catch a
drift the goldens agree with.

The column turns out to be **closed at both ends**, which is stronger than it
first appears. The ground is closed by construction — ``zadvppmwrf.F:341`` pins
``FLX(1) = 0``. The model top closes itself: unrolling the flux recurrence gives

    FLX(top) = DRJ * (1 - sum(ds))

and in sigma coordinates ``sum(ds) = faces[0] - faces[-1] = 1`` exactly, so the
top-face flux vanishes identically. Measured, it sits ~15 orders of magnitude
below the interior fluxes — round-off, not physics. Column mass is therefore
exactly conserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmaq_jax.config import DEFAULT_PPM, sigma_layer_thickness
from cmaq_jax.ppm import nonuniform_mesh
from cmaq_jax.vadv import diagnose_flux, face_velocity_from_flux, zadv

DT = 180.0


def sigma_faces(nlays: int, stretch: float = 0.625) -> np.ndarray:
    """CMAQ-like sigma faces: 1.0 at the ground, 0.0 at the model top.

    An exponent below one thins the near-surface layers, which is how CMAQ
    resolves the boundary layer.
    """
    return np.linspace(1.0, 0.0, nlays + 1) ** stretch


def make_column(
    nlays: int,
    ncols: int = 3,
    nrows: int = 2,
    *,
    mismatch: float,
    tracers: list[np.ndarray] | None = None,
    coupled_q: list[float] | None = None,
    seed: int = 20260906,
) -> tuple[np.ndarray, np.ndarray]:
    """A coupled state and the met density it is meant to relax toward.

    Layer axis first, species last, as the vertical operator expects.
    """
    rng = np.random.default_rng(seed)
    rhoj = 1.5 + 0.4 * rng.random((nlays, ncols, nrows))
    drift = mismatch * np.sin(np.linspace(0.0, 2.0 * np.pi, nlays))[:, None, None]
    met = rhoj * (1.0 + drift)

    if coupled_q is not None:
        fields = [q * rhoj for q in coupled_q]
    elif tracers is not None:
        fields = [t * rhoj for t in tracers]
    else:
        fields = [1.0 + rng.random((nlays, ncols, nrows))]
    return np.stack([*fields, rhoj], axis=-1), met


def advect(con: np.ndarray, met: np.ndarray, ds: np.ndarray, dt: float = DT):
    return zadv(con, met, ds, nonuniform_mesh(ds), dt=dt)


class TestClosedGround:
    def test_nothing_crosses_the_surface(self) -> None:
        """zadvppmwrf.F:341 pins FLX(1) = 0. The ground is solid, so whatever
        else happens the bottom face carries no mass."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        con, met = make_column(nlays, mismatch=0.25)
        flx = np.asarray(diagnose_flux(met, con[..., -1], ds, DT))
        vel = np.asarray(face_velocity_from_flux(flx, con[..., -1]))
        np.testing.assert_array_equal(flx[0], 0.0)
        np.testing.assert_array_equal(vel[0], 0.0)

    @pytest.mark.parametrize("stretch", [0.4, 0.625, 1.0, 1.6])
    def test_the_model_top_closes_itself(self, stretch: float) -> None:
        """``FLX(top) = DRJ * (1 - sum(ds))``, and the sigma thicknesses sum to
        one, so the top-face flux vanishes for any layering.

        Worth pinning across several stretchings: the cancellation is a
        property of the coordinate, not of one particular grid, and a change to
        the flux recurrence that broke it would leak mass out of the model top
        while every golden still passed.
        """
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays, stretch))
        assert ds.sum() == pytest.approx(1.0, abs=1e-15)

        con, met = make_column(nlays, mismatch=0.3)
        flx = np.asarray(diagnose_flux(met, con[..., -1], ds, DT))
        interior = float(np.abs(flx).max())
        assert float(np.abs(flx[-1]).max()) < 1e-12 * interior, "the model top is leaking"

    def test_column_mass_is_conserved(self) -> None:
        """Closed at both ends, so the flux-form update telescopes to nothing."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        con, met = make_column(nlays, mismatch=0.3)
        out, diag = advect(con, met, ds)

        before = np.einsum("l,lcr->cr", ds, con[..., 0])
        after = np.einsum("l,lcr->cr", ds, np.asarray(out)[..., 0])
        np.testing.assert_allclose(after, before, rtol=1e-12)

        # And the transport was real, not a no-op that trivially conserves.
        assert float(np.abs(np.asarray(out)[..., 0] - con[..., 0]).max()) > 1e-6
        assert float(np.asarray(diag.max_courant).max()) > 0.1


class TestDensity:
    def test_transported_density_moves_toward_the_meteorology(self) -> None:
        """The whole purpose of the vertical flux. It is a correction, not an
        assignment, so the test is that the gap shrinks -- not that it closes."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        con, met = make_column(nlays, mismatch=0.2)
        out, _ = advect(con, met, ds)

        before = float(np.abs(con[..., -1] - met).max())
        after = float(np.abs(np.asarray(out)[..., -1] - met).max())
        assert after < before, f"density gap grew: {before:.4g} -> {after:.4g}"

    def test_a_matched_column_does_not_move(self) -> None:
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        con, _ = make_column(nlays, mismatch=0.0)
        out, _ = advect(con, con[..., -1].copy(), ds)
        np.testing.assert_allclose(np.asarray(out), con, rtol=1e-12)


class TestConstancy:
    @pytest.mark.parametrize("mismatch", [0.05, 0.2, 0.5])
    def test_uniform_mixing_ratio_survives(self, mismatch: float) -> None:
        """Including at a mismatch large enough to force CFL sub-stepping, where
        the column is advanced in several unequal pieces."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        q = [0.75, 3.0]
        con, met = make_column(nlays, mismatch=mismatch, coupled_q=q)
        out, diag = advect(con, met, ds)
        out = np.asarray(out)
        for spc, expected in enumerate(q):
            np.testing.assert_allclose(out[..., spc] / out[..., -1], expected, rtol=1e-9)
        if mismatch >= 0.5:
            assert int(np.asarray(diag.substeps).max()) > 1, "expected sub-stepping here"


class TestBoundedness:
    def test_positivity_with_a_spike(self) -> None:
        """A sharp layer on a clean background is where an undershoot becomes
        negative rather than merely small."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        spike = np.zeros((nlays, 3, 2))
        spike[nlays // 2] = 8.0
        con, met = make_column(nlays, mismatch=0.4, tracers=[spike])
        out, _ = advect(con, met, ds)
        assert np.asarray(out)[..., 0].min() >= 0.0

    @pytest.mark.parametrize("mismatch", [0.05, 0.15, 0.3, 0.6])
    def test_mixing_ratio_excursions_stay_small(self, mismatch: float) -> None:
        """The mixing ratio may leave its starting range, but only slightly.

        Exact boundedness is **not** a property of this scheme, and asserting
        it would be asserting something false. CMAQ transports rho*q and rho
        separately, each with its own limiter; monotonicity of the two does not
        imply boundedness of their ratio. Getting that guarantee needs a
        consistent (coupled) limiter, which CMAQ does not implement -- what it
        does guarantee is positivity and exact constancy.

        Measured worst case over a sweep of seeds and mismatches is 1.1% of the
        input range. 5% leaves headroom while still catching a real regression:
        a broken limiter would overshoot by far more than this.
        """
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        rng = np.random.default_rng(11)
        profile = 1.0 + rng.random((nlays, 3, 2))
        con, met = make_column(nlays, mismatch=mismatch, tracers=[profile])
        out, _ = advect(con, met, ds)
        out = np.asarray(out)

        q_in = con[..., 0] / con[..., -1]
        q_out = out[..., 0] / out[..., -1]
        span = float(np.ptp(q_in))
        undershoot = max(float(q_in.min() - q_out.min()), 0.0)
        overshoot = max(float(q_out.max() - q_in.max()), 0.0)
        assert max(undershoot, overshoot) <= 0.05 * span, (
            f"excursion {max(undershoot, overshoot) / span:.1%} of the input range"
        )


class TestSubstepping:
    def test_columns_advance_independently(self) -> None:
        """A grid where one column needs sub-stepping and another does not must
        give each the same answer it would get alone. The masked fixed-count
        loop is exactly where that could leak."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        calm, calm_met = make_column(nlays, ncols=1, nrows=1, mismatch=0.02, seed=1)
        rough, rough_met = make_column(nlays, ncols=1, nrows=1, mismatch=0.6, seed=1)

        together = np.concatenate([calm, rough], axis=1)
        together_met = np.concatenate([calm_met, rough_met], axis=1)
        out_together, diag = advect(together, together_met, ds)
        out_together = np.asarray(out_together)

        assert int(np.asarray(diag.substeps)[0, 0]) == 1
        assert int(np.asarray(diag.substeps)[1, 0]) > 1, "the rough column should sub-step"

        for index, (state, state_met) in enumerate(((calm, calm_met), (rough, rough_met))):
            alone, _ = advect(state, state_met, ds)
            np.testing.assert_allclose(np.asarray(alone)[:, 0], out_together[:, index], rtol=1e-12)

    def test_substeps_stay_inside_the_cap(self) -> None:
        """A column that exhausts the fixed iteration count is reported as an
        infinite residual. Even a violent mismatch should stay well clear."""
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays))
        con, met = make_column(nlays, mismatch=1.2)
        _, diag = advect(con, met, ds)
        assert np.all(np.isfinite(np.asarray(diag.residual)))
        assert int(np.asarray(diag.substeps).max()) < DEFAULT_PPM.max_substeps


class TestFloat32:
    """CMAQ's own precision, and the likeliest GPU choice."""

    def test_invariants_hold(self) -> None:
        nlays = 16
        ds = sigma_layer_thickness(sigma_faces(nlays)).astype(np.float32)
        q = [0.75, 3.0]
        con, met = make_column(nlays, mismatch=0.2, coupled_q=q)
        con, met = con.astype(np.float32), met.astype(np.float32)

        out, _ = zadv(con, met, ds, nonuniform_mesh(ds), dt=DT)
        out = np.asarray(out)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        for spc, expected in enumerate(q):
            np.testing.assert_allclose(out[..., spc] / out[..., -1], expected, rtol=1e-5)
