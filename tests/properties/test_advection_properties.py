"""A0.8 — the mathematical guarantees PPM is chosen for.

These are the properties the Fortran comparison cannot check. A golden pins the
port to CMAQ; if CMAQ itself drifted, or a case simply never exercised a branch,
the goldens would happily agree with the wrong answer. These tests are stated in
terms of the scheme, not the reference:

* it converges to the analytic solution as the grid refines;
* it conserves mass;
* it never creates a new extremum, so no spurious oscillation;
* it keeps non-negative fields non-negative;
* a uniform mixing ratio survives a divergent wind.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmaq_jax.config import DEFAULT_PPM, sigma_layer_thickness
from cmaq_jax.ppm import nonuniform_mesh, ppm_advect_uniform, ppm_parabola_nonuniform

SWP = DEFAULT_PPM.halo_width


def _periodic_halo(interior: np.ndarray) -> np.ndarray:
    """Wrap the domain onto itself.

    A test-local helper, not library behaviour: CMAQ fills the halo from
    boundary-condition files or the outflow condition, which is chunk A1.2.
    Periodic wrapping is what makes an analytic solution available here.
    """
    return np.concatenate([interior[-SWP:], interior, interior[:SWP]])


def _advect_periodic(
    interior: np.ndarray,
    velocity: float,
    dt: float,
    ds: float,
    nsteps: int,
) -> np.ndarray:
    """Advect a periodic 1-D field for `nsteps`, refilling the halo each step."""
    field = interior
    vel = np.full((field.shape[0] + 1, 1), velocity, dtype=np.float64)
    for _ in range(nsteps):
        padded = np.asarray(ppm_advect_uniform(_periodic_halo(field), vel, dt, ds))
        field = padded[SWP:-SWP]
    return field


def _gaussian(n: int, centre: float, width: float) -> np.ndarray:
    """A periodic Gaussian on the unit interval, sampled at cell centres."""
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    offset = (x - centre + 0.5) % 1.0 - 0.5
    return np.exp(-(offset**2) / (2.0 * width**2))


class TestAccuracy:
    def test_converges_on_smooth_data(self) -> None:
        """Refining the grid must reduce the error at roughly second order.

        PPM is formally third order, but the monotonicity limiter clips the
        reconstruction at the Gaussian's peak and costs an order there. Around
        2 is the expected, correct behaviour -- markedly less would mean the
        limiter is firing where it should not; markedly more would mean it is
        not firing at all.
        """
        courant, width, distance = 0.5, 0.08, 0.25
        errors = []
        resolutions = [64, 128, 256, 512]

        for n in resolutions:
            ds = 1.0 / n
            dt = courant * ds
            nsteps = round(distance / dt)
            initial = _gaussian(n, centre=0.5, width=width)
            final = _advect_periodic(initial[:, None], 1.0, dt, ds, nsteps)
            exact = _gaussian(n, centre=0.5 + dt * nsteps, width=width)
            errors.append(float(np.abs(final[:, 0] - exact).mean()))

        orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert orders.min() > 1.7, f"convergence too slow: {orders}"
        assert float(np.mean(orders)) > 1.9, f"mean order {np.mean(orders)}"
        assert errors[-1] < 1e-4

    def test_advecting_a_full_period_returns_the_field(self) -> None:
        """After one full revolution the feature is back where it started, so
        the only difference is the scheme's own numerical diffusion."""
        n, courant, width = 256, 0.5, 0.1
        ds = 1.0 / n
        dt = courant * ds
        nsteps = round(1.0 / dt)

        initial = _gaussian(n, centre=0.5, width=width)
        final = _advect_periodic(initial[:, None], 1.0, dt, ds, nsteps)

        assert float(np.abs(final[:, 0] - initial).max()) < 0.02
        # Diffusion lowers the peak; it must not raise it.
        assert final[:, 0].max() <= initial.max() + 1e-12


class TestConservation:
    def test_mass_is_conserved_for_a_compact_feature(self) -> None:
        """Total mass is exactly conserved when nothing crosses the boundary.

        The background is zero and the feature stays clear of both ends, so the
        domain-edge donor-cell fluxes (``hppm.F:422-439``) are zero and the
        scheme's flux-form telescoping is exact.
        """
        n, ds, dt = 200, 1.0, 0.25
        x = np.arange(n, dtype=np.float64)
        feature = np.exp(-((x - 60.0) ** 2) / (2.0 * 8.0**2))
        feature[feature < 1e-12] = 0.0

        field = feature[:, None]
        initial_mass = float(field.sum()) * ds
        vel = np.full((n + 1, 1), 1.0)

        for _ in range(40):
            padded = np.concatenate(
                [np.zeros((SWP, 1)), field, np.zeros((SWP, 1))]
            )  # vacuum outside
            field = np.asarray(ppm_advect_uniform(padded, vel, dt, ds))[SWP:-SWP]

        assert field[-1, 0] < 1e-12, "feature reached the boundary; test is invalid"
        np.testing.assert_allclose(float(field.sum()) * ds, initial_mass, rtol=1e-12)

    def test_periodic_advection_conserves_mass_in_the_interior(self) -> None:
        """A periodic wrap conserves mass to the level the boundary treatment
        allows.

        Not exactly: CMAQ's outermost faces use a donor-cell flux rather than
        the parabola, so the mass leaving the right edge and the mass entering
        the left edge are computed by slightly different formulas. The drift is
        a numerical-diffusion effect, not a leak, and stays tiny.
        """
        n, ds, dt = 128, 1.0 / 128, 0.5 / 128
        initial = _gaussian(n, centre=0.5, width=0.1)
        final = _advect_periodic(initial[:, None], 1.0, dt, ds, 64)
        np.testing.assert_allclose(final.sum(), initial.sum(), rtol=1e-3)


class TestMonotonicity:
    def test_square_wave_develops_no_oscillation(self) -> None:
        """The reason to use PPM rather than an unlimited high-order scheme.

        A discontinuity is where a third-order method would ring; the limiter
        must hold the solution inside the original bounds for every step.
        """
        n, ds, dt = 200, 1.0, 0.5
        square = np.where((np.arange(n) >= 60) & (np.arange(n) < 120), 3.0, 1.0)

        field = square[:, None].astype(np.float64)
        for _ in range(60):
            padded = _periodic_halo(field)
            field = np.asarray(ppm_advect_uniform(padded, np.full((n + 1, 1), 1.0), dt, ds))[
                SWP:-SWP
            ]
            assert field.min() >= 1.0 - 1e-10, "undershoot: limiter failed"
            assert field.max() <= 3.0 + 1e-10, "overshoot: limiter failed"

    def test_positivity_is_preserved(self) -> None:
        """A tracer concentration must never go negative, whatever the wind.

        Uses a sharp spike on a zero background -- the hardest case, since any
        undershoot is immediately negative rather than merely too small.
        """
        n, ds, dt = 100, 1.0, 0.4
        spike = np.zeros(n)
        spike[50] = 10.0

        field = spike[:, None]
        wind = 0.9 * np.sin(2.0 * np.pi * np.arange(n + 1) / n)[:, None]
        for _ in range(50):
            padded = _periodic_halo(field)
            field = np.asarray(ppm_advect_uniform(padded, wind, dt, ds))[SWP:-SWP]
            assert field.min() >= 0.0, f"negative concentration: {field.min()}"


class TestConstancy:
    def test_uniform_mixing_ratio_survives_a_divergent_wind(self) -> None:
        """The CMAQ-specific invariant, over many steps.

        The state is in coupled units -- slot ``s`` holds rho*J*q_s and the last
        slot holds rho*J -- and rho*J is advected by the same scheme as
        everything else. That is the whole reason CMAQ carries it as a species.
        Break the ride-along and this is the test that notices.
        """
        n, ds, dt = 64, 1.0, 0.2
        x = np.arange(n, dtype=np.float64)
        rhoj = 1.0 + 0.3 * np.sin(2.0 * np.pi * x / n)
        q = np.array([0.5, 1.0, 4.0])

        field = np.stack([*(qq * rhoj for qq in q), rhoj], axis=1)
        wind = 0.8 * np.sin(4.0 * np.pi * np.arange(n + 1) / n)[:, None]
        assert np.ptp(wind) > 1.0, "wind must diverge for this to mean anything"

        for _ in range(80):
            padded = _periodic_halo(field)
            field = np.asarray(ppm_advect_uniform(padded, wind, dt, ds))[SWP:-SWP]

        for spc, q_expected in enumerate(q):
            np.testing.assert_allclose(field[:, spc] / field[:, -1], q_expected, rtol=1e-11)

    def test_uniform_field_is_stationary(self) -> None:
        """With a uniform field and a uniform wind, nothing should change at
        all -- not merely to a tolerance."""
        n, ds, dt = 50, 1.0, 0.5
        field = np.full((n, 2), 2.5)
        for _ in range(20):
            padded = _periodic_halo(field)
            field = np.asarray(ppm_advect_uniform(padded, np.full((n + 1, 1), 1.0), dt, ds))[
                SWP:-SWP
            ]
        np.testing.assert_array_equal(field, np.full((n, 2), 2.5))


class TestVerticalReconstruction:
    """The non-uniform reconstruction on realistic CMAQ sigma layers."""

    @staticmethod
    def _cmaq_like_sigma(nlays: int) -> np.ndarray:
        """Thin layers near the ground, thick aloft, as CMAQ stretches them.

        Sigma runs 1.0 at the surface to 0.0 at the model top, so an exponent
        below 1 thins the near-surface layers. Above 1 would invert it and put
        the thickest layer against the ground.
        """
        faces = np.linspace(1.0, 0.0, nlays + 1) ** 0.625
        return sigma_layer_thickness(faces)

    def test_constant_profile_stays_flat(self) -> None:
        ds = self._cmaq_like_sigma(35)
        cn = np.full(35, 3.0)
        parabola = ppm_parabola_nonuniform(cn, nonuniform_mesh(ds))
        np.testing.assert_array_equal(np.asarray(parabola.dc), np.zeros(35))
        np.testing.assert_array_equal(np.asarray(parabola.c6), np.zeros(35))

    def test_never_reconstructs_outside_the_data_range(self) -> None:
        """Every edge value must lie within the profile's own range, on a
        strongly stretched grid where naive weights would overshoot."""
        rng = np.random.default_rng(20260829)
        ds = self._cmaq_like_sigma(35)
        mesh = nonuniform_mesh(ds)

        for _ in range(50):
            cn = rng.random(35) * 10.0
            parabola = ppm_parabola_nonuniform(cn, mesh)
            tol = 1e-10
            for edge in (parabola.cl, parabola.cr):
                assert float(np.asarray(edge).min()) >= cn.min() - tol
                assert float(np.asarray(edge).max()) <= cn.max() + tol

    @pytest.mark.parametrize("nlays", [4, 5, 12, 35])
    def test_works_across_realistic_layer_counts(self, nlays: int) -> None:
        """CMAQ configurations run from a handful of layers to 35+; the
        reduced-order end treatment must not break at the small end."""
        ds = self._cmaq_like_sigma(nlays)
        cn = 1.0 + np.arange(nlays, dtype=np.float64)
        parabola = ppm_parabola_nonuniform(cn, nonuniform_mesh(ds))
        assert np.all(np.isfinite(np.asarray(parabola.cl)))
        assert np.all(np.isfinite(np.asarray(parabola.cr)))
