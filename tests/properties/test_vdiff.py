"""C2.5 — properties of the ACM2 vertical-diffusion operator.

What a golden cannot say: that the operator conserves what it should, that its
surface flux balances, and that the non-local plume actually homogenises a
convective column — which is the whole reason ACM2 exists.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np
import pytest

from cmaq_jax.vdiff import (
    ColumnState,
    SurfaceExchange,
    VerticalMeteorology,
    eddy_diffusivity,
    substep_counts,
    vdiff_step,
)

NCOLS, NROWS, NLAYS, NSPC = 3, 2, 16, 2

#: The surface relaxation divides by `depv` (`vdiffacmx.F:625`), so "no
#: deposition" has to be a negligible velocity rather than zero — at zero the
#: whole column becomes NaN. 1e-12 m/s removes ~1e-10 of the column per step.
NO_DEPOSITION = 1.0e-12


def build(
    *,
    seddy: float = 20.0,
    convective: bool = False,
    lpbl: int = 6,
    depv: float = NO_DEPOSITION,
    pldv: float = 0.0,
    emis: float = 0.0,
    profile: np.ndarray | None = None,
    top: float = 3000.0,
):
    face = np.linspace(60.0, top, NLAYS)
    shape2, shape3 = (NCOLS, NROWS), (NCOLS, NROWS, NLAYS)

    # The top layer gets zero diffusivity, which is what `eddyx.F` returns: Kz
    # lives on layer interfaces and there is no interface above the top layer.
    # It is not cosmetic -- see TestTheModelTop.
    kz = np.full(shape3, seddy)
    kz[..., -1] = 0.0

    state = ColumnState(
        seddy=kz,
        zf=np.broadcast_to(face, shape3) + 0.0,
        zh=np.broadcast_to(face - 30.0, shape3) + 0.0,
        pbl=np.full(shape2, float(face[lpbl - 1])),
        lpbl=np.full(shape2, lpbl, dtype=np.int32),
        hol=np.full(shape2, -3.0 if convective else 4.0),
        dens1=np.full(shape2, 1.2),
        rdepvht=np.full(shape2, 1.0 / face[0]),
        convective=np.full(shape2, convective),
    )

    if profile is None:
        profile = np.zeros(NLAYS)
        profile[0] = 100.0
    conc = np.zeros((NCOLS, NROWS, NLAYS, NSPC))
    conc[..., 0] = profile
    conc[..., 1] = 5.0

    surface = SurfaceExchange(
        depv=np.full((NCOLS, NROWS, NSPC), depv),
        pldv=np.full((NCOLS, NROWS, NSPC), pldv),
        emis=np.concatenate(
            [
                np.full((NCOLS, NROWS, 1, NSPC), emis),
                np.zeros((NCOLS, NROWS, NLAYS - 1, NSPC)),
            ],
            axis=2,
        ),
    )
    # Layer thickness: dzh[0] = zf[0], then differences.
    dzh = np.concatenate([[face[0]], np.diff(face)])
    return conc, state, surface, dzh


def run(dtsec: float = 300.0, **kwargs):
    conc, state, surface, dzh = build(**kwargs)
    bound = int(np.asarray(substep_counts(state, dtsec)).max())
    out, ddep = vdiff_step(conc, state, surface, dtsec=dtsec, max_substeps=bound)
    return np.asarray(out), np.asarray(ddep), conc, dzh, bound


def mass(field: np.ndarray, dzh: np.ndarray, spc: int = 0) -> float:
    """Column burden. Summing concentrations alone is not mass — the layers
    have very different thicknesses."""
    return float((field[0, 0, :, spc] * dzh).sum())


class TestMass:
    @pytest.mark.parametrize("convective", [False, True])
    def test_a_closed_column_conserves_mass(self, convective: bool) -> None:
        """With no deposition and no emission, nothing leaves the column: the
        top is a no-flux boundary and the surface exchange is switched off. Any
        drift is a leak in the operator rather than physics.
        """
        out, _, before, dzh, _ = run(convective=convective)
        np.testing.assert_allclose(mass(out, dzh), mass(before, dzh), rtol=1e-9)

    def test_it_conserves_across_deep_substepping(self) -> None:
        """Many passes, so a per-sub-step leak would accumulate visibly."""
        out, _, before, dzh, bound = run(seddy=400.0, dtsec=3600.0, convective=True)
        assert bound > 10, f"only {bound} sub-steps; this test needs more"
        np.testing.assert_allclose(mass(out, dzh), mass(before, dzh), rtol=1e-8)

    def test_deposition_removes_mass(self) -> None:
        out, ddep, before, dzh, _ = run(depv=0.01)
        assert mass(out, dzh) < mass(before, dzh)
        assert float(ddep[0, 0, 0]) > 0.0

    def test_more_deposition_removes_more(self) -> None:
        losses = []
        for depv in (0.002, 0.01, 0.05):
            out, _, before, dzh, _ = run(depv=depv)
            losses.append(mass(before, dzh) - mass(out, dzh))
        assert losses[0] < losses[1] < losses[2]

    def test_emissions_add_mass(self) -> None:
        out, _, before, dzh, _ = run(emis=0.5)
        assert mass(out, dzh) > mass(before, dzh)


class TestNoOp:
    def test_zero_diffusivity_and_no_surface_flux_changes_nothing(self) -> None:
        """The operator has to be able to do nothing. If this fails, every
        conservation test above is measuring the wrong thing."""
        out, _, before, _, _ = run(seddy=1.0e-20)
        np.testing.assert_allclose(out, before, rtol=1e-9, atol=1e-12)

    def test_a_uniform_profile_is_untouched(self) -> None:
        """Diffusion is driven by gradients, so a uniform column has nothing to
        do — including in a convective column, where the plume's up- and
        down-mixing must cancel exactly. That balance is what `MBARKS(LCBL) =
        MDWN(LCBL)` (`vdiffacmx.F:501`) exists to enforce, and it is the closure
        most likely to be got wrong.
        """
        out, _, _initial, _, _ = run(profile=np.full(NLAYS, 7.0), convective=True, seddy=50.0)
        np.testing.assert_allclose(out[..., 0], 7.0, rtol=1e-9)


class TestTheModelTop:
    """The top boundary is closed only because the diffusivity is zero there.

    `vdiffacmx.F:675` sets `BB2(L) = 1 - CC(L) - EE2(L)` for every layer
    including the last, so the top row's diagonal carries an upward-flux term
    `EE2(NLAYS) = -DFSP(NLAYS)*EDDY(NLAYS)`. The right-hand side at that layer
    has no matching `LFAC3` term (`vdiffacmx.F:1007-1008`), so the flux is
    one-sided: mass leaves through the model top into nothing.

    It never happens in CMAQ, because `eddyx.F` returns `EDDYV = 0` for the top
    layer — Kz lives on interfaces and the top layer has none above it. The two
    facts are load-bearing together: the scheme is conservative because the
    diffusivity happens to vanish exactly where the matrix would otherwise leak.

    Verified against the Fortran, which leaks identically: a uniform column with
    a nonzero top-layer Kz loses mass in proportion to it, and exactly zero when
    it is zero.
    """

    def test_a_nonzero_top_diffusivity_leaks_mass(self) -> None:
        conc, state, surface, dzh = build(profile=np.full(NLAYS, 7.0))
        kz = state.seddy.copy()
        kz[..., -1] = 20.0
        leaky = state._replace(seddy=kz)
        bound = int(np.asarray(substep_counts(leaky, 300.0)).max())
        out, _ = vdiff_step(conc, leaky, surface, dtsec=300.0, max_substeps=bound)
        drift = (mass(np.asarray(out), dzh) - mass(conc, dzh)) / mass(conc, dzh)
        assert drift < -1e-4, f"expected a leak, got {drift:+.3e}"

    def test_zeroing_it_closes_the_boundary(self) -> None:
        """The same column with the top-layer diffusivity `eddyx.F` actually
        supplies. This is the contrast that identifies the cause."""
        out, _, initial, dzh, _ = run(profile=np.full(NLAYS, 7.0))
        np.testing.assert_allclose(mass(out, dzh), mass(initial, dzh), rtol=1e-9)

    def test_eddyx_really_does_return_zero_there(self) -> None:
        """Guard on the assumption the two tests above rest on. If `eddyx`
        stopped zeroing the top layer, the operator would start leaking and
        nothing else here would notice."""
        nlays = 8
        shape2, shape3 = (2, 2), (2, 2, nlays)
        dot = (3, 3, nlays)
        met = VerticalMeteorology(
            pbl=np.full(shape2, 1000.0),
            ustar=np.full(shape2, 0.3),
            moli=np.zeros(shape2),
            zf=np.broadcast_to(np.linspace(50.0, 2000.0, nlays), shape3) + 0.0,
            zh=np.broadcast_to(np.linspace(30.0, 1980.0, nlays), shape3) + 0.0,
            kzmin=np.full(shape3, 0.5),
            thetav=np.broadcast_to(300.0 + np.arange(nlays), shape3) + 0.0,
            ta=np.full(shape3, 290.0),
            qv=np.full(shape3, 0.005),
            qc=np.zeros(shape3),
            uwind=np.zeros(dot),
            vwind=np.zeros(dot),
        )
        assert np.all(np.asarray(eddy_diffusivity(met))[..., -1] == 0.0)


class TestMixing:
    def test_a_surface_release_spreads_upward(self) -> None:
        out, _, before, _, _ = run(seddy=50.0)
        assert out[0, 0, 0, 0] < before[0, 0, 0, 0], "the surface layer did not drain"
        assert out[0, 0, 1, 0] > 0.0, "nothing reached the layer above"

    def test_the_plume_mixes_the_cbl_faster_than_local_diffusion(self) -> None:
        """The point of the non-local stage.

        A surface release in a convective column reaches the top of the CBL
        directly, rather than diffusing layer by layer. Compared against the
        same column with `CONVCT` false — the only difference — the concentration
        near the CBL top must be higher.
        """
        common = {"seddy": 30.0, "lpbl": 8, "dtsec": 600.0}
        local, _, _, _, _ = run(convective=False, **common)
        acm2, _, _, _, _ = run(convective=True, **common)
        near_top = 6
        assert acm2[0, 0, near_top, 0] > local[0, 0, near_top, 0], (
            "the non-local plume did not outrun local diffusion"
        )

    def test_a_convective_column_approaches_well_mixed(self) -> None:
        """Given long enough, the CBL homogenises. This is what ACM2 is for, and
        it fails if the up- and down-mixing rates are mismatched: the column
        would drift toward a gradient rather than away from one.
        """
        lpbl = 10
        out, _, _, _, _ = run(seddy=250.0, dtsec=36000.0, convective=True, lpbl=lpbl, top=2000.0)
        inside = out[0, 0, : lpbl - 1, 0]
        spread = (inside.max() - inside.min()) / inside.mean()
        assert spread < 0.35, f"CBL still stratified: spread {spread:.3f}"


class TestBoundedness:
    def test_positivity(self) -> None:
        out, _, _, _, _ = run(seddy=80.0, convective=True)
        assert out[..., 0].min() >= -1e-9

    def test_no_new_maximum_without_sources(self) -> None:
        """Diffusion averages; with no emission it cannot exceed the initial
        peak."""
        out, _, before, _, _ = run(seddy=80.0, convective=True)
        assert out[..., 0].max() <= before[..., 0].max() + 1e-9


class TestJitAndGradients:
    def test_it_jits(self) -> None:
        conc, state, surface, _ = build(convective=True)
        bound = int(np.asarray(substep_counts(state, 300.0)).max())
        step = jax.jit(partial(vdiff_step, dtsec=300.0, max_substeps=bound))
        out, ddep = step(conc, state, surface)
        assert np.all(np.isfinite(np.asarray(out)))
        assert np.all(np.isfinite(np.asarray(ddep)))

    def test_gradients_match_finite_differences(self) -> None:
        """The first operator in this port whose gradient goes through a linear
        solve. JAX differentiates the scan directly; the adjoint of a
        tridiagonal solve is another tridiagonal solve, so if this is ever slow
        the fix is a custom VJP — but correctness comes first.
        """
        conc, state, surface, _ = build(convective=True, seddy=30.0, depv=0.005)
        bound = int(np.asarray(substep_counts(state, 300.0)).max())

        def total(field):
            out, _ = vdiff_step(field, state, surface, dtsec=300.0, max_substeps=bound)
            return (out[..., 0] ** 2).sum()

        grad = np.asarray(jax.grad(total)(conc))

        eps = 1e-4
        rng = np.random.default_rng(0)
        for _ in range(4):
            index = tuple(int(rng.integers(0, n)) for n in conc.shape)
            up, down = conc.copy(), conc.copy()
            up[index] += eps
            down[index] -= eps
            numeric = (float(total(up)) - float(total(down))) / (2 * eps)
            assert numeric == pytest.approx(float(grad[index]), rel=1e-5, abs=1e-7)

    def test_gradients_reach_the_diffusivity(self) -> None:
        """Differentiating with respect to `Kz` is what an inverse problem would
        actually want, and it goes through the sub-step count as well as the
        solve."""
        conc, state, surface, _ = build(convective=True, seddy=30.0)
        bound = int(np.asarray(substep_counts(state, 300.0)).max())

        def total(seddy):
            out, _ = vdiff_step(
                conc,
                state._replace(seddy=seddy),
                surface,
                dtsec=300.0,
                max_substeps=bound,
            )
            return (out[..., 0] ** 2).sum()

        grad = np.asarray(jax.grad(total)(state.seddy))
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0, "the diffusivity gradient is identically zero"
