# A0 — Foundation

Parent: [`../PLAN-advection.md`](../PLAN-advection.md)

Repository, Fortran golden harness, constants, and the two 1-D PPM kernels.
Nothing here knows about grids, winds, or boundaries — just the scheme.

**Gate:** both kernels match the Fortran goldens at rtol ≤ 1e-5 (float32
downcast) across every case, and the 1-D analytic tests pass.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **A0.1** | Repo scaffold: layout, `pyproject.toml`, `CLAUDE.md`, LICENSE, CI, vendored Fortran + `PROVENANCE.md` | `pip install -e ".[dev]"` succeeds; `ruff`/`mypy` clean on an empty package | `ruff check . && mypy --strict src/` |
| **A0.2** | `config.py` — `GridConfig`, `PPMConstants`. Every constant traced to its Fortran line | No magic number appears outside `config.py` | `pytest tests/unit/test_config.py` |
| **A0.3** | Fortran harness: `stubs.f90`, `harness_hppm.f90`, `harness_vppm.f90`, `Makefile` | `hppm.F`/`vppm.F` compile **unmodified**; harness runs | `make -C reference` |
| **A0.4** | `scripts/generate_goldens.py` + `--check` drift mode; goldens committed under `data/goldens/` | One fresh process per `(NI, NSPCS)`; `--check` is clean on a fresh run | `python scripts/generate_goldens.py --check` |
| **A0.5** | `ppm.py`: `ppm_parabola_uniform` (`hppm.F:283-353`) | Matches `hppm` goldens on the reconstruction arrays | `pytest tests/regression -k uniform` |
| **A0.6** | `ppm.py`: `ppm_parabola_nonuniform` (`vppm.F:396-544`) incl. mesh coefficients | Matches `vppm` goldens; reduces to the uniform form when `ds` is constant | `pytest tests/regression -k nonuniform` |
| **A0.7** | `ppm.py`: `ppm_flux_update` (`hppm.F:377-445`) — upwind fluxes + conservative update | Matches goldens; exactly conservative in exact arithmetic | `pytest tests/regression -k flux` |
| **A0.8** | 1-D property tests: monotonicity, positivity, mass conservation, constancy | All pass for square wave, Gaussian, spike, near-CFL-1 | `pytest tests/properties` |
| **A0.9** | Figures `docs/figures/a0/` + `scripts/make_a0_figures.py` | Square-wave and Gaussian advection vs. exact; limiter action visualised | `python scripts/make_a0_figures.py` |

## Notes

**The `SAVE` trap.** `hppm.F` and `vppm.F` allocate work arrays on first call and
`SAVE` them. A second call with a different `NI` or `NSPCS` reuses the
first-call sizes and returns silently wrong numbers. `generate_goldens.py` must
spawn **one process per configuration**. If goldens ever look subtly off, check
this before anything else.

**Harness stub surface** is deliberately tiny:

- `hppm.F` → `M3EXIT`, `XSTAT1` (`UTILIO_DEFN`); `BUDGET_DIAG`, `BUDGET_HPPM`
  (`PA_DEFN`); `SUBST_HI_LO_BND_PE`. `USE HGRD_DEFN` is vestigial — nothing from
  it is referenced.
- `vppm.F` → `M3EXIT`, `XSTAT1`; `N_GC_TRNS`, `N_AE_TRNS`, `N_NR_TRNS`,
  `N_TR_ADV` (`CGRID_SPCS`).

`SUBST_HI_LO_BND_PE` is a cpp macro (`bldit_cctm.csh:413`), so compile with
`-cpp -DSUBST_HI_LO_BND_PE=stub_hi_lo_bnd_pe` and have the stub set both flags
`.TRUE.` (serial ⇒ the whole domain is a boundary). Build **without**
`-Dparallel`, `-Disam`, `-Dsens`.

**Golden cases:** smooth random field; step function; single-cell spike; uniform
field with constant wind (must be preserved exactly); reversing wind; zero wind;
Courant number near 1; non-uniform `ds` (VPPM only).
