# C2 — The ACM2 driver

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md) · Depends on C1

**Gate:** the two-stage semi-implicit step matches `vdiffacmx.F` on both a
convective and a stable column, over multiple sub-steps.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C2.1** | `vdiff.py`: `substep_limit` (`vdiffacmx.F:457-516`) | Reproduces `NLP` for a given `SEDDY`; ceiling not floor | `pytest tests/unit -k substep` |
| **C2.2** | `vdiff.py`: convective stage — `MBAR`, `MBARKS`, `MDWN`, the first-column matrix | Matches on a convective column; masked correctly above `LCBL` | `pytest tests/regression -k convective` |
| **C2.3** | `vdiff.py`: local stage — the tridiagonal assembly and solve | Matches on a stable column, where the convective stage is skipped entirely | `pytest tests/regression -k local` |
| **C2.4** | `vdiff.py`: `vdiff_step` — surface exchange, sub-step loop, both stages | Matches the driver golden | `pytest tests/regression -k vdiff_driver` |
| **C2.5** | Property tests: mass conservation, positivity, well-mixed limit | A column with no surface flux conserves mass; strong mixing → uniform | `pytest tests/properties -k vdiff` |

## Notes

**The stable column is the sharper first target.** With `CONVCT` false the
convective stage is skipped and the whole thing reduces to Crank–Nicolson
diffusion with a surface flux — far easier to reason about, and if that does not
match there is no point debugging the ACM2 terms.

**Mass conservation has a boundary term here, unlike `hdiff`.** The surface
exchange is a genuine flux in and out of the column, so "mass is conserved" only
holds with `depv = 0` and no emissions. Assert that case exactly, and assert the
flux balance — mass lost equals accumulated dry deposition — in the general one.
Picking one loose tolerance across both would hide a sign error in the
deposition term.

**The well-mixed limit is the ACM2-specific test.** Given long enough and strong
enough mixing, a convective column must approach a uniform mixing ratio. That is
what the non-local plume exists to produce, and it is the property that fails if
`MBARKS`/`MDWN` are mismatched — the up- and down-mixing have to balance, or the
column drifts rather than homogenising.
