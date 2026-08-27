# C2 — The ACM2 driver

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md) · Depends on C1

**Gate: C2.1–C2.4 passed.** `vdiff_step` matches `vdiffacmx.F` across 8 cases in
both precisions — concentrations worst 9.4 float32 ULPs (on the 46-sub-step
case; every other is under 2), dry deposition worst 1.3. C2.5 (property tests)
remains.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C2.1** ✅ | `vdiff.py`: `substep_limit` (`vdiffacmx.F:457-516`) | Reproduces `NLP` for a given `SEDDY`; ceiling not floor | `pytest tests/unit -k substep` |
| **C2.2** ✅ | `vdiff.py`: convective stage — `MBAR`, `MBARKS`, `MDWN`, the first-column matrix | Matches on a convective column; masked correctly above `LCBL` | `pytest tests/regression -k convective` |
| **C2.3** ✅ | `vdiff.py`: local stage — the tridiagonal assembly and solve | Matches on a stable column, where the convective stage is skipped entirely | `pytest tests/regression -k local` |
| **C2.4** ✅ | `vdiff.py`: `vdiff_step` — surface exchange, sub-step loop, both stages | Matches the driver golden | `pytest tests/regression -k vdiff_driver` |
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

**`depv` cannot be zero.** The surface layer relaxes toward `POL = PLDV/DEPV`
(`vdiffacmx.F:625`), so a zero deposition velocity gives 0/0 and NaNs the whole
column — including species with no emission, since the division happens before
any branch on `PLDV`. A "no deposition" case therefore uses a negligible value
rather than zero, and `SurfaceExchange` says so.

**The ragged sub-step loop became a masked scan.** `NLP` is per column, so every
column runs `max_substeps` scan iterations and masks those past its own `NLP`.
Each column then takes exactly its own `NLP` steps of its own `DTSEC/NLP`, which
is CMAQ's arithmetic unchanged — the loop is rectangular, not the computation.
`substep_counts` is separate from `vdiff_step` because the bound has to be a
Python integer for the scan length, and hiding that reduction inside the jitted
step would force a host sync every call.

**One bug the concentrations could not have caught.** The evasion term
(`- DTS·DENS1·PLDV`) appears in *both* halves of the Crank–Nicolson step for a
plain species; only the heterogeneous-HONO branches omit it from the second
(`vdiffacmx.F:696` and `1104-1106`). Copying their form cost exactly
`THBAR·DTS·DENS1·PLDV` in the deposition accumulator while the concentrations
still matched to 0.4 ULPs. Deposition is accumulated rather than solved, so it
needs its own assertion — and a case with nonzero `PLDV`, which is why one
exists and why a guard test checks that it does.
