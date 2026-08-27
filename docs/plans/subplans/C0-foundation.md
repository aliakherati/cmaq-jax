# C0 — Foundation

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md)

Vendor, harness, constants. No JAX yet.

**Gate:** `tri.F`, `matrix1.F` and `eddyx.F` compile unmodified and emit
goldens; `generate_goldens.py --check` is clean.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C0.1** | Vendor `tri.F`, `matrix1.F`, `eddyx.F`, `vdiffacmx.F`, `conv_cgrid.F` + `PROVENANCE.md` rows | sha256 recorded; byte-identical to `origin/5.5+` | `git diff` vs upstream |
| **C0.2** | `config.py`: `ACM2Constants` — `theta`, the 0.75 sub-step factor, `KARMAN`, `GAMAH`, `BETAH`, `RIC` | Every constant carries its Fortran line | `pytest tests/unit/test_config.py` |
| **C0.3** | Harness `harness_tri.f90`, `harness_matrix1.f90` | Both compile against the existing stub layer | `make -C reference` |
| **C0.4** | Goldens for both solvers | Includes an ill-conditioned and a deep-CBL case | `python scripts/generate_goldens.py --check` |

## Notes

**The solvers are the cheapest possible harness** — `tri.F` needs only
`VGRD_DEFN` and `CGRID_SPCS`, both of which the advection stubs already provide,
and `matrix1.F` adds only `UTILIO_DEFN`. Neither touches the I/O API. Start
here: they are pure linear algebra, so a mismatch is unambiguous.

**`eddyx.F` is a different matter** and may not be worth harnessing. It reads
~11 fields off `Met_Data` (`ASX_DATA_MOD.F`, 1154 lines), which the advection
harness never opened. Decide at C0.3 whether to harness it or to validate it
against the analytic similarity relations it implements; the second is weaker
evidence but may cost an order of magnitude less. Record which was chosen and
why — an unharnessed kernel should be labelled as such in the README status
table rather than left looking equivalent to the others.

**`ALPHA` underflow needs its own case.** `matrix1.F` accumulates
`ALPHA = Π(−E/B)` over the CBL. A deep CBL with strong mixing drives it toward
zero in float32, and it divides into `GAMA`. One golden case should be built to
make that as bad as it realistically gets, and the observed magnitude recorded.
