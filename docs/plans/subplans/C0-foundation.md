# C0 — Foundation

Parent: [`../PLAN-vdiff.md`](../PLAN-vdiff.md)

Vendor, harness, constants. No JAX yet.

**Gate: solvers passed.** `tri.F` and `matrix1.F` compile unmodified; 9 golden
cases committed, `--check` clean. `eddyx.F` and `C0.2` remain — see the note on
`eddyx` below, which is still an open decision.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **C0.1** ✅ | Vendor `tri.F`, `matrix1.F`, `eddyx.F`, `vdiffacmx.F`, `conv_cgrid.F` + `PROVENANCE.md` rows | sha256 recorded; byte-identical to `origin/5.5+` | `git diff` vs upstream |
| **C0.2** | `config.py`: `ACM2Constants` — `theta`, the 0.75 sub-step factor, `KARMAN`, `GAMAH`, `BETAH`, `RIC` | Every constant carries its Fortran line | `pytest tests/unit/test_config.py` |
| **C0.3** ✅ | Harness `harness_tri.f90`, `harness_matrix1.f90` | Both compile against the existing stub layer | `make -C reference` |
| **C0.4** ✅ | Goldens for both solvers | Includes an ill-conditioned and a deep-CBL case | `python scripts/generate_goldens.py --check` |

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

**`ALPHA` underflow: measured, and benign.** The `alpha_underflow` case drives
`ALPHA = Π(−E/B)` to **1.86e-38** over 29 layers — right at float32's smallest
normal (1.18e-38). It costs nothing. `ALPHA` weights contributions to `GAMA`
that are already negligible beside `B(1)`: measured `GAMA` is 2.246, and the
float32 solve agrees with the Fortran to 0.79 ULPs, no worse than any other
case. Guarding against the underflow would be guarding against the arithmetic
working correctly, so the port does not.

Observed `min|ALPHA|` across the cases: 3.5e-1 (`minimal_cbl`), 2.4e-3
(`shallow_cbl`), 2.7e-8 (`whole_column`), 7.6e-15 (`deep_cbl`), 1.9e-38
(`alpha_underflow`).

**Matrix conventions were confirmed by residual, not read.** Both files document
their storage in a comment block, but the mapping from `L`/`D`/`U` and
`A`/`B`/`E` onto matrix positions is exactly the sort of thing a comment gets
subtly wrong. Each was checked by assembling the dense matrix from the inferred
convention, multiplying the compiled Fortran's own solution back, and requiring
the residual to vanish — 1.2e-07 for `tri`, 1.1e-07 for `matrix1`, both float32
noise. The `asymmetric` golden exists so that a transposed sub/super-diagonal
cannot pass, and a test demonstrates that swapping them does change the answer.
