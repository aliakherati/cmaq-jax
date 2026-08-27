# B0 — Foundation

Parent: [`../PLAN-hdiff.md`](../PLAN-hdiff.md)

Vendor the Fortran, extend the harness, and pin the constants. No JAX yet.

**Gate: passed.** `deform.F` and `hcdiff3d.F` compile unmodified and run; 8
golden cases committed, `--check` clean. The analytic cases land exactly:
`du/dy = 3/dx2` gives 2.500000e-04, `du/dx = 5/dx1` gives 4.166666e-04, uniform
wind gives zero deformation and the diffusivity floor 105.2632
(`KHA*KHMIN/(KHA+KHMIN)`), and the saturating case reaches 222.2170 against
`KHA = 222.2222`.

`rho_j.F` is not harnessed. It reads `DENSA_J` and `JACOBM` and reassembles
rho*J, which the port takes directly from `io_mcip`; there is no numerics in it
to validate.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **B0.1** ✅ | Vendor `hdiff.F`, `hcdiff3d.F`, `deform.F`, `rho_j.F` + `PROVENANCE.md` rows | sha256 recorded; files byte-identical to `origin/5.5+` | `git diff --stat` empty vs upstream |
| **B0.2** ✅ | `config.py`: `HDiffConstants` — `KH`, `KHMIN`, `DXB`, `ALP`, `CFC` | Every constant carries its `hcdiff3d.F` line | `pytest tests/unit/test_config.py` |
| **B0.3** ✅ | Harness `harness_deform.f90`, `harness_hcdiff3d.f90` | `deform.F`/`hcdiff3d.F` compile unmodified; one call per process | `make -C reference` |
| **B0.4** ✅ | Goldens for deformation, diffusivity and face coefficients | One process per `(NCOLS, NROWS, NLAYS)`; `--check` clean | `python scripts/generate_goldens.py --check` |

## Notes

**The stub surface grows.** Unlike `hppm.F`/`vppm.F`, these reach the I/O API:
`deform.F` calls `interpolate_var('UHAT_JD', ...)` and `rho_j.F` reads
`DENSA_J`/`JACOBM`. The `interpolate_var` stub written for A3.4's `hcontvel`
harness already covers this — extend its variable table rather than writing a
second one.

**`MSFD2` comes from `GRID_DOT_2D`**, which the advection harness never opened.
The stub returns 1.0 unless a case says otherwise, and the `map_factor` case
uses a non-unit field so the multiplication is actually exercised — it is ~1 on
the benchmark Lambert grid, so a dropped multiplication would pass every other
case.

**Boundary reads had to be implemented in the stub.** `deform.F` takes the
non-WINDOW path and reads `DENSA_J`'s perimeter ring through the `'b'` form of
`interpolate_var`. The advection harnesses never reached it — `hcontvel.F`
returns early on the C-staggered path — so the stub raised on it. It now stores
perimeter fields registered by `cio_put_bndy`, in CMAQ's
South/East/North/West order (`deform.F:264-292`), and still fails loudly on an
unregistered name: a zero halo density would make `deform.F` divide by zero at
the domain edge, producing infinities rather than an obviously blank result.
