# B0 — Foundation

Parent: [`../PLAN-hdiff.md`](../PLAN-hdiff.md)

Vendor the Fortran, extend the harness, and pin the constants. No JAX yet.

**Gate:** `deform.F`, `hcdiff3d.F` and `rho_j.F` compile unmodified and emit
goldens; `generate_goldens.py --check` is clean.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **B0.1** | Vendor `hdiff.F`, `hcdiff3d.F`, `deform.F`, `rho_j.F` + `PROVENANCE.md` rows | sha256 recorded; files byte-identical to `origin/5.5+` | `git diff --stat` empty vs upstream |
| **B0.2** | `config.py`: `HDiffConstants` — `KH`, `KHMIN`, `DXB`, `ALP`, `CFC` | Every constant carries its `hcdiff3d.F` line | `pytest tests/unit/test_config.py` |
| **B0.3** | Harness `harness_deform.f90`, `harness_hcdiff3d.f90` | Zeroes `DEFORM3D` before the call (see the plan's note) | `make -C reference` |
| **B0.4** | Goldens for deformation, diffusivity and face coefficients | One process per `(NCOLS, NROWS, NLAYS)`; `--check` clean | `python scripts/generate_goldens.py --check` |

## Notes

**The stub surface grows.** Unlike `hppm.F`/`vppm.F`, these reach the I/O API:
`deform.F` calls `interpolate_var('UHAT_JD', ...)` and `rho_j.F` reads
`DENSA_J`/`JACOBM`. The `interpolate_var` stub written for A3.4's `hcontvel`
harness already covers this — extend its variable table rather than writing a
second one.

**`MSFD2` comes from `GRID_DOT_2D`**, which the advection harness never opened.
The stub returns 1.0 unless a case says otherwise, and one golden case uses a
non-unit field so the multiplication is actually exercised.
