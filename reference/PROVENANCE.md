# Provenance of the vendored Fortran reference

These files are copied verbatim from CMAQ. They are **never edited** — they are
the reference the JAX port is validated against. `hppm.F` and `vppm.F` are
compiled by `scripts/generate_goldens.py`, as are `deform.F`, `hcdiff3d.F`
and `hdiff.F` for the horizontal-diffusion port; the rest are here so the port
can be read against its source without a second checkout.

The `hdiff/multiscale/*` files were vendored at the same commit as the advection
files, and are identical on `main` and `5.5+`.

The `vdiff/acm2_m3dry/*` files come from the same commit. **Unlike advection and
horizontal diffusion, `vdiff` is not identical between `main` and `5.5+`** —
`acm2_stage/{opddep,vdiffacmx,vdiffproc}.F` differ. All three are in
`acm2_stage`, which is the non-default `DepMod` and is not ported;
`acm2_m3dry`, the default (`bldit_cctm.csh:113`), is unchanged between the
branches.

| | |
|---|---|
| Upstream | `git@github.com:aliakherati/CMAQ.git` (fork of `USEPA/CMAQ`) |
| Branch | `5.5+` (the bugfix branch for CMAQ v5.5) |
| Commit | `b8e4303c069c3310550c3ba35754592b9c380ffc` |
| Date | 2026-07-09 |
| Subject | Merge pull request #261 from tnskipper/bugfix/stage_hg_bidi |

Source path prefix: `CCTM/src/`

| Vendored as | Upstream path | sha256 (first 16) |
|---|---|---|
| `hppm.F` | `hadv/ppm/hppm.F` | `71997028be7a1d51` |
| `hadvppm.F` | `hadv/ppm/hadvppm.F` | `bd6d7700caefcae4` |
| `x_ppm.F` | `hadv/ppm/x_ppm.F` | `4ebf0c505cbe4bc4` |
| `y_ppm.F` | `hadv/ppm/y_ppm.F` | `93c5f6f4e5052495` |
| `hcontvel.F` | `hadv/ppm/hcontvel.F` | `bece69f79bb29490` |
| `rdbcon.F` | `hadv/ppm/rdbcon.F` | `0be6bc78a6202399` |
| `advbc_map.F` | `hadv/ppm/advbc_map.F` | `1a5183975bf6d2f7` |
| `zfdbc.f` | `hadv/ppm/zfdbc.f` | `3e88eaee060c8b85` |
| `vppm.F` | `vadv/local_cons/vppm.F` | `606570ed134b7260` |
| `zadvppmwrf.F` | `vadv/wrf_cons/zadvppmwrf.F` | `b0f656667286184b` |
| `zadvyppm.F` | `vadv/local_cons/zadvyppm.F` | `e62f8ab4a2e00822` |
| `advstep.F` | `driver/advstep.F` | `e0bf8ca9479a0b49` |
| `xy_budget.F` | `hadv/ppm/xy_budget.F` | `9f07de250262e4d2` |
| `hdiff.F` | `hdiff/multiscale/hdiff.F` | `b4650bd43d725216` |
| `hcdiff3d.F` | `hdiff/multiscale/hcdiff3d.F` | `e30ba3c021cdf72d` |
| `deform.F` | `hdiff/multiscale/deform.F` | `e1f69a1220e5ad90` |
| `rho_j.F` | `hdiff/multiscale/rho_j.F` | `c48c2b0f3d561ef0` |
| `tri.F` | `vdiff/acm2_m3dry/tri.F` | `f08404b5b4361493` |
| `matrix1.F` | `vdiff/acm2_m3dry/matrix1.F` | `5aba51abbc71ba24` |
| `eddyx.F` | `vdiff/acm2_m3dry/eddyx.F` | `fc9776c4e2683205` |
| `vdiffacmx.F` | `vdiff/acm2_m3dry/vdiffacmx.F` | `9abcb684e65e1a21` |
| `conv_cgrid.F` | `vdiff/acm2_m3dry/conv_cgrid.F` | `7b981d7ba9073ade` |

## Also vendored

The serial stencil-exchange layer. CMAQ ships a no-op implementation for
non-MPI builds, and the Makefile points the `SUBST_*` cpp macros at it exactly
as `bldit_cctm.csh` does for a serial build -- so the halo-exchange calls inside
`x_ppm.F` and `hcontvel.F` resolve to the real upstream no-ops rather than
anything written here.

| Vendored as | Upstream path | sha256 (first 16) |
|---|---|---|
| `stenex/noop_comm_module.f` | `STENEX/noop/noop_comm_module.f` | `8c2b0820cf43a9ad` |
| `stenex/noop_data_copy_module.f` | `STENEX/noop/noop_data_copy_module.f` | `d6bd5ede5715acf2` |
| `stenex/noop_gather_module.f` | `STENEX/noop/noop_gather_module.f` | `4166e90e9978a330` |
| `stenex/noop_global_max_module.f` | `STENEX/noop/noop_global_max_module.f` | `678e2ab19a3bae39` |
| `stenex/noop_global_min_module.f` | `STENEX/noop/noop_global_min_module.f` | `8a7806225c00d4f4` |
| `stenex/noop_global_sum_module.f` | `STENEX/noop/noop_global_sum_module.f` | `be13dae612ef495d` |
| `stenex/noop_init_module.f` | `STENEX/noop/noop_init_module.f` | `a1ebe01f2b0b3d33` |
| `stenex/noop_modules.f` | `STENEX/noop/noop_modules.f` | `b87f58fd334e6730` |
| `stenex/noop_slice_module.f` | `STENEX/noop/noop_slice_module.f` | `17b5cc9d6b0174d0` |
| `stenex/noop_term_module.f` | `STENEX/noop/noop_term_module.f` | `90f74b8956edba79` |
| `stenex/noop_util_module.f` | `STENEX/noop/noop_util_module.f` | `8ef30b9a6180fb75` |

CMAQ include files, named by the `INCLUDE SUBST_*` cpp macros
(`bldit_cctm.csh:506-508`).

| Vendored as | Upstream path | sha256 (first 16) |
|---|---|---|
| `include/CONST.EXT` | `ICL/fixed/const/CONST.EXT` | `8a4de000b45b198a` |
| `include/FILES_CTM.EXT` | `ICL/fixed/filenames/FILES_CTM.EXT` | `a31a9e9091431b46` |
| `include/PE_COMM.EXT` | `ICL/fixed/mpi/PE_COMM.EXT` | `0da645006411c20e` |

## Notes

- `CCTM/src/vadv/wrf_cons/vppm.F` is a **symlink** to
  `../local_cons/vppm.F` upstream. Both vertical advection options share one
  kernel; only the `FLX`/`VEL` diagnosis differs (`zadvppmwrf.F` vs
  `zadvyppm.F`). We vendor the real file once.
- The advection sources are **identical** between `main` and `5.5+` at this
  commit: `git diff main 5.5+ -- CCTM/src/{hadv,vadv,couple,grid,spcs}` is
  empty. `5.5+` is used for provenance correctness, not because it changes the
  numerics.
- `zadvyppm.F` (the legacy `local_cons` driver) is vendored for comparison only.
  The port targets `zadvppmwrf.F`, which is the CMAQ v5.5 default
  (`ModAdv = wrf_cons`).

## Re-vendoring

```sh
scripts/vendor_reference.sh /path/to/CMAQ <commit-ish>
```
