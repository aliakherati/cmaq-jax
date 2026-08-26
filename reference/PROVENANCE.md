# Provenance of the vendored Fortran reference

These files are copied verbatim from CMAQ. They are **never edited** — they are
the reference the JAX port is validated against. `hppm.F` and `vppm.F` are
compiled by `scripts/generate_goldens.py`; the rest are here so the port can be
read against its source without a second checkout.

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
