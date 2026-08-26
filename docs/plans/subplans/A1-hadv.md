# A1 — Horizontal advection (HADV)

Parent: [`../PLAN-advection.md`](../PLAN-advection.md) · Depends on A0

Wrap the uniform-spacing PPM kernel in CMAQ's horizontal machinery: contravariant
velocity, boundary conditions, halo, per-layer sub-stepping, and the X-Y/Y-X
alternation.

**Gate:** 2-D rotating-cone advection matches the analytic solution to the
expected order, and a uniform mixing ratio survives a divergent wind unchanged.

| Chunk | Deliverable | Success criterion | Verify |
|---|---|---|---|
| **A1.1** ✅ | `bc.py`: `zfdbc` (`zfdbc.f`), branchless | Matches the Fortran function on a sign/magnitude sweep incl. `\|v1\| < 1e-3` | `pytest tests/unit/test_bc.py` |
| **A1.2** ✅ | `bc.py`: halo fill — width-3 ghost region, BCON inflow vs. ZFDBC outflow per face (`x_ppm.F:418-441`) | Halo is a first-class array region, filled locally; swap point for `shard_map` documented | `pytest tests/unit/test_halo.py` |
| **A1.3** ✅ | `velocity.py`: contravariant velocity (`hcontvel.F:329-351`) | `UHAT = UHAT_JD / mean(DENSA_J)` on the staggered face; matches a hand-worked case | `pytest tests/unit/test_velocity.py` |
| **A1.4** ✅ | `hadv.py`: `sweep(c, vel, axis)` — axis-generic whole-array PPM sweep | X and Y sweeps are the same function; one fused kernel over rows × layers × species | `pytest tests/unit/test_sweep.py` |
| **A1.5** ✅ | `hadv.py`: `hadv()` driver — layer grouping by `ASTEP`, sub-step loop, X-Y/Y-X alternation (`hadvppm.F:197-257`) | Alternation parity matches Fortran `XYFIRST` semantics | `pytest tests/unit/test_hadv.py` |
| **A1.6** ✅ | 2-D property tests: rotating cone, horizontal constancy preservation, mass conservation with closed boundaries | Cone shape preserved within PPM's expected diffusion; constancy exact to round-off | `pytest tests/properties -k hadv` |
| **A1.7** ✅ | Figures `docs/figures/a1/` | Rotating cone at t=0/½/1 revolution; constancy-error field | `python scripts/make_a1_figures.py` |

## Notes

**Layer sub-stepping is ragged.** `ASTEP(LVL)` differs by layer
(`hadvppm.F:199`). `ASTEP` is host-side data taking few distinct values, so
**statically partition layers into groups sharing a sub-step count** and run one
`lax.fori_loop` per group. No masking, no `while_loop`.

**Keep the halo.** The width-3 ghost region (`SWP = 3`) is retained as an
explicit array region even though it is filled locally. Multi-GPU later swaps
the local fill for a collective permute under `shard_map`; the kernels don't
change. Deleting the halo now would mean redesigning then.

**Boundary-cell fluxes are donor-cell**, not parabolic (`hppm.F:422-439`) —
easy to miss, and it changes the answer at the domain edge.
