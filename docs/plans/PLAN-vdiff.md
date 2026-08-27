# cmaq-jax — vertical diffusion (ACM2)

Parent: [`../ULTRAPLAN.md`](../ULTRAPLAN.md) · Owns chunk IDs `C0`–`C3` ·
Depends on `A` and `B` for the harness, config and halo machinery.

## What this is

Operator 3. `sciproc.F` runs it **first** in the science sequence, before
coupling and transport:

```
VDIFF → COUPLE → HADV → ZADV → HDIFF → DECOUPLE → PHOT → CLDPROC → CHEM → AERO
```

The Asymmetric Convective Model version 2 (Pleim 2007). Two build options,
`acm2_m3dry` and `acm2_stage`, selected by `DepMod` — **`m3dry` is the default**
(`bldit_cctm.csh:113`), and that is what this ports.

**Branch note, and this one differs from `A` and `B`:** `vdiff` is *not*
identical between `main` and `5.5+`. Three files changed —
`acm2_stage/{opddep,vdiffacmx,vdiffproc}.F`. All are in `acm2_stage`, which we
are not porting; `acm2_m3dry` is unchanged. Vendored from `5.5+` regardless.

## Scope: transport, not deposition

`vdiffacmx.F` is 1202 lines and only a minority of it is vertical diffusion. It
interleaves the ACM2 solve with dry-deposition bookkeeping, emission injection,
heterogeneous HONO chemistry, aerosol sedimentation, BDSNP soil-NO coupling, and
ISAM/DDM branches.

The cut, and the reason for it: **deposition velocities and emission fluxes are
inputs, not something this operator computes** — exactly as meteorology is an
input to advection. What CMAQ calls "vertical diffusion" is the operator that
mixes a column given a surface flux; the modules that decide what that surface
flux is (`depv/m3dry`, `depv/stage`) are a separate concern and a separate port.

| In scope | Out of scope |
|---|---|
| `eddyx.F` — eddy diffusivity `Kz` | `depv/*` — deposition velocity calculation |
| `tri.F` — tridiagonal (Thomas) solver | `aero_depv.F`, `aero_sedv.F`, `SEDIMENTATION.F` |
| `matrix1.F` — the ACM1 solver | HET HONO special cases (chemistry, not transport) |
| ACM2 assembly + two-stage solve | BDSNP soil-NO coupling |
| Sub-time-stepping and its limit | ISAM, DDM-3D, `VDIFF_DIAG` |
| Surface exchange, given `depv` and emissions | `opddep.F` — deposition output files |

The surface-exchange interface is small and well defined
(`vdiffacmx.F:693-696`): given a deposition velocity `depv` and an emission flux
`pldv`, the surface layer relaxes exponentially toward the equilibrium
`pldv/depv`, and the accumulated dry deposition is a diagnostic output.

## The numerics

**Crank–Nicolson**, `THETA = 0.5` (`vdiffacmx.F:94`) — the file supports pure
explicit and pure implicit through the same constant, which is worth carrying
rather than hard-coding.

**1. Eddy diffusivity** (`eddyx.F:104-234`). A real parameterization, not a
constant: Monin–Obukhov similarity inside the PBL, Richardson-number mixing
above it, and a moist correction where cloud water is present. Pointwise and
branch-heavy, so it maps cleanly onto `jnp.where` with no solve.

**2. Sub-step limit** (`vdiffacmx.F:457-516`):

```
DTLIM  = min(DTSEC, 0.75/(SEDDY·DZHI·DZFI), 0.75·DTACM)
NLP    = int(DTSEC/DTLIM + 0.99)          # ceiling
DTS    = DTSEC/NLP
```

Note this is an **accuracy** limit, not a stability one — Crank–Nicolson is
unconditionally stable — which is a real difference from `hdiff`, where the
analogous constant sits past the stability boundary (see
[`PLAN-hdiff.md`](PLAN-hdiff.md)).

**3. The ACM2 split.** Each sub-step is two stages:

*Convective stage*, only where `CONVCT` (`vdiffacmx.F:855-925`). The non-local
plume: mass rises from the surface layer directly to every layer in the
convective boundary layer, and returns by layer-to-layer subsidence. The matrix
is tridiagonal **plus a full first column**, which is why it needs `MATRIX1`
rather than `TRI`:

```
MBAR       = MEDDY · FNL                      non-local mixing rate
MBARKS(L)  = MBAR                             up from layer 1
MDWN(L)    = MBAR·(PBL − ZF(L−1))·DZHI(L)     subsidence
```

*Local stage*, always (`vdiffacmx.F:955-1023`). Ordinary vertical diffusion,
solved with `TRI`.

**4. The two solvers.** `TRI` is the Thomas algorithm; `MATRIX1` eliminates the
first column via a running product

```
ALPHA(L) = Π_{k=2..L} (−E(k)/B(k))
X(1)     = [D(1) + Σ ALPHA(L)·D(L)] / [B(1) + Σ ALPHA(L)·A(L)]
```

then back-substitutes. Both share one matrix across all species, differing only
in the right-hand side — a factorisation done once serves the whole column.

## What will be hard, and where

Not the physics. Three structural things:

**`LCBL` is per column.** The convective stage runs over layers `1..LCBL`, the
top of the convective boundary layer, which is data-dependent. Like `A2`'s
sub-stepping this becomes a masked fixed-extent loop rather than a ragged one —
and masking a *linear solve* is more delicate than masking an explicit update,
because a masked-out row must not make the matrix singular.

**`ALPHA` is a cumulative product that can underflow.** `−E(L)/B(L)` is
repeatedly multiplied over the CBL depth. In float32 over 35 layers this can
reach zero, and `GAMA` is a denominator. Worth measuring rather than assuming,
and worth a golden case with a deep CBL.

**The sequential recurrences.** Both solvers are first-order recurrences over
layers. With `NLAYS ≈ 35` a `lax.scan` is fine on CPU; whether it is fine on GPU
is a question the benchmark should answer rather than the plan.

## Execution order

| Phase | Subplan | Gate |
|---|---|---|
| **C0** | [`subplans/C0-foundation.md`](subplans/C0-foundation.md) | Vendored Fortran compiles; goldens for both solvers and `eddyx` |
| **C1** | [`subplans/C1-solvers.md`](subplans/C1-solvers.md) | `tri`, `matrix1`, `eddy_diffusivity` match |
| **C2** | [`subplans/C2-acm2.md`](subplans/C2-acm2.md) | The two-stage driver matches, including a convective column |
| **C3** | [`subplans/C3-integrate.md`](subplans/C3-integrate.md) | Wired into `transport_step`; differentiable; benchmarked |

## Deliberate deviations

As for `A` and `B`, plus:

- Deposition velocities and emissions are **inputs**, per the scope table.
- The HET HONO branches are omitted: they modify `depv` for three named species
  using chemistry state, which does not belong in a transport operator.
- `acm2_stage` is not ported. It is the non-default `DepMod`, and it is the only
  part of `vdiff` that `5.5+` changes.
