# A2 figures — vertical advection

Regenerate with:

```bash
python scripts/make_a2_figures.py
```

Vertical advection is structurally unlike the horizontal operator, and these
figures exist mostly to show how.

---

## `flux_diagnosis.png`

**There is no vertical wind field to read.** The flux is *diagnosed* from how far
the transported ρ·J has drifted from the meteorology, so that advecting ρ·J
closes the gap. This figure walks the chain: mismatch → face flux → face
velocity → Courant number.

Two things are worth noticing in the flux panel:

- **The ground is pinned closed** by construction (`zadvppmwrf.F:341` sets
  `FLX(1) = 0`).
- **The model top closes itself.** Unrolling the recurrence gives
  `FLX(top) = DRJ · (1 − Σds)`, and in sigma coordinates `Σds = 1` exactly, so
  the top-face flux vanishes identically. Measured, it sits ~15 orders of
  magnitude below the interior fluxes. Column mass is therefore exactly
  conserved — a property the horizontal operator does *not* have, since its
  domain edges are genuinely open.

---

## `vertical_transport.png`

What one sync step does, and what repeating it converges to.

A layer-versus-time Hovmöller was the obvious choice here and it was the wrong
one — the first draft of this figure was nearly blank. Because the flux is
diagnosed *from the mismatch*, once a step closes the gap there is nothing left
to drive transport. In a full run horizontal advection reopens the gap every
sync step; in isolation the operator is a **relaxation**, and this shows it
relaxing.

The gap does not reach zero. It stops at **exactly the column-mean mismatch**
(0.451 → 0.047, against a column-mean offset of 0.0470), and that floor is
structural rather than a convergence failure: the flux conserves column mass, so
it can redistribute a mismatch through the column but can never remove a uniform
offset. Closing that is the coupling step's job, not advection's.

Column mass holds to 7e-16 throughout.

---

## `substepping.png`

When the driver splits the sync step, and by how much.

The Courant number here is not set by a wind speed — it follows from the size of
the density mismatch, since that is what the diagnosed flux has to close. The
first two panels show mismatch driving Courant, and Courant driving the number
of sub-steps.

The third panel is the one that matters for the port: a grid where neighbouring
columns need anywhere from one to several sub-steps, all advancing **together**
in a single fixed-count loop with finished columns masked off. That is what lets
ragged per-column work run as one batched kernel.

---

## No animation here

Unlike the [A0 rotation and swirl](../a0/README.md), there is nothing temporal to
show. The operator is a single relaxation per sync step, and its interesting
behaviour is structural — where the flux vanishes, what the residual floor is,
how the sub-stepping distributes. Static panels carry that better than a loop.
