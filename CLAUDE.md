# Working agreement — cmaq-jax

## 1. Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

## 3. Surgical changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the request.

## 4. Goal-driven execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, and clarifying questions come before
implementation rather than after mistakes.

---

## 5. Documentation is the key

Documentation is a deliverable, not an afterthought.

- **Plan hierarchy.** `docs/ULTRAPLAN.md` (the whole CMAQ→JAX arc) →
  `docs/plans/PLAN-*.md` (one project) → `docs/plans/subplans/*.md` (one phase,
  holding its chunk table) → chunks (`A0.1`, `A1.4`, …) are the tasks. Every
  chunk carries a success criterion and a verification command.
- **One chunk = one commit.** The commit message names the chunk ID.
- **`README.md` carries the status table**: module | purpose | chunk | status.
  Update it in the same commit that lands the module.
- **Every scientific chunk ships a figure** under `docs/figures/<chunk-id>/`,
  regenerable via `scripts/make_<chunk>_figures.py`, with a one-line caption in
  the README saying what it demonstrates.
- **Every deviation from the Fortran gets written down** in `README.md` with the
  reason. A silent numerical deviation is a bug in the making.
- Docstrings cite the Fortran they port: file and line range
  (e.g. `hppm.F:289-303`) plus the Colella & Woodward equation number.

## 6. Virtual environment is a must

- `uv` manages it. `uv venv && uv pip install -e ".[dev]"`.
- Python pinned in `.python-version` (3.12).
- Never `pip install` into the system interpreter; never assume a package is
  present — add it to `pyproject.toml`.
- Runtime deps stay minimal (`numpy`, `jax`). Anything needed only by tests,
  figures, or scripts goes in an extra (`dev`, `io`).

## 7. Repository hygiene

Things go where they make sense, and the layout is predictable:

```
src/cmaq_jax/     library code, importable, typed (py.typed)
tests/            unit/ properties/ regression/ differentiability/ fixtures/
docs/             ULTRAPLAN.md, plans/, figures/<chunk-id>/
scripts/          one-shot and regeneration scripts, never imported by src/
reference/        vendored CMAQ Fortran — READ ONLY, never edited
data/goldens/     committed Fortran reference outputs
```

- `reference/fortran/` is verbatim upstream. Changing a file there invalidates
  every golden. Re-vendor via `scripts/vendor_reference.sh` and update
  `reference/PROVENANCE.md`.
- No scratch files, notebooks, or `.DS_Store` in commits.
- `ruff check . && mypy --strict src/` passes before every commit.

## 8. Config file to abstract all the constants

No magic numbers in kernels.

- `src/cmaq_jax/config.py` owns every constant: grid dimensions, `dx`, sigma
  layer thicknesses, PPM tolerances, iteration caps, CFL limits, dtype.
- Kernels take a config object; they never read a global or hard-code a bound.
- Each constant records where it came from in the Fortran
  (e.g. `EPSF = 1.0e-3  # vppm.F:145`), so a value can be traced upstream.
- If a constant appears in two places, it belongs in `config.py`.

## 9. Leverage agents, worktrees, loop engineering

- **Agents** for genuinely parallel, independent work — e.g. porting `x_ppm`
  and `vppm` at once, or a broad read across the Fortran. Not for work that
  needs to stay in one head.
- **Worktrees** to keep an experimental port isolated from a known-good tree,
  so the golden suite always has something green to compare against.
- **Loop engineering** — this is what rule 4 buys. Because every chunk has a
  verification command, work can run to completion unattended: write, run the
  gate, fix, repeat. Set the gate before starting, not after.

---

## Project-specific rules

- **The Fortran is the reference, not a suggestion.** When JAX and Fortran
  disagree, the Fortran is right until proven otherwise. Investigate before
  adjusting a tolerance.
- **`hppm.F` and `vppm.F` use `SAVE`d allocatables sized on first call.** The
  golden harness must run **one fresh process per (NI, NSPCS) configuration**,
  or results silently corrupt. This is the easiest way to produce wrong goldens.
- **ρ·J rides along as the last advected species.** That is CMAQ's
  mass-conservation mechanism. Any change that breaks it is wrong even if every
  species test still passes.
- **Stay device-agnostic.** No CUDA-specific code, no hard-coded `jax.devices()`,
  no host callbacks in the hot path. Development is CPU-on-laptop; production is
  GPU, eventually multi-node.
- **Keep the halo abstraction** even though the halo is filled locally today.
  It is what makes `shard_map` a drop-in later.
