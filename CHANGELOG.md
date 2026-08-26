# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A0.1** — Repository scaffold: package layout, `pyproject.toml` (ruff +
  mypy strict + pytest), CI workflow, MIT license, `CLAUDE.md` working
  agreement.
- **A0.1** — CMAQ Fortran reference vendored verbatim from branch `5.5+`
  @ `b8e4303` under `reference/fortran/`, with per-file sha256 in
  `reference/PROVENANCE.md` and a `scripts/vendor_reference.sh` to re-vendor.
- **A0.1** — Plan hierarchy: `docs/ULTRAPLAN.md` (the CMAQ→JAX arc),
  `docs/plans/PLAN-advection.md`, and phase subplans `A0`–`A3`.
