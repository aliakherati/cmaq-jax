#!/usr/bin/env bash
# Re-vendor the CMAQ Fortran reference into reference/fortran/.
# Usage: scripts/vendor_reference.sh /path/to/CMAQ [commit-ish]
#
# The vendored files are the validation reference and are never edited by hand.
# After running this, update the commit/sha table in reference/PROVENANCE.md.
set -euo pipefail

CMAQ_REPO="${1:?usage: $0 /path/to/CMAQ [commit-ish]}"
REF="${2:-origin/5.5+}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/reference/fortran"

FILES=(
  hadv/ppm/hppm.F
  hadv/ppm/hadvppm.F
  hadv/ppm/x_ppm.F
  hadv/ppm/y_ppm.F
  hadv/ppm/hcontvel.F
  hadv/ppm/rdbcon.F
  hadv/ppm/advbc_map.F
  hadv/ppm/zfdbc.f
  vadv/local_cons/vppm.F
  vadv/wrf_cons/zadvppmwrf.F
  vadv/local_cons/zadvyppm.F
  driver/advstep.F
  hadv/ppm/xy_budget.F
)

# The serial stencil-exchange layer and the CMAQ include files named by the
# INCLUDE SUBST_* cpp macros.
STENEX_DEST="$(cd "$(dirname "$0")/.." && pwd)/reference/stenex"
INCLUDE_DEST="$(cd "$(dirname "$0")/.." && pwd)/reference/include"
INCLUDES=(
  ICL/fixed/const/CONST.EXT
  ICL/fixed/filenames/FILES_CTM.EXT
  ICL/fixed/mpi/PE_COMM.EXT
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  git -C "$CMAQ_REPO" show "$REF:CCTM/src/$f" > "$DEST/$(basename "$f")"
done

mkdir -p "$STENEX_DEST" "$INCLUDE_DEST"
for f in $(git -C "$CMAQ_REPO" ls-tree --name-only "$REF" CCTM/src/STENEX/noop/); do
  git -C "$CMAQ_REPO" show "$REF:$f" > "$STENEX_DEST/$(basename "$f")"
done
for f in "${INCLUDES[@]}"; do
  git -C "$CMAQ_REPO" show "$REF:CCTM/src/$f" > "$INCLUDE_DEST/$(basename "$f")"
done

echo "vendored $(git -C "$CMAQ_REPO" rev-parse "$REF") -> $DEST"
( cd "$DEST" && shasum -a 256 ./*.F ./*.f )
