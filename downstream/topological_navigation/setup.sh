#!/usr/bin/env bash
# Set up the downstream object-goal navigation experiment.
#
# Clones the upstream RoboHop/ObjectReact code at the pinned commit and applies
# our overlay: the segment-matcher adapters, the localizer dispatch, the run
# configs and the analysis scripts.
#
#   bash downstream/topological_navigation/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/object-rel-nav"
UPSTREAM_URL="https://github.com/oravus/object-rel-nav.git"
UPSTREAM_PIN="6da34c30871e9b8a8cdc46c8607fe3683763020e"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mXX\033[0m  %s\n' "$*" >&2; exit 1; }

if [ -d "$DEST/.git" ]; then
  info "object-rel-nav already present at $DEST"
else
  info "cloning object-rel-nav at $UPSTREAM_PIN"
  git clone --quiet "$UPSTREAM_URL" "$DEST" || die "could not clone $UPSTREAM_URL"
  git -C "$DEST" checkout --quiet --detach "$UPSTREAM_PIN" \
    || die "pin $UPSTREAM_PIN not found upstream"
fi

HAVE="$(git -C "$DEST" rev-parse HEAD)"
[ "$HAVE" = "$UPSTREAM_PIN" ] || die "\
$DEST is at $HAVE, but the overlay is built against $UPSTREAM_PIN.
Re-checkout that commit, or remove the directory and re-run this script."

info "applying the overlay"
if git -C "$DEST" apply --check "$HERE/overlay.patch" 2>/dev/null; then
  git -C "$DEST" apply "$HERE/overlay.patch"
  info "overlay applied: 48 new files, 7 upstream files patched"
elif git -C "$DEST" apply --reverse --check "$HERE/overlay.patch" 2>/dev/null; then
  info "overlay already applied"
else
  die "the overlay does not apply to $DEST; is the working tree dirty?"
fi

cat <<'NOTE'

Next steps, which this script deliberately does not do for you:

  1. Create the conda environment. habitat-sim cannot be installed with uv:

       conda create -n nav python=3.10
       conda activate nav
       conda install habitat-sim=0.3.3 headless -c conda-forge -c aihabitat
       pip install -r object-rel-nav/requirements.txt
       pip install -e ../..            # the muviseg package

  2. Download HM3D and the InstanceImageNav episodes into object-rel-nav/data/.
     See object-rel-nav/Readme.md and this directory's README.md.

  3. Fetch the trained MuViSeg heads:

       python ../../scripts/download_checkpoints.py

NOTE
info "done"
