#!/usr/bin/env bash
# Clone the pinned third-party dependencies and deploy our patches.
#
# Deliberately a script, not git submodules: the SegMASt3R remote needs an
# explicit pin and a protocol choice, and the two patch files below have to be
# copied into the checkout afterwards -- a submodule update would drop them.
#
# Usage:  bash setup_third_party.sh [--force]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$HERE/third_party"
FORCE="${1:-}"

SEGMAST3R_URL="https://github.com/SegMASt3R/segmast3r.git"
SEGMAST3R_PIN="0977d0661ab51b6ebbd086aff045bf6e98a6ab76"
VGGT_URL="https://github.com/facebookresearch/vggt.git"
VGGT_PIN="44b3afbd1869d8bde4894dd8ea1e293112dd5eba"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mXX\033[0m  %s\n' "$*" >&2; exit 1; }

clone_pinned() {  # $1=url $2=pin $3=dest
  local url="$1" pin="$2" dest="$3" name; name="$(basename "$dest")"
  if [ -d "$dest/.git" ]; then
    local have; have="$(git -C "$dest" rev-parse HEAD)"
    if [ "$have" = "$pin" ]; then info "$name already at $pin"; return 0; fi
    warn "$name is at $have, expected $pin"
    [ "$FORCE" = "--force" ] || die "refusing to move $name; re-run with --force, or remove $dest"
    info "fetching $pin into $name"
    git -C "$dest" fetch --quiet origin "$pin" || git -C "$dest" fetch --quiet origin
    git -C "$dest" checkout --quiet --detach "$pin"
  else
    info "cloning $name"
    git clone --quiet "$url" "$dest" || die "could not clone $url
  If this repository is not public, request access from its authors; the
  MASt3R-backbone models (ARCH=sinkhorn/lightglue/lightglue_v2) cannot be
  built without it. The VGGT-backbone models do not need it."
    git -C "$dest" checkout --quiet --detach "$pin" || die "pin $pin not found in $url"
  fi
  info "$name pinned at $(git -C "$dest" rev-parse --short HEAD)"
}

mkdir -p "$TP"
clone_pinned "$SEGMAST3R_URL" "$SEGMAST3R_PIN" "$TP/segmast3r"
clone_pinned "$VGGT_URL"      "$VGGT_PIN"      "$TP/vggt"

# --- deploy our patches into the segmast3r checkout -------------------------
DEST="$TP/segmast3r/src/models/mast3r_segfeat"
[ -d "$DEST" ] || die "expected $DEST in the segmast3r checkout; is the pin correct?"
for f in segment_attention.py double_softmax_matcher.py; do
  src="$TP/segmast3r_patches/$f"
  if cmp -s "$src" "$DEST/$f"; then
    info "patch $f already deployed"
  else
    cp -v "$src" "$DEST/$f"
  fi
done

# --- report on the weights we cannot download for you ----------------------
echo
info "checkpoint status"
check() {  # $1=path $2=what $3=where
  if [ -f "$1" ]; then printf '  present  %s\n' "$2"
  else printf '  MISSING  %-34s get it from: %s\n' "$2" "$3"; fi
}
check "$TP/vggt_weights.pt" "VGGT-1B backbone (5.0 GB)" \
      "huggingface.co/facebook/VGGT-1B (or let the code auto-download)"
check "$TP/segmast3r/mast3r_src/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
      "MASt3R backbone (2.8 GB)" "github.com/naver/mast3r#checkpoints"
check "$HERE/FastSAM-x.pt" "FastSAM-x (145 MB, ablations + demo only)" \
      "github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints"
echo
info "trained MuViSeg heads: see docs/checkpoints.md"
info "done"
