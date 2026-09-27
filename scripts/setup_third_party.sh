#!/usr/bin/env bash
# Fetch the external sources that are not installable from PyPI.
#
#     bash scripts/setup_third_party.sh verl     # reinforcement learning
#     bash scripts/setup_third_party.sh pseudo   # cognitive maps from images
#     bash scripts/setup_third_party.sh all
#
# Neither is needed to prepare data, run inference, or evaluate.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WHAT="${1:-}"
case "$WHAT" in
  verl|pseudo|all) ;;
  *) echo "usage: setup_third_party.sh <verl|pseudo|all>"; exit 1 ;;
esac

VERL_TAG="${VERL_TAG:-v0.6.1}"

setup_verl() {
  echo "== veRL $VERL_TAG (policy optimisation) =="
  # Pinned to a tag: the trainer's configuration keys move between releases.
  if [[ ! -d third_party/verl ]]; then
    git clone --depth 1 --branch "$VERL_TAG" https://github.com/volcengine/verl third_party/verl
  fi
  # Without --no-deps it would repin torch and vLLM.
  pip install -e third_party/verl --no-deps
}

setup_pseudo() {
  mkdir -p modules

  echo "== VGGT (3D geometry) =="
  if [[ ! -d modules/vggt ]]; then
    git clone --depth 1 https://github.com/facebookresearch/vggt.git modules/vggt
  fi
  pip install -e modules/vggt --no-deps

  if [[ ! -f modules/VGGT-1B/model.pt ]]; then
    echo "-- fetching the VGGT-1B weights, about 4.8 GB"
    mkdir -p modules/VGGT-1B
    curl -fL -o modules/VGGT-1B/model.pt \
      https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
  fi

  echo "== SAM 3 (promptable segmentation) =="
  if [[ ! -d modules/sam3 ]]; then
    git clone --depth 1 https://github.com/facebookresearch/sam3.git modules/sam3
  fi
  pip install -e modules/sam3 --no-deps

  if [[ ! -d modules/sam3/checkpoints ]]; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
      huggingface-cli download facebook/sam3 --local-dir modules/sam3/checkpoints
    else
      echo "-- SAM 3's weights are licence-gated. Accept the licence at"
      echo "   https://huggingface.co/facebook/sam3, then set HF_TOKEN and rerun,"
      echo "   or leave --sam3-checkpoint empty to fetch them at run time."
    fi
  fi
}

[[ "$WHAT" == "verl"   || "$WHAT" == "all" ]] && setup_verl
[[ "$WHAT" == "pseudo" || "$WHAT" == "all" ]] && setup_pseudo

echo
echo "✅ Done."
