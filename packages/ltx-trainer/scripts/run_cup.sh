#!/usr/bin/env bash
set -euo pipefail

# Example inference run for a "cup" prompt.
# Usage:
#   scripts/run_cup.sh /path/to/model.safetensors /path/to/gemma-model [output.mp4]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint.safetensors> <text_encoder_path> [output.mp4]" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
TEXT_ENCODER_PATH="$2"
OUTPUT_PATH="${3:-outputs/cup_example.mp4}"

uv run python scripts/inference.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --text-encoder-path "$TEXT_ENCODER_PATH" \
  --prompt "A ceramic cup on a wooden table, soft morning light, cinematic camera movement" \
  --skip-audio \
  --output "$OUTPUT_PATH"
