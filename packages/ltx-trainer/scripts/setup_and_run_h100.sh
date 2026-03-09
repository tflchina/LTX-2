#!/usr/bin/env bash
set -euo pipefail

# Setup + launch helper for H100 GPU inference.
#
# Usage:
#   scripts/setup_and_run_h100.sh <checkpoint.safetensors> <text_encoder_path> <prompt> [output.mp4] [extra inference args...]
#
# Example:
#   scripts/setup_and_run_h100.sh \
#     /path/to/model.safetensors \
#     /path/to/gemma \
#     "A cinematic drone shot over snowy mountains" \
#     outputs/h100_inference.mp4 \
#     --num-inference-steps 40 --guidance-scale 5.0

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <checkpoint.safetensors> <text_encoder_path> <prompt> [output.mp4] [extra inference args...]" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
TEXT_ENCODER_PATH="$2"
PROMPT="$3"
OUTPUT_PATH="${4:-outputs/h100_inference.mp4}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ ! -d "$TEXT_ENCODER_PATH" ]]; then
  echo "Text encoder directory not found: $TEXT_ENCODER_PATH" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required but was not found." >&2
  exit 1
fi

GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$GPU_INFO" != *"H100"* ]]; then
  echo "Warning: detected GPU '$GPU_INFO' (expected an H100)." >&2
fi

# Install project dependencies (idempotent).
uv sync

# H100-friendly defaults.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_FREEZING=1

# Optional extra args begin at $5.
EXTRA_ARGS=()
if [[ $# -gt 4 ]]; then
  EXTRA_ARGS=("${@:5}")
fi

uv run python scripts/inference.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --text-encoder-path "$TEXT_ENCODER_PATH" \
  --prompt "$PROMPT" \
  --output "$OUTPUT_PATH" \
  "${EXTRA_ARGS[@]}"
