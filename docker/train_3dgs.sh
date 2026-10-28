#!/usr/bin/env bash
# Entry point of the `train-3dgs` service.
# Reconstructs the scene with gaussian-splatting (/opt/gaussian-splatting) and
# saves chkpnt<ITERATIONS>.pth, which train-style uses as --start_checkpoint.
# Extra arguments given to `docker compose run --rm train-3dgs ...` are appended as-is.
set -euo pipefail

is_true() { [[ "${1,,}" == "true" || "$1" == "1" ]]; }

args=(
    --source_path "${SOURCE_PATH:?}"
    --model_path "${MODEL_PATH:?}"
    --iterations "${ITERATIONS:?}"
    --resolution "${RESOLUTION:?}"
    --save_iterations "${ITERATIONS}"
    --checkpoint_iterations "${ITERATIONS}"
    --ip "${VIEWER_IP:?}"
    --port "${VIEWER_PORT:?}"
)
is_true "${EVAL:-false}" && args+=(--eval)
is_true "${DISABLE_VIEWER:-true}" && args+=(--disable_viewer)
is_true "${WHITE_BACKGROUND:-false}" && args+=(--white_background)

cd /opt/gaussian-splatting
echo "[train-3dgs] python train.py ${args[*]} $*"
exec python train.py "${args[@]}" "$@"
