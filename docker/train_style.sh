#!/usr/bin/env bash
# Entry point of the `train-style` service.
# Converts the environment variables defined in docker-compose.yml into
# train_style.py arguments. Extra arguments given to
# `docker compose run --rm train-style ...` are appended as-is.
set -euo pipefail

is_true() { [[ "${1,,}" == "true" || "$1" == "1" ]]; }

START_CHECKPOINT="${GS_MODEL_PATH:?}/chkpnt${GS_CHECKPOINT_ITERATION:?}.pth"

if [[ ! -f "${START_CHECKPOINT}" ]]; then
    echo "[train-style] 3DGS checkpoint not found: ${START_CHECKPOINT}" >&2
    echo "[train-style] Run 'docker compose run --rm train-3dgs' first, or check SCENE / GS_OUTPUT_DIR in .env." >&2
    exit 1
fi
if [[ "${SD_MODEL_KEY:?}" == /* && ! -f "${SD_MODEL_KEY}/model_index.json" ]]; then
    echo "[train-style] Stable Diffusion weights not found: ${SD_MODEL_KEY}" >&2
    echo "[train-style] Run 'docker compose run --rm download-sd' first, or check MODELS_DIR in .env." >&2
    exit 1
fi
if [[ ! -f "${STYLE_IMAGE_PATH:?}" ]]; then
    echo "[train-style] Style image not found: ${STYLE_IMAGE_PATH}" >&2
    echo "[train-style] Check STYLE_IMAGE / STYLE_DIR in .env." >&2
    exit 1
fi

args=(
    --source_path "${SOURCE_PATH:?}"
    --model_path "${MODEL_PATH:?}"
    --start_checkpoint "${START_CHECKPOINT}"
    --style_image_path "${STYLE_IMAGE_PATH}"
    --sd_model_key "${SD_MODEL_KEY:?}"
    --iterations "${ITERATIONS:?}"
    --max_iteration "${MAX_ITERATION:?}"
    --resolution "${RESOLUTION:?}"
    --feature_lr "${FEATURE_LR:?}"
    --batch_size "${BATCH_SIZE:?}"
    --start_layer "${START_LAYER:?}"
    --end_layer "${END_LAYER:?}"
    --content_weight "${CONTENT_WEIGHT:?}"
    --set_timestep "${SET_TIMESTEP:?}"
    --start_reset_iter "${START_RESET_ITER:?}"
    --end_reset_iter "${END_RESET_ITER:?}"
    --thresh "${THRESH:?}"
    --lgt_thresh "${LGT_THRESH:?}"
    --reset_rate "${RESET_RATE:?}"
    --height "${HEIGHT:?}"
    --width "${WIDTH:?}"
    --ip "${VIEWER_IP:?}"
    --port "${VIEWER_PORT:?}"
)
is_true "${FP16:-false}" && args+=(--fp16)
is_true "${ADD_NOISE:-false}" && args+=(--add_noise)
is_true "${DISABLE_VIEWER:-true}" && args+=(--disable_viewer)
is_true "${WHITE_BACKGROUND:-false}" && args+=(--white_background)

cd /workspace
echo "[train-style] python train_style.py ${args[*]} $*"
exec python train_style.py "${args[@]}" "$@"
