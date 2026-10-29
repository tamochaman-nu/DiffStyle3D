#!/usr/bin/env bash
# Entry point of the `render-orbit` service.
# Renders an orbit video around the subject of ORBIT_MODEL_PATH
# (see docker/render_orbit.py). Extra arguments are appended as-is.
set -euo pipefail

is_true() { [[ "${1,,}" == "true" || "$1" == "1" ]]; }

if [[ ! -f "${ORBIT_MODEL_PATH:?}/cameras.json" ]]; then
    echo "[render-orbit] Model not found: ${ORBIT_MODEL_PATH}" >&2
    echo "[render-orbit] Run train-style (or train-3dgs) first, or check SCENE / EXP_NAME in .env." >&2
    exit 1
fi

args=(
    --model_path "${ORBIT_MODEL_PATH}"
    --iteration "${ORBIT_ITERATION:?}"
    --sh_degree "${SH_DEGREE:?}"
    --n_frames "${N_FRAMES:?}"
    --fps "${FPS:?}"
    --span_scale "${SPAN_SCALE:?}"
    --center "${CENTER:?}"
    --max_scale "${MAX_SCALE:?}"
)
is_true "${PINGPONG:-true}" || args+=(--no-pingpong)
is_true "${WHITE_BACKGROUND:-false}" && args+=(--white_background)

cd /workspace
echo "[render-orbit] python docker/render_orbit.py ${args[*]} $*"
exec python docker/render_orbit.py "${args[@]}" "$@"
