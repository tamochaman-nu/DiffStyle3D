#!/usr/bin/env bash
# Entry point of the `prepare-scene` service.
# Resizes every view of RAW_SCENE to one resolution and writes it as SCENE
# (see docker/prepare_scene.py). Extra arguments are appended as-is.
set -euo pipefail

if [[ -z "${RAW_SCENE:-}" ]]; then
    echo "[prepare-scene] Set RAW_SCENE in .env (source scene, relative to DATA_DIR)." >&2
    exit 1
fi

cd /workspace
exec python docker/prepare_scene.py \
    --src "/data/${RAW_SCENE}" \
    --dst "${SOURCE_PATH:?}" \
    --width "${TARGET_WIDTH:?}" \
    --height "${TARGET_HEIGHT:?}" \
    "$@"
