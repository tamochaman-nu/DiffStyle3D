#!/usr/bin/env bash
# Entry point of the `download-sd` service.
# Downloads the Stable Diffusion weights used by train-style into a local
# diffusers directory. train-style must load SD from a local directory:
# the bundled stable_diffusion/pipeline_utils.py fails when it has to download
# a Hugging Face repo id itself.
# Only the fp32 safetensors of the components DiffStyle3D loads are fetched
# (safety_checker is disabled in guidance/AttnGuidance.py).
set -euo pipefail

echo "[download-sd] ${SD_REPO_ID:?} -> ${SD_LOCAL_DIR:?}"
exec huggingface-cli download "${SD_REPO_ID}" \
    --local-dir "${SD_LOCAL_DIR}" \
    --include \
        "model_index.json" \
        "scheduler/*" \
        "tokenizer/*" \
        "feature_extractor/*" \
        "text_encoder/config.json" \
        "text_encoder/model.safetensors" \
        "unet/config.json" \
        "unet/diffusion_pytorch_model.safetensors" \
        "vae/config.json" \
        "vae/diffusion_pytorch_model.safetensors" \
    "$@"
