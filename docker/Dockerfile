# DiffStyle3D runtime image (Python 3.10 / CUDA 12.1 / PyTorch 2.1.0).
#
# The 3DGS CUDA extensions (diff-gaussian-rasterization, simple-knn, fused-ssim)
# are built from a pinned gaussian-splatting checkout cloned into
# /opt/gaussian-splatting, which is also used by the `train-3dgs` service to
# reconstruct the scene checkpoint that DiffStyle3D starts from.
#
# No GPU is visible during `docker build`, so the target architectures must be
# given explicitly via TORCH_CUDA_ARCH_LIST (set from docker-compose.yml).
# MAX_JOBS caps ninja's parallelism: each nvcc job can take several GB of RAM.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.9"
ARG MAX_JOBS=2
ARG GS_REPO=https://github.com/graphdeco-inria/gaussian-splatting.git
ARG GS_COMMIT=54c035f7834b564019656c3e3fcc3646292f727d

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ninja-build build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# setuptools<70: torch 2.1's cpp_extension still does `from pkg_resources import packaging`.
RUN python -m pip install --upgrade pip "setuptools<70" wheel

# joblib: required by gaussian-splatting's train.py (train-3dgs service), not listed in requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt joblib

# Only the submodules needed for training are fetched (SIBR_viewers is skipped).
RUN git clone ${GS_REPO} /opt/gaussian-splatting \
    && cd /opt/gaussian-splatting \
    && git checkout ${GS_COMMIT} \
    && git submodule update --init --recursive \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        submodules/fused-ssim

# --no-build-isolation: their setup.py imports torch at build time without declaring it.
RUN python -m pip install --no-build-isolation \
        /opt/gaussian-splatting/submodules/diff-gaussian-rasterization \
        /opt/gaussian-splatting/submodules/simple-knn \
        /opt/gaussian-splatting/submodules/fused-ssim

# The project is also bind-mounted over /workspace by docker-compose.yml so that
# local edits take effect without a rebuild; the copy keeps the image usable alone.
WORKDIR /workspace
COPY . /workspace
