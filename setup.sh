#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
ROOT="$PWD"
UV_BIN="${UV_BIN:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-2}"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS=120 BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE
mkdir -p vendor logs
if [[ ! -d vendor/FlashVSR ]]; then
    git clone https://github.com/OpenImagingLab/FlashVSR.git vendor/FlashVSR
    git -C vendor/FlashVSR checkout cf910c61a60733e610e9c6e8b607f80c3a6c202b
fi
if [[ ! -d vendor/Block-Sparse-Attention ]]; then
    git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git vendor/Block-Sparse-Attention
    git -C vendor/Block-Sparse-Attention checkout 49d6c39e4dc0303442cda3bb758b3925d4399c49
    git -C vendor/Block-Sparse-Attention submodule update --init --recursive
fi
[[ "$(git -C vendor/FlashVSR rev-parse HEAD)" == cf910c61a60733e610e9c6e8b607f80c3a6c202b ]]
[[ -z "$(git -C vendor/FlashVSR status --porcelain --untracked-files=no)" ]]
[[ "$(git -C vendor/Block-Sparse-Attention rev-parse HEAD)" == 49d6c39e4dc0303442cda3bb758b3925d4399c49 ]]
if git -C vendor/Block-Sparse-Attention apply --check "$ROOT/provenance/attention-torch214-cxx20.patch" 2>/dev/null; then
    git -C vendor/Block-Sparse-Attention apply "$ROOT/provenance/attention-torch214-cxx20.patch"
else
    git -C vendor/Block-Sparse-Attention apply --reverse --check "$ROOT/provenance/attention-torch214-cxx20.patch"
fi
if [[ ! -x .venv/bin/python ]]; then
    "$UV_BIN" venv --python 3.12 .venv
fi
"$UV_BIN" pip install --python .venv/bin/python torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
"$UV_BIN" pip install --python .venv/bin/python -r requirements-runtime.txt
"$UV_BIN" pip install --python .venv/bin/python -r requirements-decoder.txt -r requirements-analysis.txt
"$UV_BIN" pip install --python .venv/bin/python --no-deps --no-build-isolation ./vendor/Block-Sparse-Attention
.venv/bin/python fetch_assets.py --references
