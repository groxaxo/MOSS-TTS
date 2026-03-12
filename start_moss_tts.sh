#!/usr/bin/env bash
# MOSS-TTS Realtime OpenAI-compatible server launcher
# Optimized for RTX 3060 (12 GB) — codec loaded in float16, torch.compile + SDPA
#
# Usage:
#   ./start_moss_tts.sh                  # defaults below
#   MOSS_TTS_GPU=GPU-xxx ./start_moss_tts.sh
#   MOSS_TTS_PORT=8013 ./start_moss_tts.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-moss-tts-ampere}"
CONDA_BASE="${CONDA_BASE:-/home/op/miniconda3}"

# GPU: default to the RTX 3060 with most free VRAM
MOSS_TTS_GPU="${MOSS_TTS_GPU:-GPU-cbfc8a5f-0df1-ca71-f704-0d09a707d2ac}"
MOSS_TTS_HOST="${MOSS_TTS_HOST:-0.0.0.0}"
MOSS_TTS_PORT="${MOSS_TTS_PORT:-8012}"
MOSS_TTS_DEVICE="${MOSS_TTS_DEVICE:-cuda:0}"
MOSS_TTS_WARMUP_ON_START="${MOSS_TTS_WARMUP_ON_START:-true}"
MOSS_TTS_ATTN_IMPLEMENTATION="${MOSS_TTS_ATTN_IMPLEMENTATION:-sdpa}"
MOSS_TTS_COMPILE_BACKBONE="${MOSS_TTS_COMPILE_BACKBONE:-true}"

LOG_FILE="${MOSS_TTS_LOG:-/tmp/moss_tts_server.log}"

echo "[moss-tts] Starting on GPU=${MOSS_TTS_GPU} host=${MOSS_TTS_HOST}:${MOSS_TTS_PORT}"
echo "[moss-tts] Log: ${LOG_FILE}"

# Activate conda and launch
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${MOSS_TTS_GPU}"
export MOSS_TTS_HOST
export MOSS_TTS_PORT
export MOSS_TTS_DEVICE
export MOSS_TTS_WARMUP_ON_START
export MOSS_TTS_ATTN_IMPLEMENTATION
export MOSS_TTS_COMPILE_BACKBONE

exec moss-tts-realtime-openai 2>&1 | tee -a "${LOG_FILE}"
