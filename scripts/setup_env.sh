#!/usr/bin/env bash
# GPU environment setup for optimized-llama (NVIDIA A40 / Ampere, CUDA 12.8)
set -euo pipefail

echo "==> Creating venv"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip wheel setuptools

echo "==> Installing core inference stack"
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install vllm transformers accelerate sentencepiece datasets

echo "==> Installing quantization tooling (INT8 / SmoothQuant)"
pip install bitsandbytes llm-compressor

echo "==> Installing benchmarking tooling"
pip install numpy pandas matplotlib

echo "==> Done. Activate with: source .venv/bin/activate"
