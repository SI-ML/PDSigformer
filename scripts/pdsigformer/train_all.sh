#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 34 配置 × 3 seeds；R4 同时对应主实验 PDSigformer 与消融第 4 行。
python scripts/test_release.py training --gpus "${1:-0}" --epochs 100 --seeds 42 43 44 --output "${2:-outputs/training_all}" "${@:3}"
