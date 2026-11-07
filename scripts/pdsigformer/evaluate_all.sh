#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/test_release.py inference --gpus "${1:-0}" --output "${2:-outputs/evaluation_all}" "${@:3}"
