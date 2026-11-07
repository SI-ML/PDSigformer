#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
profile="${1:-shimmer_r4}"
seed="${2:-42}"
output="${3:-outputs/eval_${profile}_s${seed}}"
python run_wearable.py eval --profile "$profile" --seed "$seed" --output "$output" "${@:4}"
