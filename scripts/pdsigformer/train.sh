#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
profile="${1:-shimmer_r4}"
seed="${2:-42}"
output="${3:-outputs/train_${profile}_s${seed}}"
python run_wearable.py train --profile "$profile" --seed "$seed" --output "$output" "${@:4}"
