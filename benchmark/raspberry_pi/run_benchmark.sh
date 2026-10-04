#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_CORE="${CPU_CORE:-3}"

"$PYTHON_BIN" verify_bundle.py

mkdir -p results
taskset -c "$CPU_CORE" "$PYTHON_BIN" benchmark_audiomnist_raspberry_pi.py \
  --system all \
  --test-split-path "$ROOT/data/fabricated/stimuli/protocol.json" \
  --raw-audiomnist-path "$ROOT/data/AudioMNIST" \
  --fabricated-feature-path "$ROOT/data/fabricated/captures" \
  --model-16k-dir "$ROOT/models/digital_16k" \
  --model-400-digital-dir "$ROOT/models/digital_envelope_400hz" \
  --model-fabricated-path "$ROOT/models/fabricated_400hz/best_envelope_400hz_5d_pooled_tcn_classifier.pt" \
  --model-fabricated-baselines-dir "$ROOT/models/fabricated_400hz_5d_baseline_models" \
  --results-dir "$ROOT/results" \
  --warmup 50 \
  --runs 6000
