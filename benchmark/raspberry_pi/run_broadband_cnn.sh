#!/usr/bin/env bash
# Held-out test for the three CNN baselines on the single broadband 400 Hz RMS
# envelope, with the same settings as the six digital_envelope_400hz rows
# (run_benchmark.sh: CPU 3, warmup 50, 6000 runs).
# Results: results/.partials/digital_envelope_400hz__<model>_{summary.json,timings.csv}
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
CPU_CORE="${CPU_CORE:-3}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$ROOT/.venv/bin/python"; else PYTHON_BIN=python3; fi
fi
echo "python: $PYTHON_BIN"

MODELS=(tiny_conv cnn_one_fstride4 cnn_trad_fpool3)
for model in "${MODELS[@]}"; do
  path="$ROOT/models/digital_envelope_400hz/${model}_1d/best_validation_checkpoint.pt"
  if [[ ! -f "$path" ]]; then echo "MISSING $path -- copy the exported checkpoints first"; exit 1; fi
done

for model in "${MODELS[@]}"; do
  echo "=== digital_envelope_400hz / $model ==="
  taskset -c "$CPU_CORE" "$PYTHON_BIN" benchmark_broadband_cnn_stage.py --model "$model"
done

"$PYTHON_BIN" - "$ROOT/results/.partials" <<'SUMMARY'
import json, sys
from pathlib import Path
print(f"\n{'system':<44} {'acc %':>7} {'MACs':>10} {'params':>7} {'nn ms':>7}")
for path in sorted(Path(sys.argv[1]).glob("digital_envelope_400hz__*_summary.json")):
    s = json.loads(path.read_text())
    print(f"{s['system']:<44} {s['accuracy']:7.2f} {s['nn_macs']:10,d} {s['params']:7,d} {s['nn_median_ms']:7.2f}")
SUMMARY
echo "all runs complete"
