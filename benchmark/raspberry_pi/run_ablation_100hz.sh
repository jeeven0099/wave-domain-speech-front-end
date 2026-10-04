#!/usr/bin/env bash
# Held-out test for the 100 Hz envelope-rate ablation: --stage full
# (load + frontend + NN + accuracy/params/size/MACs) for all nine architectures,
# with the same RUNS/WARMUP as run_full_selected.sh used for the 400 Hz rows.
# Results: results/ablation_100hz/fabricated_100hz_ablation_post_envelope__<model>__full_*.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

RUNS="${RUNS:-2000}"
WARMUP="${WARMUP:-0}"

# Same interpreter as the 400 Hz runs: the bundle's .venv (README setup).
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$ROOT/.venv/bin/python"; else PYTHON_BIN=python3; fi
fi
if ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
  echo "no torch in $PYTHON_BIN -- activate the bundle venv or set PYTHON_BIN=/path/to/python"
  exit 1
fi
echo "python: $PYTHON_BIN"

MODELS=(bc_resnet_1 bc_resnet_2 edgespeechnet_a matchboxnet_3x1x64 mlperf_tiny_dscnn tc_resnet8 cnn_one_fstride4 tiny_conv cnn_trad_fpool3)

echo "--- preflight: ${#MODELS[@]} checkpoints and 100 Hz test envelopes ---"
missing=0
for model in "${MODELS[@]}"; do
  path="$ROOT/models/ablation_envelope_100hz/$model/best_ablation_100hz_5d_${model}_classifier.pt"
  if [[ -f "$path" ]]; then echo "  ok      $model"; else echo "  MISSING $path"; missing=1; fi
done
count=$(find "$ROOT/data/fabricated/captures/test" -name envelopes_100Hz_ablation_capture.npz | wc -l)
echo "  100 Hz test envelope files: $count"
if [[ "$missing" -ne 0 || "$count" -eq 0 ]]; then
  echo "aborting: copy the exported checkpoints and envelopes from the PC first"
  exit 1
fi

for model in "${MODELS[@]}"; do
  echo "=== fabricated 100 Hz ablation / $model / full ==="
  "$PYTHON_BIN" benchmark_ablation_100hz_stage.py \
    --model "$model" --stage full \
    --runs "$RUNS" --warmup "$WARMUP" \
    --test-split-path "$ROOT/data/fabricated/stimuli/protocol.json" \
    --fabricated-feature-path "$ROOT/data/fabricated/captures"
done

"$PYTHON_BIN" - "$ROOT/results/ablation_100hz" <<'SUMMARY'
import json, sys
from pathlib import Path
rows = []
for path in sorted(Path(sys.argv[1]).glob("*__full_summary.json")):
    s = json.loads(path.read_text())
    rows.append((s["model"], s["accuracy"], s["nn_macs"], s["params"], s["nn_median_ms"]))
print(f"\n{'model':<20} {'acc %':>7} {'MACs':>11} {'params':>7} {'nn ms':>7}")
for model, acc, macs, params, nn_ms in rows:
    print(f"{model:<20} {acc:7.2f} {macs:11,d} {params:7,d} {nn_ms:7.2f}")
SUMMARY
echo "all runs complete"
