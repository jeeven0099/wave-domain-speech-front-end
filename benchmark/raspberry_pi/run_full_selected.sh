#!/usr/bin/env bash
# Runs --stage full (load + frontend + NN + accuracy/params/size) for the
# remaining fabricated/fabricated_400hz_baselines_post_envelope models:
#   - the 4 resonator baselines not already covered by tc_resnet8/bc_resnet_1
#   - the 3 literature-minimal CNN baselines (same system, same checkpoint
#     loader -- cnn_baseline_models.py was trained on the identical 400 Hz
#     envelope input as the resonator baselines)
# digital_16k is already done; this script touches nothing else.
set -euo pipefail
cd "$(dirname "$0")"

RUNS=2000
WARMUP=0

FABRICATED_MODELS=(bc_resnet_2 edgespeechnet_a matchboxnet_3x1x64 mlperf_tiny_dscnn cnn_one_fstride4 tiny_conv cnn_trad_fpool3)

# Default the CNN baseline dir to the copy shipped in this bundle, but let an
# existing export win. ':=' only assigns when unset/empty.
: "${MODEL_FABRICATED_CNN_BASELINES_DIR:=$PWD/models/fabricated_cnn_baselines}"
export MODEL_FABRICATED_CNN_BASELINES_DIR

# Preflight: resolve every checkpoint through the benchmark script's own
# checkpoint_for() -- not a copy of its path rules -- so a missing or
# misplaced file fails now instead of an hour into the sweep.
echo "--- preflight: resolving ${#FABRICATED_MODELS[@]} checkpoints ---"
python3 - "$PWD" "${FABRICATED_MODELS[@]}" <<'PREFLIGHT'
import sys

# Read from stdin, so __file__ does not exist here; the bundle dir is argv[1]
# (the script cd's to its own directory above).
sys.path.insert(0, sys.argv[1])
import benchmark_single_model_stage as stage

missing = []
for name in sys.argv[2:]:
    path = stage.checkpoint_for(
        "fabricated",
        name,
        stage.MODEL_16K_DIR,
        stage.MODEL_FABRICATED_BASELINES_DIR,
        stage.MODEL_FABRICATED_CNN_BASELINES_DIR,
    )
    ok = path.is_file()
    print(f"  {'ok     ' if ok else 'MISSING'} {name:<20} {path}")
    if not ok:
        missing.append(name)

if missing:
    print(f"\naborting: {len(missing)} checkpoint(s) not found: {', '.join(missing)}")
    sys.exit(1)
print("preflight ok\n")
PREFLIGHT

for model in "${FABRICATED_MODELS[@]}"; do
  echo "=== fabricated / $model / full ==="
  python3 benchmark_single_model_stage.py \
    --representation fabricated --model "$model" --stage full \
    --runs "$RUNS" --warmup "$WARMUP"
done

echo "all runs complete"
