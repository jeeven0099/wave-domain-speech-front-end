#!/usr/bin/env python3
"""Raspberry Pi held-out test for the three CNN baselines on the single
broadband 400 Hz RMS envelope (spectral-channel ablation).

A thin wrapper around benchmark_audiomnist_raspberry_pi.run_worker: the
digital_envelope_400hz system is used unchanged (48 kHz WAV loading, the
timed preprocess_400hz_envelope frontend, timing loop, MAC profiling, RAM and
accuracy). The only change is that the three one-channel CNN constructors
from broadband_cnn_models.py are registered next to the six existing
architectures, so results land in results/.partials/ next to the other six as
digital_envelope_400hz__<model>_summary.json.

Example (from the bundle directory):
    python3 benchmark_broadband_cnn_stage.py --model tiny_conv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_audiomnist_raspberry_pi as bench  # noqa: E402
from broadband_cnn_models import BROADBAND_CNN_FACTORIES  # noqa: E402

NAMES = tuple(BROADBAND_CNN_FACTORIES)
bench.DIGITAL_400_FACTORIES.update(BROADBAND_CNN_FACTORIES)
bench.DIGITAL_MODEL_NAMES = tuple(bench.DIGITAL_MODEL_NAMES) + NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=6_000)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    options = parser.parse_args()
    args = bench.parse_args([
        "--worker-system", "digital_envelope_400hz",
        "--worker-model", options.model,
        "--test-split-path", str(HERE / "data" / "fabricated" / "stimuli" / "protocol.json"),
        "--raw-audiomnist-path", str(HERE / "data" / "AudioMNIST"),
        "--model-400-digital-dir", str(HERE / "models" / "digital_envelope_400hz"),
        "--results-dir", str(options.results_dir),
        "--warmup", str(options.warmup),
        "--runs", str(options.runs),
    ])
    bench.run_worker(args)


if __name__ == "__main__":
    main()
