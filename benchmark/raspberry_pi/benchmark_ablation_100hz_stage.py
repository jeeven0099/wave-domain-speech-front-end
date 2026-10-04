#!/usr/bin/env python3
"""Raspberry Pi held-out test for the 100 Hz envelope-rate ablation.

A thin wrapper around benchmark_single_model_stage.py: the load / frontend /
full stages, timing, MAC profiling, and accuracy code are exactly the ones
used for the 400 Hz fabricated_400hz_baselines_post_envelope rows. Only three
things change:

  * envelopes are read from envelopes_100Hz_ablation_capture.npz
    (10 ms AC-RMS window, 10 ms hop) instead of envelopes_400Hz_capture.npz;
  * checkpoints come from models/ablation_envelope_100hz/<model>/;
  * results are reported under the system name
    fabricated_100hz_ablation_post_envelope.

Digit segmentation is identical to the 400 Hz path (event start/end +/- the
checkpoint's context_ms, 5 ms causal availability delay), evaluated on the
100 Hz timestamps.

Example (from the bundle directory):
    python3 benchmark_ablation_100hz_stage.py --model tc_resnet8 --stage full \\
        --runs 2000 --warmup 0
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_single_model_stage as stage  # noqa: E402

bench = stage.bench

ENVELOPE_RATE_HZ = 100.0
RMS_WINDOW_MS = 10.0
ENVELOPE_NPZ = "envelopes_100Hz_ablation_capture.npz"
ENVELOPE_METADATA = "envelopes_100Hz_ablation_metadata.json"
SYSTEM = "fabricated_100hz_ablation_post_envelope"
MODEL_DIR = Path(
    os.environ.get("ABLATION_100HZ_MODEL_DIR", str(HERE / "models" / "ablation_envelope_100hz"))
)
RESULTS_DIR = HERE / "results" / "ablation_100hz"


def checkpoint_path(model_name: str) -> Path:
    return MODEL_DIR / model_name / f"best_ablation_100hz_5d_{model_name}_classifier.pt"


def capture_directories(capture_root: Path, stimuli_root: Path) -> list[Path]:
    if capture_root.name == "test":
        capture_root = capture_root.parent
    expected = bench._expected_test_recording_ids(stimuli_root)
    by_recording: dict[int, Path] = {}
    for metadata_path in sorted((capture_root / "test").rglob("metadata.json")):
        run_dir = metadata_path.parent
        if not all(
            (run_dir / name).is_file()
            for name in ("aligned_events.csv", ENVELOPE_METADATA, ENVELOPE_NPZ)
        ):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        recording_id = int(metadata.get("manifest_recording_id", -1))
        if recording_id not in expected:
            continue
        if recording_id in by_recording:
            raise RuntimeError(f"Duplicate test capture for recording {recording_id}")
        by_recording[recording_id] = run_dir
    missing = sorted(expected - set(by_recording))
    if missing:
        raise RuntimeError(
            f"Missing 100 Hz envelopes for {len(missing)} test captures; first IDs: "
            f"{missing[:12]} -- run ablation_envelope_rate.py --stage export-pi on the PC"
        )
    return [by_recording[index] for index in sorted(by_recording)]


def build_provenance(
    capture_root: Path, test_split_path: Path, context_ms: float
) -> list[dict[str, Any]]:
    stimuli_root = test_split_path.parent
    permitted_speakers = bench._speaker_set_from_json(stimuli_root / "protocol.json")
    if permitted_speakers is None:
        raise ValueError("The fabricated capture loader requires stimuli/protocol.json")

    provenance: list[dict[str, Any]] = []
    for run_dir in capture_directories(capture_root, stimuli_root):
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
        speaker = int(metadata["dataset_speaker_id"])
        if speaker not in permitted_speakers:
            raise RuntimeError(f"{run_dir.name}: speaker {speaker:02d} is not in the test split")

        envelope_metadata = json.loads((run_dir / ENVELOPE_METADATA).read_text(encoding="ascii"))
        if not math.isclose(float(envelope_metadata["envelope_rate_hz"]), ENVELOPE_RATE_HZ):
            raise RuntimeError(f"{run_dir.name}: saved envelope rate is not 100 Hz")
        if not math.isclose(float(envelope_metadata["rms_window_duration_ms"]), RMS_WINDOW_MS):
            raise RuntimeError(f"{run_dir.name}: RMS window is not 10 ms")

        npz_path = run_dir / ENVELOPE_NPZ
        with np.load(npz_path, allow_pickle=False) as archive:
            center_times = np.asarray(archive["time_s"], dtype=np.float64)
            if not math.isclose(float(archive["sample_rate_hz"]), ENVELOPE_RATE_HZ):
                raise RuntimeError(f"{run_dir.name}: envelope npz sample_rate_hz is not 100 Hz")
        availability_times = center_times + RMS_WINDOW_MS / 2000.0

        with (run_dir / "aligned_events.csv").open("r", encoding="utf-8", newline="") as handle:
            events = [row for row in csv.DictReader(handle) if str(row["event_type"]) == "letter"]
        for event in sorted(events, key=lambda row: float(row["start_s"])):
            label = int(event["label"])
            start_s = max(
                float(availability_times[0]), float(event["start_s"]) - context_ms / 1_000.0
            )
            end_s = min(
                float(availability_times[-1]) + 1.0 / ENVELOPE_RATE_HZ,
                float(event["end_s"]) + context_ms / 1_000.0,
            )
            first = int(np.searchsorted(availability_times, start_s, side="left"))
            last = int(np.searchsorted(availability_times, end_s, side="left"))
            if last <= first:
                raise RuntimeError(f"{run_dir.name}: empty envelope segment for digit {label}")
            repetition = int(float(event["source_repetition"]))
            provenance.append(
                {
                    "sample_id": bench.canonical_sample_id(speaker, label, repetition),
                    "npz_path": npz_path,
                    "first": first,
                    "last": last,
                    "label": label,
                }
            )

    provenance.sort(key=lambda item: item["sample_id"])
    if len(provenance) != bench.EXPECTED_TEST_EXAMPLES:
        raise RuntimeError(
            f"Expected {bench.EXPECTED_TEST_EXAMPLES} test examples, found {len(provenance)}"
        )
    return provenance


def load_example(entry: dict[str, Any]) -> bench.FabricatedExample:
    with np.load(entry["npz_path"], allow_pickle=False) as archive:
        envelope_values = np.asarray(archive["envelope_v"], dtype=np.float32)
    features = np.ascontiguousarray(
        envelope_values[entry["first"] : entry["last"]], dtype=np.float32
    )
    return bench.FabricatedExample(
        sample_id=entry["sample_id"],
        features=features,
        label=entry["label"],
        audio_duration_seconds=(entry["last"] - entry["first"]) / ENVELOPE_RATE_HZ,
    )


_original_load_model = bench.load_model


def load_model(system: str, path: Path, model_name: str | None = None):
    """Reuse the 400 Hz post-envelope loader (same architectures, same
    architecture-ID check), then relabel the system as the 100 Hz ablation."""
    checkpoint = bench._torch_load(path)
    rate = checkpoint.get("envelope_rate_hz")
    if rate is None or not math.isclose(float(rate), ENVELOPE_RATE_HZ):
        raise RuntimeError(f"{path} is not a 100 Hz ablation checkpoint (envelope_rate_hz={rate})")
    loaded = _original_load_model("fabricated_400hz_baselines_post_envelope", path, model_name)
    base = loaded.name.split("/", 1)[1]
    loaded.name = f"{SYSTEM}/{base}"
    loaded.representation = (
        "precomputed five-channel 100 Hz AC-RMS envelopes (10 ms window, 10 ms hop; "
        f"envelope-rate ablation) -> {base} (masked-mean endpoint)"
    )
    loaded.source_sample_rate_hz = int(ENVELOPE_RATE_HZ)
    loaded.feature_rate_hz = int(ENVELOPE_RATE_HZ)
    loaded.checkpoint_metadata["envelope_rate_hz"] = float(rate)
    return loaded


# Route benchmark_single_model_stage's fabricated path through the 100 Hz
# files and checkpoints. Everything else (timing loop, preprocessing,
# MAC profiling, accuracy, output files) is used unchanged.
stage.build_fabricated_provenance = build_provenance
stage.load_fabricated_example = load_example
stage.checkpoint_for = lambda _representation, model_name, *_dirs: checkpoint_path(model_name)
bench.load_model = load_model


def main() -> None:
    argv = sys.argv[1:]
    if any(arg == "--representation" or arg.startswith("--representation=") for arg in argv):
        raise SystemExit("--representation is fixed to the fabricated 100 Hz ablation")
    if not any(arg == "--results-dir" or arg.startswith("--results-dir=") for arg in argv):
        argv += ["--results-dir", str(RESULTS_DIR)]
    args = stage.parse_args(["--representation", "fabricated", *argv])
    stage.validate_paths(args)
    summary, rows = stage.run_stage(args)
    summary["details"]["envelope_rate_hz"] = ENVELOPE_RATE_HZ
    summary["details"]["envelope_file"] = ENVELOPE_NPZ
    stage.write_outputs(args.results_dir, summary, rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
