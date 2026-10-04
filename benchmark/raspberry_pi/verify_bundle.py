#!/usr/bin/env python3
"""Fast preflight check for the self-contained Raspberry Pi benchmark bundle."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_SPEAKERS = {5, 13, 18, 22, 26, 32, 33, 42, 45, 51, 52, 60}
REQUIRED_CAPTURE_FILES = (
    "raw_capture.npz",
    "metadata.json",
    "aligned_events.csv",
    "envelopes_400Hz_annotated.csv",
    "envelopes_400Hz_metadata.json",
)
DIGITAL_MODELS = (
    "bc_resnet_1",
    "bc_resnet_2",
    "edgespeechnet_a",
    "matchboxnet_3x1x64",
    "mlperf_tiny_dscnn",
    "tc_resnet8",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    errors: list[str] = []
    required = [
        ROOT / "benchmark_audiomnist_raspberry_pi.py",
        ROOT / "data/fabricated/stimuli/protocol.json",
        ROOT / "data/fabricated/stimuli/audiomnist_test_manifest.csv",
        ROOT / "models/digital_16k/compact_train_normalizer.npz",
        ROOT
        / "models/fabricated_400hz/best_envelope_400hz_5d_pooled_tcn_classifier.pt",
    ]
    for model in DIGITAL_MODELS:
        required.append(ROOT / "models/digital_16k" / f"{model}_best.pt")
        required.append(
            ROOT
            / "models/digital_envelope_400hz"
            / f"{model}_1d"
            / "best_validation_checkpoint.pt"
        )
        required.append(
            ROOT
            / "models/fabricated_400hz_5d_baseline_models"
            / model
            / f"best_fabricated_400hz_5d_{model}_classifier.pt"
        )
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        errors.append("missing required file(s): " + ", ".join(map(str, missing)))

    protocol = json.loads(
        (ROOT / "data/fabricated/stimuli/protocol.json").read_text(encoding="utf-8")
    )
    speakers = {int(value) for value in protocol["speaker_splits"]["test"]}
    if speakers != TEST_SPEAKERS:
        errors.append(f"unexpected test speakers: {sorted(speakers)}")

    audio_root = ROOT / "data/AudioMNIST/data"
    wavs = [
        path
        for speaker in sorted(TEST_SPEAKERS)
        for path in (audio_root / f"{speaker:02d}").glob("*.wav")
    ]
    if len(wavs) != 6_000:
        errors.append(f"expected 6,000 held-out WAV files, found {len(wavs):,}")

    manifest_path = ROOT / "data/fabricated/stimuli/audiomnist_test_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        expected_recordings = {int(row["recording_id"]) for row in csv.DictReader(handle)}

    capture_root = ROOT / "data/fabricated/captures/test"
    captures: dict[int, Path] = {}
    for metadata_path in capture_root.rglob("metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        recording_id = int(metadata.get("manifest_recording_id", -1))
        if recording_id in expected_recordings:
            captures[recording_id] = metadata_path.parent
    absent = sorted(expected_recordings - captures.keys())
    if absent:
        errors.append(f"missing fabricated recording IDs: {absent}")
    for recording_id, directory in captures.items():
        missing_names = [name for name in REQUIRED_CAPTURE_FILES if not (directory / name).is_file()]
        if missing_names:
            errors.append(f"recording {recording_id} is missing: {', '.join(missing_names)}")

    if errors:
        fail("\nERROR: ".join(errors))

    try:
        import benchmark_audiomnist_raspberry_pi as benchmark

        for model in DIGITAL_MODELS:
            benchmark.load_model(
                "digital_16k",
                ROOT / "models/digital_16k" / f"{model}_best.pt",
                model,
            )
            benchmark.load_model(
                "digital_envelope_400hz",
                ROOT
                / "models/digital_envelope_400hz"
                / f"{model}_1d"
                / "best_validation_checkpoint.pt",
                model,
            )
            benchmark.load_model(
                "fabricated_400hz_baselines",
                ROOT
                / "models/fabricated_400hz_5d_baseline_models"
                / model
                / f"best_fabricated_400hz_5d_{model}_classifier.pt",
                model,
            )
        benchmark.load_model(
            "fabricated_400hz",
            ROOT
            / "models/fabricated_400hz/best_envelope_400hz_5d_pooled_tcn_classifier.pt",
        )
    except Exception as error:
        fail(f"checkpoint contract validation failed: {error}")

    print(
        f"Bundle OK: {len(wavs):,} held-out WAVs, "
        f"{len(expected_recordings)} fabricated captures, 12 digital checkpoints, "
        f"1 fabricated checkpoint, and {len(DIGITAL_MODELS)} fabricated-baseline "
        "checkpoints."
    )


if __name__ == "__main__":
    main()
