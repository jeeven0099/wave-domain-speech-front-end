#!/usr/bin/env python3
"""Run exactly one architecture, on one input representation, at one pipeline
stage -- a narrower companion to benchmark_audiomnist_raspberry_pi.py for
interactive Raspberry Pi sessions where sweeping all systems is unwanted.

Representations (exactly two, both required to include tc_resnet8):
  digital_16k - original 48 kHz AudioMNIST WAV, resampled to 16 kHz before the
                timed frontend (log-Mel), matching the frontier digital system.
  fabricated  - precomputed five-channel 400 Hz AC-RMS envelope rows read
                straight from envelopes_400Hz_capture.npz (the hardware-native
                post-envelope path; never recomputed from a raw waveform here).

Stages:
  load     - read each sample fresh from disk only. Nothing else runs.
  frontend - load, then run front-end feature extraction (log-Mel / envelope
             normalization). No NN forward pass.
  full     - load + frontend + NN forward pass. Accuracy, parameter count,
             and fp32 model size are ONLY computed at this stage.

MACs-per-inference is a static property of the model and input shape; it is
measured once via an untimed profiling pass (synthetic input, same shape a
real example would produce) and reported at every stage, regardless of
which stage's timings are being collected.

Edit the paths below, or override with the matching CLI flag / environment
variable of the same name used by benchmark_audiomnist_raspberry_pi.py.

Example:
    python benchmark_single_model_stage.py --representation fabricated \\
        --model tc_resnet8 --stage full --runs 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_audiomnist_raspberry_pi as bench  # noqa: E402


# ---------------------------------------------------------------------------
# User configuration. Edit these paths directly, or override with the
# matching CLI flag / environment variable.
# ---------------------------------------------------------------------------
TEST_SPLIT_PATH = Path(os.environ.get("TEST_SPLIT_PATH", str(bench.TEST_SPLIT_PATH)))
RAW_AUDIOMNIST_PATH = Path(os.environ.get("RAW_AUDIOMNIST_PATH", str(bench.RAW_AUDIOMNIST_PATH)))
FABRICATED_FEATURE_PATH = Path(
    os.environ.get("FABRICATED_FEATURE_PATH", str(bench.FABRICATED_FEATURE_PATH))
)
MODEL_16K_DIR = Path(os.environ.get("MODEL_16K_DIR", str(bench.MODEL_16K_DIR)))
MODEL_FABRICATED_BASELINES_DIR = Path(
    os.environ.get(
        "MODEL_FABRICATED_BASELINES_DIR", str(bench.MODEL_FABRICATED_BASELINES_DIR)
    )
)
# Unlike the other model dirs, this one falls back to the copy shipped next to
# this script rather than a placeholder, so the CNN baselines run without the
# caller having to remember one more export.
MODEL_FABRICATED_CNN_BASELINES_DIR = Path(
    os.environ.get(
        "MODEL_FABRICATED_CNN_BASELINES_DIR",
        str(Path(__file__).resolve().parent / "models" / "fabricated_cnn_baselines"),
    )
)
RESULTS_DIR = Path(os.environ.get("SINGLE_MODEL_RESULTS_DIR", "results/single_model_stage"))

REPRESENTATIONS = ("digital_16k", "fabricated")
# 6 resonator names (includes tc_resnet8) + 3 literature-minimal CNN baselines.
# Both families are loaded under the same "fabricated" representation / same
# fabricated_400hz_baselines_post_envelope system -- the system name tracks
# the input representation, not the architecture family.
MODEL_NAMES = bench.DIGITAL_MODEL_NAMES + bench.CNN_BASELINE_MODEL_NAMES
STAGES = ("load", "frontend", "full")
DEFAULT_RUNS = 200
DEFAULT_WARMUP = 20
DEFAULT_CONTEXT_MS = 50.0  # Overridden by the checkpoint's own context_ms when present.


# ---------------------------------------------------------------------------
# Provenance: cheap per-sample disk-location metadata, built once up front
# (untimed) so the timed "load" stage can re-read each sample fresh from disk
# on every run instead of working from a preloaded in-memory example list.
# ---------------------------------------------------------------------------
def build_digital_16k_provenance(test_split_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    rows = bench._raw_manifest_rows(test_split_path, dataset_root)
    provenance: list[dict[str, Any]] = []
    for row in rows:
        wav_path = bench._resolve_raw_path(row, dataset_root)
        stem_digit, stem_speaker, stem_repetition = (int(value) for value in wav_path.stem.split("_"))
        label = int(row.get("label", row.get("digit", stem_digit)))
        speaker = int(row.get("speaker", row.get("speaker_id", stem_speaker)))
        repetition = int(row.get("repetition", stem_repetition))
        provenance.append(
            {
                "sample_id": bench.canonical_sample_id(speaker, label, repetition),
                "wav_path": wav_path,
                "label": label,
            }
        )
    provenance.sort(key=lambda item: item["sample_id"])
    if len(provenance) != bench.EXPECTED_TEST_EXAMPLES:
        raise RuntimeError(
            f"Expected {bench.EXPECTED_TEST_EXAMPLES} digital_16k test examples, found {len(provenance)}"
        )
    return provenance


def load_digital_16k_example(entry: dict[str, Any]) -> bench.RawExample:
    """Fresh from-disk read + the exact non-timed resample/cast load_raw_test_examples performs."""
    import soundfile as sf

    values, sample_rate = sf.read(entry["wav_path"], always_2d=False, dtype="float32")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected mono waveform: {entry['wav_path']}")
    values = bench._resample_poly(values, int(sample_rate), 16_000)
    values = values.astype(np.float16).astype(np.float32)
    return bench.RawExample(
        sample_id=entry["sample_id"],
        waveform=np.ascontiguousarray(values),
        source_rate=16_000,
        label=entry["label"],
    )


def fabricated_capture_directories(capture_root: Path, stimuli_root: Path) -> list[Path]:
    """Like bench._raw_capture_directories, but only requires the files this
    script actually reads (aligned_events.csv, envelopes_400Hz_metadata.json,
    envelopes_400Hz_capture.npz) -- not raw_capture.npz or
    envelopes_400Hz_annotated.csv, which the Pi deployment omits entirely."""
    if capture_root.name == "test":
        capture_root = capture_root.parent
    expected = bench._expected_test_recording_ids(stimuli_root)
    by_recording: dict[int, Path] = {}
    for metadata_path in sorted((capture_root / "test").rglob("metadata.json")):
        run_dir = metadata_path.parent
        if not (run_dir / "aligned_events.csv").is_file():
            continue
        if not (run_dir / "envelopes_400Hz_metadata.json").is_file():
            continue
        if not (run_dir / "envelopes_400Hz_capture.npz").is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        recording_id = int(metadata.get("manifest_recording_id", -1))
        if recording_id not in expected:
            continue
        if recording_id in by_recording:
            raise RuntimeError(f"Duplicate fabricated capture for recording {recording_id}")
        by_recording[recording_id] = run_dir
    missing = sorted(expected - set(by_recording))
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} fabricated test captures; first IDs: {missing[:12]}"
        )
    return [by_recording[index] for index in sorted(by_recording)]


def build_fabricated_provenance(
    capture_root: Path, test_split_path: Path, context_ms: float
) -> list[dict[str, Any]]:
    stimuli_root = test_split_path.parent
    permitted_speakers = bench._speaker_set_from_json(stimuli_root / "protocol.json")
    if permitted_speakers is None:
        raise ValueError("The fabricated capture loader requires stimuli/protocol.json")

    provenance: list[dict[str, Any]] = []
    run_dirs = fabricated_capture_directories(capture_root, stimuli_root)
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
        speaker = int(metadata["dataset_speaker_id"])
        if speaker not in permitted_speakers:
            raise RuntimeError(f"{run_dir.name}: speaker {speaker:02d} is not in the test split")

        envelope_metadata = json.loads(
            (run_dir / "envelopes_400Hz_metadata.json").read_text(encoding="ascii")
        )
        if not math.isclose(float(envelope_metadata["envelope_rate_hz"]), 400.0):
            raise RuntimeError(f"{run_dir.name}: saved envelope rate is not 400 Hz")

        npz_path = run_dir / "envelopes_400Hz_capture.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"{run_dir.name}: missing envelopes_400Hz_capture.npz -- run "
                "convert_envelopes_400hz_to_npz.py first"
            )
        with np.load(npz_path, allow_pickle=False) as archive:
            center_times = np.asarray(archive["time_s"], dtype=np.float64)
            if not math.isclose(float(archive["sample_rate_hz"]), 400.0):
                raise RuntimeError(f"{run_dir.name}: envelope npz sample_rate_hz is not 400 Hz")
        availability_times = center_times + 0.005

        with (run_dir / "aligned_events.csv").open("r", encoding="utf-8", newline="") as handle:
            events = [row for row in csv.DictReader(handle) if str(row["event_type"]) == "letter"]
        for event in sorted(events, key=lambda row: float(row["start_s"])):
            label = int(event["label"])
            start_s = max(
                float(availability_times[0]), float(event["start_s"]) - context_ms / 1_000.0
            )
            end_s = min(
                float(availability_times[-1]) + 1.0 / 400.0,
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
            f"Expected {bench.EXPECTED_TEST_EXAMPLES} fabricated test examples, found {len(provenance)}"
        )
    return provenance


def load_fabricated_example(entry: dict[str, Any]) -> bench.FabricatedExample:
    """Fresh from-disk read of the saved post-envelope npz slice for one sample."""
    with np.load(entry["npz_path"], allow_pickle=False) as archive:
        envelope_values = np.asarray(archive["envelope_v"], dtype=np.float32)
    features = np.ascontiguousarray(
        envelope_values[entry["first"] : entry["last"]], dtype=np.float32
    )
    return bench.FabricatedExample(
        sample_id=entry["sample_id"],
        features=features,
        label=entry["label"],
        audio_duration_seconds=(entry["last"] - entry["first"]) / 400.0,
    )


def checkpoint_for(
    representation: str,
    model_name: str,
    model_16k_dir: Path,
    model_fabricated_baselines_dir: Path,
    model_fabricated_cnn_baselines_dir: Path,
) -> Path:
    if representation == "digital_16k":
        return model_16k_dir / f"{model_name}_best.pt"
    if model_name in bench.CNN_BASELINE_MODEL_NAMES:
        return (
            model_fabricated_cnn_baselines_dir
            / model_name
            / f"best_fabricated_basic_cnn_{model_name}_classifier.pt"
        )
    return (
        model_fabricated_baselines_dir
        / model_name
        / f"best_fabricated_400hz_5d_{model_name}_classifier.pt"
    )


def validate_paths(args: argparse.Namespace) -> None:
    required = {"test_split_path": args.test_split_path}
    if args.representation == "digital_16k":
        required["raw_audiomnist_path"] = args.raw_audiomnist_path
    else:
        required["fabricated_feature_path"] = args.fabricated_feature_path
    required["checkpoint"] = checkpoint_for(
        args.representation,
        args.model,
        args.model_16k_dir,
        args.model_fabricated_baselines_dir,
        args.model_fabricated_cnn_baselines_dir,
    )
    invalid = [
        f"{name}={path}"
        for name, path in required.items()
        if str(path).startswith("/path/to/") or not path.exists()
    ]
    if invalid:
        raise FileNotFoundError(
            "Configure the following paths at the top of the script or with CLI flags:\n  "
            + "\n  ".join(invalid)
        )


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------
def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    bench.configure_torch_runtime()
    representation = args.representation
    system = "digital_16k" if representation == "digital_16k" else "fabricated_400hz_baselines_post_envelope"
    checkpoint_path = checkpoint_for(
        representation,
        args.model,
        args.model_16k_dir,
        args.model_fabricated_baselines_dir,
        args.model_fabricated_cnn_baselines_dir,
    ).resolve()
    loaded = bench.load_model(system, checkpoint_path, args.model)
    preprocess = bench._preprocessor(loaded)

    if representation == "digital_16k":
        provenance = build_digital_16k_provenance(
            args.test_split_path.resolve(), args.raw_audiomnist_path.resolve()
        )
        load_example: Callable[[dict[str, Any]], Any] = load_digital_16k_example
    else:
        context_ms = args.context_ms
        if context_ms is None:
            context_ms = float(loaded.checkpoint_metadata.get("context_ms", DEFAULT_CONTEXT_MS))
        provenance = build_fabricated_provenance(
            args.fabricated_feature_path.resolve(), args.test_split_path.resolve(), context_ms
        )
        load_example = load_fabricated_example

    stage = args.stage
    runs = args.runs
    warmup = args.warmup

    # MACs: a static model/input-shape property. One untimed profiling pass
    # on a synthetic input of the shape a real example would produce, so it
    # is available regardless of which stage's timings are being collected.
    sample_example = load_example(provenance[0])
    nn_macs = bench.profile_model(loaded.model, bench._profile_input(loaded, sample_example))

    for index in range(warmup):
        entry = provenance[bench.scheduled_example_index(index, warmup, len(provenance))]
        example = load_example(entry)
        if stage != "load":
            prepared = preprocess(example)
            if stage == "full":
                bench._forward(loaded.model, prepared)

    rows: list[dict[str, Any]] = []
    for run_index in range(runs):
        entry = provenance[bench.scheduled_example_index(run_index, runs, len(provenance))]

        load_start = time.perf_counter_ns()
        example = load_example(entry)
        load_ns = time.perf_counter_ns() - load_start

        row: dict[str, Any] = {
            "run_index": run_index,
            "sample_id": entry["sample_id"],
            "label": int(entry["label"]),
            "load_ms": load_ns / 1e6,
        }
        total_ns = load_ns

        if stage in ("frontend", "full"):
            frontend_start = time.perf_counter_ns()
            prepared = preprocess(example)
            frontend_ns = time.perf_counter_ns() - frontend_start
            row["frontend_ms"] = frontend_ns / 1e6
            total_ns += frontend_ns

        if stage == "full":
            nn_start = time.perf_counter_ns()
            logits = bench._forward(loaded.model, prepared)
            nn_ns = time.perf_counter_ns() - nn_start
            row["nn_ms"] = nn_ns / 1e6
            row["prediction"] = int(logits.argmax(dim=1).item())
            total_ns += nn_ns

        row["total_ms"] = total_ns / 1e6
        rows.append(row)

    load_values = [row["load_ms"] for row in rows]
    total_values = [row["total_ms"] for row in rows]
    summary: dict[str, Any] = {
        "system": loaded.name,
        "representation": loaded.representation,
        "stage": stage,
        "model": args.model,
        "measured_runs": runs,
        "warmup_iterations": warmup,
        "nn_macs": int(nn_macs),
        "load_median_ms": statistics.median(load_values),
        "load_p95_ms": bench.percentile(load_values, 95),
        "total_median_ms": statistics.median(total_values),
        "total_p95_ms": bench.percentile(total_values, 95),
    }
    if stage in ("frontend", "full"):
        frontend_values = [row["frontend_ms"] for row in rows]
        summary["frontend_median_ms"] = statistics.median(frontend_values)
        summary["frontend_p95_ms"] = bench.percentile(frontend_values, 95)
    if stage == "full":
        nn_values = [row["nn_ms"] for row in rows]
        summary["nn_median_ms"] = statistics.median(nn_values)
        summary["nn_p95_ms"] = bench.percentile(nn_values, 95)

        print(f"loading all {len(provenance)} test examples for accuracy ...", flush=True)
        full_examples = [load_example(entry) for entry in provenance]
        accuracy, _ = bench.evaluate_accuracy(loaded.model, full_examples, preprocess)
        params = sum(parameter.numel() for parameter in loaded.model.parameters())
        summary["accuracy"] = accuracy
        summary["params"] = int(params)
        summary["size_kb"] = params * 4 / 1024.0

    summary["details"] = {
        "checkpoint": str(checkpoint_path),
        "architecture": loaded.architecture,
        "stage_definition": {
            "load": "disk I/O only; no frontend, no NN forward pass",
            "frontend": "load + front-end feature extraction; no NN forward pass",
            "full": "load + frontend + NN forward pass; accuracy/params/size computed here only",
        }[stage],
        "nn_macs_definition": "static model/input-shape property from one untimed profiling pass",
    }
    return summary, rows


def write_outputs(results_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    key = f"{summary['system'].replace('/', '__')}__{summary['stage']}"
    summary_path = results_dir / f"{key}_summary.json"
    timing_path = results_dir / f"{key}_timings.csv"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    if rows:
        fields = list(rows[0].keys())
        with timing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {summary_path}")
    print(f"wrote {timing_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--context-ms",
        type=float,
        default=None,
        help="Fabricated event padding in ms. Defaults to the checkpoint's own context_ms.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--test-split-path", type=Path, default=TEST_SPLIT_PATH)
    parser.add_argument("--raw-audiomnist-path", type=Path, default=RAW_AUDIOMNIST_PATH)
    parser.add_argument("--fabricated-feature-path", type=Path, default=FABRICATED_FEATURE_PATH)
    parser.add_argument("--model-16k-dir", type=Path, default=MODEL_16K_DIR)
    parser.add_argument(
        "--model-fabricated-baselines-dir", type=Path, default=MODEL_FABRICATED_BASELINES_DIR
    )
    parser.add_argument(
        "--model-fabricated-cnn-baselines-dir",
        type=Path,
        default=MODEL_FABRICATED_CNN_BASELINES_DIR,
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    validate_paths(args)
    summary, rows = run_stage(args)
    write_outputs(args.results_dir, summary, rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
