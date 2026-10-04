#!/usr/bin/env python3
"""Train the pooled causal TCN on five 400 Hz AC-RMS envelope channels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import train_direct_400hz_5d_pooled_tcn_classifier as base
from train_direct_400hz_5d_tcn_ctc import (
    DIRECT_INPUT_FEATURES,
    DIRECT_INPUT_RATE_HZ,
    FRONTEND_KERNEL_SIZE,
    FRONTEND_STRIDE,
    DirectDigitExample,
    allowed_speakers,
    capture_directories,
)


RMS_WINDOW_MS = 10.0
ENVELOPE_HOP_MS = 2.5
ENVELOPE_RATE_HZ = 400.0
ENVELOPE_COLUMNS = (
    "AI0_R300_rms_v",
    "AI1_R3500_rms_v",
    "AI2_R7500_rms_v",
    "AI3_R1800_rms_v",
    "AI4_R4500_rms_v",
)
FEATURE_NAMES = (
    "R300_ac_rms_envelope_v",
    "R3500_ac_rms_envelope_v",
    "R7500_ac_rms_envelope_v",
    "R1800_ac_rms_envelope_v",
    "R4500_ac_rms_envelope_v",
)
FEATURE_CONTRACT = (
    "audiomnist_ac_rms_envelope_5d_400hz_10ms_window_"
    "2p5ms_hop_causal_tcn_stats_pool_softmax_v1"
)
OUTPUT_ROOT = base.HERE / "models" / "envelope_400hz_5d_pooled_tcn_classifier"
CACHE_ROOT = OUTPUT_ROOT / "envelope_400hz_capture_cache"
CHECKPOINT_FILENAME = "best_envelope_400hz_5d_pooled_tcn_classifier.pt"
ARCHITECTURE_ID = "envelope_400hz_c32_fk32_long_tcn_stats_pool_softmax"


def load_cached_envelope_capture(
    run_dir: Path,
    cache_root: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
    recording_id = int(metadata["manifest_recording_id"])
    envelope_path = run_dir / "envelopes_400Hz_annotated.csv"
    envelope_metadata_path = run_dir / "envelopes_400Hz_metadata.json"
    if not envelope_path.is_file() or not envelope_metadata_path.is_file():
        raise FileNotFoundError(f"{run_dir.name}: 400 Hz envelope files are missing")
    envelope_metadata = json.loads(envelope_metadata_path.read_text(encoding="ascii"))
    if not math.isclose(float(envelope_metadata["envelope_rate_hz"]), ENVELOPE_RATE_HZ):
        raise RuntimeError(f"{run_dir.name}: envelope rate is not 400 Hz")
    if not math.isclose(
        float(envelope_metadata["rms_window_duration_ms"]), RMS_WINDOW_MS
    ):
        raise RuntimeError(f"{run_dir.name}: RMS window is not 10 ms")
    if not math.isclose(
        float(envelope_metadata["envelope_hop_duration_ms"]), ENVELOPE_HOP_MS
    ):
        raise RuntimeError(f"{run_dir.name}: envelope hop is not 2.5 ms")

    source_stat = envelope_path.stat()
    cache_dir = cache_root / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{recording_id:04d}_{run_dir.name}_rms_envelope400.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path) as cached:
                size_ok = int(cached["source_size_bytes"]) == source_stat.st_size
                time_ok = int(cached["source_mtime_ns"]) == source_stat.st_mtime_ns
                if size_ok and time_ok:
                    times = np.asarray(cached["availability_times_s"], dtype=np.float64)
                    values = np.asarray(cached["envelope_v"], dtype=np.float32)
                    events = pd.read_csv(run_dir / "aligned_events.csv")
                    return times, values, events, metadata
        except (OSError, ValueError, KeyError):
            pass

    table = pd.read_csv(envelope_path, usecols=("time_s", *ENVELOPE_COLUMNS))
    # Stored timestamps are window centers. A causal system can use each value only
    # after the trailing 10 ms window has ended, 5 ms after its center.
    times = table["time_s"].to_numpy(np.float64) + RMS_WINDOW_MS / 2000.0
    values = table.loc[:, ENVELOPE_COLUMNS].to_numpy(np.float32)
    if values.ndim != 2 or values.shape[1] != DIRECT_INPUT_FEATURES:
        raise RuntimeError(f"{run_dir.name}: expected five envelope channels")
    if not np.isfinite(values).all() or np.min(values) < 0.0:
        raise RuntimeError(f"{run_dir.name}: invalid RMS-envelope values")
    np.savez_compressed(
        cache_path,
        availability_times_s=times,
        envelope_v=values,
        source_size_bytes=np.int64(source_stat.st_size),
        source_mtime_ns=np.int64(source_stat.st_mtime_ns),
    )
    events = pd.read_csv(run_dir / "aligned_events.csv")
    return times, values, events, metadata


def build_envelope_split(
    split: str,
    capture_root: Path,
    stimuli_root: Path,
    cache_root: Path,
    context_ms: float,
    require_complete: bool,
) -> tuple[list[DirectDigitExample], pd.DataFrame, int]:
    manifest_path = stimuli_root / f"audiomnist_{split}_manifest.csv"
    protocol_path = stimuli_root / "protocol.json"
    permitted_speakers = allowed_speakers(protocol_path, split)
    run_dirs = capture_directories(
        capture_root / split, manifest_path, require_complete=require_complete
    )
    examples: list[DirectDigitExample] = []
    audit_rows: list[dict[str, object]] = []
    context_s = context_ms / 1000.0
    seen_speakers: set[str] = set()

    for run_index, run_dir in enumerate(run_dirs, start=1):
        times, values, events, metadata = load_cached_envelope_capture(
            run_dir, cache_root, split
        )
        metadata_split = str(metadata.get("dataset_split", split))
        if metadata_split != split:
            raise RuntimeError(
                f"{run_dir.name}: expected split {split}, found {metadata_split}"
            )
        speaker_id = str(metadata.get("dataset_speaker_id", "")).zfill(2)
        if speaker_id not in permitted_speakers:
            raise RuntimeError(
                f"{run_dir.name}: speaker {speaker_id} is not in the {split} split"
            )
        seen_speakers.add(speaker_id)
        recording_id = int(metadata["manifest_recording_id"])
        digits = events.loc[events["event_type"].eq("letter")].sort_values("start_s")
        if digits.empty:
            raise RuntimeError(f"{run_dir.name}: no digit events")

        for event in digits.itertuples(index=False):
            label = int(event.label)
            if label not in range(10):
                raise RuntimeError(f"{run_dir.name}: invalid digit label {event.label}")
            start_s = max(float(times[0]), float(event.start_s) - context_s)
            end_s = min(
                float(times[-1]) + 1.0 / ENVELOPE_RATE_HZ,
                float(event.end_s) + context_s,
            )
            first = int(np.searchsorted(times, start_s, side="left"))
            last = int(np.searchsorted(times, end_s, side="left"))
            segment = values[first:last].copy()
            if len(segment) < FRONTEND_KERNEL_SIZE:
                raise RuntimeError(
                    f"{run_dir.name}: digit {label} has only {len(segment)} envelope steps"
                )
            repetition = int(event.source_repetition)
            examples.append(
                DirectDigitExample(
                    values=segment,
                    label=label,
                    speaker_id=speaker_id,
                    repetition=repetition,
                    run=run_dir.name,
                    recording_id=recording_id,
                    start_s=start_s,
                    end_s=end_s,
                )
            )
            audit_rows.append(
                {
                    "split": split,
                    "run": run_dir.name,
                    "recording_id": recording_id,
                    "speaker_id": speaker_id,
                    "repetition": repetition,
                    "digit": label,
                    "input_steps_400hz": len(segment),
                    "model_output_steps_100hz": math.ceil(
                        len(segment) / FRONTEND_STRIDE
                    ),
                    "input_features_per_step": DIRECT_INPUT_FEATURES,
                    "rms_window_ms": RMS_WINDOW_MS,
                    "envelope_hop_ms": ENVELOPE_HOP_MS,
                    "feature_availability_delay_ms": RMS_WINDOW_MS,
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
        print(
            f"envelope {split} [{run_index:03d}/{len(run_dirs):03d}] {run_dir.name}",
            flush=True,
        )

    if seen_speakers != permitted_speakers and require_complete:
        missing = sorted(permitted_speakers - seen_speakers)
        raise RuntimeError(f"Missing {split} speakers: {missing}")
    return examples, pd.DataFrame(audit_rows), len(run_dirs)


def configure_base() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SHARED_CACHE_ROOT = CACHE_ROOT
    base.FEATURE_CONTRACT = FEATURE_CONTRACT
    base.FEATURE_NAMES = FEATURE_NAMES
    base.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
    base.ARCHITECTURE_ID = ARCHITECTURE_ID
    base.REPRESENTATION_DESCRIPTION = "five-channel 400 Hz AC-RMS envelope"
    base.REPRESENTATION_SUMMARY = {
        "source_capture_rate_hz_nominal": 48_000.0,
        "envelope_rate_hz_per_sensor": ENVELOPE_RATE_HZ,
        "physical_sensor_count": DIRECT_INPUT_FEATURES,
        "input_features_per_2p5ms_step": DIRECT_INPUT_FEATURES,
        "input_scalar_rate_per_second": int(
            ENVELOPE_RATE_HZ * DIRECT_INPUT_FEATURES
        ),
        "feature_names": FEATURE_NAMES,
        "rms_window_ms": RMS_WINDOW_MS,
        "envelope_hop_ms": ENVELOPE_HOP_MS,
        "window_overlap_fraction": 0.75,
        "feature_availability_latency_ms": RMS_WINDOW_MS,
        "magnitude_operation_after_envelope": False,
        "fixed_rms_or_average_used": True,
        "anti_alias_filter_used": False,
        "analog_envelope_detector_used": False,
        "envelope_implementation": "software AC-RMS from full-rate DAQ waveform",
        "estimated_streaming_rms_arithmetic_ops_per_second": 1_200_000,
        "envelope_square_roots_per_second": 2_000,
    }
    base.build_split = build_envelope_split


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
