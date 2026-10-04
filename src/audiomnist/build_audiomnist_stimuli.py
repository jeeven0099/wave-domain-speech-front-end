#!/usr/bin/env python3
"""Build synchronized AudioMNIST playback chunks for the five-resonator DAQ.

The builder uses the published AudioMNIST digit split 1. A playback chunk never
contains more than one source speaker, and the held-out test speakers are kept
in a separate manifest. Each source utterance is peak-normalized to make this a
controlled front-end experiment rather than a playback-loudness classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATASET = ROOT / "datasets/AudioMNIST"
DEFAULT_OUTPUT = HERE / "stimuli"
SAMPLE_RATE = 48_000
MARKER_PEAK = 10 ** (-9.0 / 20.0)
FILE_PATTERN = re.compile(r"(?P<digit>[0-9])_(?P<speaker>[0-9]{2})_(?P<repetition>[0-9]+)\.wav$")

# Published AudioMNIST digit-classification split 1 from preprocess_data.py.
SPLIT_SPEAKERS = {
    "train": (
        28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53,
        36, 57, 9, 24, 37, 2, 8, 17, 29, 39, 48, 54,
        43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55,
    ),
    "validation": (12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50),
    "test": (26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronization_preamble() -> tuple[np.ndarray, list[dict[str, object]]]:
    lead_samples = int(SAMPLE_RATE * 1.0)
    burst_samples = int(SAMPLE_RATE * 0.08)
    gap_samples = int(SAMPLE_RATE * 0.22)
    settle_samples = int(SAMPLE_RATE * 1.0)
    burst_time = np.arange(burst_samples) / SAMPLE_RATE
    burst = signal.chirp(
        burst_time,
        f0=700.0,
        f1=8500.0,
        t1=burst_time[-1],
        method="logarithmic",
    )
    burst *= signal.windows.hann(burst_samples, sym=False) * MARKER_PEAK

    parts = [np.zeros(lead_samples, dtype=np.float64)]
    events: list[dict[str, object]] = []
    cursor = lead_samples
    for marker_index in range(3):
        start = cursor / SAMPLE_RATE
        parts.append(burst)
        cursor += len(burst)
        events.append(
            {
                "event_type": "sync",
                "label": f"SYNC{marker_index + 1}",
                "start_s": start,
                "end_s": cursor / SAMPLE_RATE,
                "source_speaker_id": "",
                "source_repetition": "",
                "source_wav": "",
            }
        )
        if marker_index < 2:
            parts.append(np.zeros(gap_samples, dtype=np.float64))
            cursor += gap_samples
    parts.append(np.zeros(settle_samples, dtype=np.float64))
    return np.concatenate(parts), events


def load_utterance(path: Path, target_peak: float) -> np.ndarray:
    sample_rate, samples = wavfile.read(path)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, found {sample_rate} Hz")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{path}: expected mono audio, found shape {values.shape}")
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        values /= max(abs(info.min), info.max)
    values -= np.mean(values)
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-8:
        raise ValueError(f"{path}: source utterance is silent")
    values *= target_peak / peak
    fade_samples = min(len(values) // 2, int(round(0.005 * SAMPLE_RATE)))
    if fade_samples:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade_samples)) ** 2
        values[:fade_samples] *= ramp
        values[-fade_samples:] *= ramp[::-1]
    return values


def load_inventory(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    data_root = dataset_root / "data"
    metadata_path = data_root / "audioMNIST_meta.txt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"AudioMNIST metadata was not found at {metadata_path}")
    speaker_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for path in sorted(data_root.glob("[0-9][0-9]/*.wav")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected AudioMNIST filename: {path}")
        speaker_id = int(match.group("speaker"))
        rows.append(
            {
                "digit": int(match.group("digit")),
                "speaker_id": speaker_id,
                "repetition": int(match.group("repetition")),
                "source_wav": str(path.resolve()),
            }
        )
    inventory = pd.DataFrame(rows)
    if len(inventory) != 30_000:
        raise RuntimeError(f"Expected 30,000 AudioMNIST WAV files, found {len(inventory):,}")
    counts = inventory.groupby(["speaker_id", "digit"]).size()
    if len(counts) != 600 or not counts.eq(50).all():
        raise RuntimeError("AudioMNIST must contain 50 repetitions of every digit for every speaker")
    all_split_speakers = [speaker for values in SPLIT_SPEAKERS.values() for speaker in values]
    if len(all_split_speakers) != 60 or set(all_split_speakers) != set(range(1, 61)):
        raise RuntimeError("Published speaker split is not a disjoint partition of speakers 1-60")
    return inventory, speaker_metadata


def shuffled_speaker_rows(inventory: pd.DataFrame, speaker_id: int, seed: int) -> list[dict[str, object]]:
    speaker = inventory.loc[inventory["speaker_id"].eq(speaker_id)]
    by_key = {
        (int(row.digit), int(row.repetition)): row._asdict()
        for row in speaker.itertuples(index=False)
    }
    rng = random.Random(seed + speaker_id)
    repetition_order = list(range(50))
    rng.shuffle(repetition_order)
    ordered: list[dict[str, object]] = []
    for repetition in repetition_order:
        digits = list(range(10))
        rng.shuffle(digits)
        ordered.extend(by_key[(digit, repetition)] for digit in digits)
    return ordered


def write_chunk(
    split: str,
    speaker_id: int,
    chunk_index: int,
    recording_id: int,
    samples: list[np.ndarray],
    events: list[dict[str, object]],
    output_root: Path,
    metadata: dict[str, object],
    target_peak_dbfs: float,
) -> dict[str, object]:
    trailing = np.zeros(SAMPLE_RATE, dtype=np.float64)
    stimulus = np.concatenate([*samples, trailing])
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = f"audiomnist_{split}_spk{speaker_id:02d}_chunk{chunk_index:02d}"
    wav_path = split_dir / f"{stem}.wav"
    event_path = split_dir / f"{stem}_events.csv"
    wavfile.write(wav_path, SAMPLE_RATE, np.int16(np.clip(stimulus, -1.0, 1.0) * 32767))
    pd.DataFrame(events).to_csv(event_path, index=False, float_format="%.6f")

    digit_events = [event for event in events if event["event_type"] == "letter"]
    speaker_key = f"{speaker_id:02d}"
    speaker_info = metadata[speaker_key]
    return {
        "recording_id": recording_id,
        "task_name": "AudioMNIST isolated spoken-digit classification",
        "capture_prefix": "audiomnist",
        "dataset_split": split,
        "speaker_id": speaker_key,
        "speaker_gender": speaker_info.get("gender", ""),
        "speaker_accent": speaker_info.get("accent", ""),
        "speaker_native": speaker_info.get("native speaker", ""),
        "chunk_index": chunk_index,
        "digit_order": "".join(str(event["label"]) for event in digit_events),
        "letter_order": "".join(str(event["label"]) for event in digit_events),
        "token_count": len(digit_events),
        "stimulus_wav": str(wav_path.resolve()),
        # The established capture tool displays this field but VLC plays stimulus_wav.
        "stimulus_mp3": str(wav_path.resolve()),
        "event_table": str(event_path.resolve()),
        "sample_rate_hz": SAMPLE_RATE,
        "duration_s": len(stimulus) / SAMPLE_RATE,
        "target_peak_dbfs_per_utterance": target_peak_dbfs,
        "marker_peak_dbfs": -9.0,
        "wav_sha256": sha256(wav_path),
    }


def build_speaker_chunks(
    inventory: pd.DataFrame,
    metadata: dict[str, object],
    split: str,
    speaker_id: int,
    first_recording_id: int,
    output_root: Path,
    max_seconds: float,
    gap_seconds: float,
    target_peak_dbfs: float,
    seed: int,
) -> list[dict[str, object]]:
    preamble, sync_events = synchronization_preamble()
    max_samples = int(round(max_seconds * SAMPLE_RATE))
    gap = np.zeros(int(round(gap_seconds * SAMPLE_RATE)), dtype=np.float64)
    target_peak = 10 ** (target_peak_dbfs / 20.0)
    ordered = shuffled_speaker_rows(inventory, speaker_id, seed)
    output_rows: list[dict[str, object]] = []
    chunk_index = 1
    chunk_parts: list[np.ndarray] = [preamble]
    chunk_events = [dict(event) for event in sync_events]
    cursor = len(preamble)
    token_count = 0

    def finalize() -> None:
        nonlocal chunk_index, chunk_parts, chunk_events, cursor, token_count
        if token_count == 0:
            return
        output_rows.append(
            write_chunk(
                split,
                speaker_id,
                chunk_index,
                first_recording_id + len(output_rows),
                chunk_parts,
                chunk_events,
                output_root,
                metadata,
                target_peak_dbfs,
            )
        )
        chunk_index += 1
        chunk_parts = [preamble]
        chunk_events = [dict(event) for event in sync_events]
        cursor = len(preamble)
        token_count = 0

    for row in ordered:
        source_path = Path(str(row["source_wav"]))
        utterance = load_utterance(source_path, target_peak)
        required = len(utterance) + len(gap) + SAMPLE_RATE
        if token_count and cursor + required > max_samples:
            finalize()
        start_s = cursor / SAMPLE_RATE
        chunk_parts.append(utterance)
        cursor += len(utterance)
        chunk_events.append(
            {
                "event_type": "letter",
                "label": int(row["digit"]),
                "start_s": start_s,
                "end_s": cursor / SAMPLE_RATE,
                "source_speaker_id": f"{speaker_id:02d}",
                "source_repetition": int(row["repetition"]),
                "source_wav": str(source_path.resolve()),
            }
        )
        chunk_parts.append(gap)
        cursor += len(gap)
        token_count += 1
    finalize()
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-stimulus-seconds", type=float, default=120.0)
    parser.add_argument("--inter-token-gap-ms", type=float, default=150.0)
    parser.add_argument("--target-peak-dbfs", type=float, default=-9.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated subset of train,validation,test",
    )
    args = parser.parse_args()

    requested = [value.strip() for value in args.splits.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(SPLIT_SPEAKERS))
    if unknown:
        raise ValueError(f"Unknown splits: {unknown}")
    if args.max_stimulus_seconds < 30:
        raise ValueError("Use chunks of at least 30 seconds")
    if not 0 <= args.inter_token_gap_ms <= 1000:
        raise ValueError("Inter-token gap must be between 0 and 1000 ms")
    if not -30 <= args.target_peak_dbfs <= -3:
        raise ValueError("Target peak must be between -30 and -3 dBFS")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory, speaker_metadata = load_inventory(dataset_root)

    manifest_rows: list[dict[str, object]] = []
    recording_id = 1
    for split in ("train", "validation", "test"):
        if split not in requested:
            continue
        split_rows: list[dict[str, object]] = []
        for index, speaker_id in enumerate(SPLIT_SPEAKERS[split], start=1):
            rows = build_speaker_chunks(
                inventory,
                speaker_metadata,
                split,
                speaker_id,
                recording_id,
                output_root,
                args.max_stimulus_seconds,
                args.inter_token_gap_ms / 1000.0,
                args.target_peak_dbfs,
                args.seed,
            )
            recording_id += len(rows)
            split_rows.extend(rows)
            print(
                f"{split}: speaker {speaker_id:02d} [{index:02d}/{len(SPLIT_SPEAKERS[split]):02d}], "
                f"{len(rows)} chunks"
            )
        split_manifest = pd.DataFrame(split_rows)
        split_manifest.to_csv(output_root / f"audiomnist_{split}_manifest.csv", index=False)
        pd.DataFrame(
            {
                "run_order": np.arange(1, len(split_rows) + 1),
                "recording_id": split_manifest["recording_id"],
                "speaker_id": split_manifest["speaker_id"],
                "chunk_index": split_manifest["chunk_index"],
                "status": "pending",
                "notes": "",
            }
        ).to_csv(output_root / f"audiomnist_{split}_run_plan.csv", index=False)
        manifest_rows.extend(split_rows)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "audiomnist_all_manifest.csv", index=False)
    split_summary = (
        manifest.groupby("dataset_split")
        .agg(chunks=("recording_id", "count"), speakers=("speaker_id", "nunique"), tokens=("token_count", "sum"), playback_seconds=("duration_s", "sum"))
        .reset_index()
    )
    split_summary["playback_hours"] = split_summary["playback_seconds"] / 3600.0
    split_summary.to_csv(output_root / "audiomnist_stimulus_summary.csv", index=False)
    protocol = {
        "dataset": "AudioMNIST",
        "dataset_root": str(dataset_root),
        "published_digit_split_index": 1,
        "speaker_splits": {key: [f"{value:02d}" for value in values] for key, values in SPLIT_SPEAKERS.items()},
        "sample_rate_hz": SAMPLE_RATE,
        "resonator_mapping_hz": [300, 3500, 7500, 1800, 4500],
        "source_utterances": len(inventory),
        "inter_token_gap_ms": args.inter_token_gap_ms,
        "per_utterance_peak_normalization_dbfs": args.target_peak_dbfs,
        "maximum_stimulus_duration_s": args.max_stimulus_seconds,
        "sync": "three 80 ms logarithmic chirps from 700 to 8500 Hz",
        "test_policy": "Do not capture or evaluate the test split until the model and decoder are frozen.",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="ascii"
    )
    print("\n" + split_summary.to_string(index=False))
    print(f"\nStimuli: {output_root}")


if __name__ == "__main__":
    main()
