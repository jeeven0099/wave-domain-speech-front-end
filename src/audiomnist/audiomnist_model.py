"""Shared data and model code for the physical AudioMNIST experiment."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset


SENSORS = ("R300", "R3500", "R7500", "R1800", "R4500")
RMS_COLUMNS = tuple(f"AI{index}_{sensor}_rms_v" for index, sensor in enumerate(SENSORS))
SUBFRAMES_PER_STEP = 4
MODEL_HOP_MS = 10.0
FEATURE_NAMES = tuple(
    f"{column}_subframe_{subframe}"
    for column in RMS_COLUMNS
    for subframe in range(SUBFRAMES_PER_STEP)
)


@dataclass
class DigitExample:
    values: np.ndarray
    label: int
    speaker_id: str
    repetition: int
    run: str
    recording_id: int
    start_s: float
    end_s: float


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def protocol_speakers(protocol_path: Path, split: str) -> set[str]:
    protocol = json.loads(protocol_path.read_text(encoding="ascii"))
    return {str(value).zfill(2) for value in protocol["speaker_splits"][split]}


def expected_recording_ids(manifest_path: Path) -> set[int]:
    manifest = pd.read_csv(manifest_path)
    return {int(value) for value in manifest["recording_id"]}


def capture_directories(
    capture_root: Path,
    manifest_path: Path,
    require_complete: bool = True,
) -> list[Path]:
    expected = expected_recording_ids(manifest_path)
    by_recording: dict[int, Path] = {}
    for metadata_path in sorted(capture_root.rglob("metadata.json")):
        run_dir = metadata_path.parent
        envelope_path = run_dir / "envelopes_400Hz_annotated.csv"
        event_path = run_dir / "aligned_events.csv"
        if not envelope_path.is_file() or not event_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        raw_id = metadata.get("manifest_recording_id")
        if raw_id is None or int(raw_id) not in expected:
            continue
        recording_id = int(raw_id)
        if recording_id in by_recording:
            raise RuntimeError(
                f"Recording {recording_id} has duplicate accepted captures: "
                f"{by_recording[recording_id]} and {run_dir}"
            )
        by_recording[recording_id] = run_dir
    missing = sorted(expected - set(by_recording))
    if require_complete and missing:
        preview = ", ".join(map(str, missing[:12]))
        suffix = "..." if len(missing) > 12 else ""
        raise RuntimeError(
            f"Missing {len(missing)} manifest captures under {capture_root}: {preview}{suffix}"
        )
    return [by_recording[key] for key in sorted(by_recording)]


def load_capture(run_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
    table = pd.read_csv(
        run_dir / "envelopes_400Hz_annotated.csv",
        usecols=["time_s", *RMS_COLUMNS],
    )
    events = pd.read_csv(run_dir / "aligned_events.csv")
    times_400hz = table["time_s"].to_numpy(np.float64)
    rms = table[list(RMS_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(rms).all():
        raise RuntimeError(f"{run_dir.name}: non-finite RMS values")

    sync = events.loc[events["event_type"].eq("sync")]
    if sync.empty:
        raise RuntimeError(f"{run_dir.name}: no synchronization events")
    baseline_mask = times_400hz < float(sync["start_s"].min()) - 0.10
    if baseline_mask.sum() < 200:
        raise RuntimeError(f"{run_dir.name}: insufficient pre-marker quiet baseline")
    baseline = np.median(rms[baseline_mask], axis=0)
    db_400hz = 20.0 * np.log10(np.maximum(rms, 1e-12) / np.maximum(baseline, 1e-12))
    db_400hz = np.clip(db_400hz, -30.0, 50.0)

    usable = len(db_400hz) - len(db_400hz) % SUBFRAMES_PER_STEP
    packed = (
        db_400hz[:usable]
        .reshape(-1, SUBFRAMES_PER_STEP, len(SENSORS))
        .transpose(0, 2, 1)
        .reshape(-1, len(FEATURE_NAMES))
    )
    times_10ms = times_400hz[:usable].reshape(-1, SUBFRAMES_PER_STEP).mean(axis=1)
    return times_10ms, packed.astype(np.float32), events, metadata


def build_examples(
    run_dirs: list[Path],
    split: str,
    allowed_speakers: set[str],
    context_ms: float = 50.0,
) -> tuple[list[DigitExample], pd.DataFrame]:
    examples: list[DigitExample] = []
    audit_rows: list[dict[str, object]] = []
    context_s = context_ms / 1000.0
    seen_speakers: set[str] = set()
    for run_index, run_dir in enumerate(run_dirs, start=1):
        times, features, events, metadata = load_capture(run_dir)
        recording_id = int(metadata["manifest_recording_id"])
        metadata_split = str(metadata.get("dataset_split", split))
        if metadata_split != split:
            raise RuntimeError(f"{run_dir.name}: expected split {split}, found {metadata_split}")
        speaker_id = str(metadata.get("dataset_speaker_id", "")).zfill(2)
        if speaker_id not in allowed_speakers:
            raise RuntimeError(f"{run_dir.name}: speaker {speaker_id} is not in the {split} split")
        seen_speakers.add(speaker_id)
        digits = events.loc[events["event_type"].eq("letter")].sort_values("start_s")
        if digits.empty:
            raise RuntimeError(f"{run_dir.name}: no digit events")
        for event in digits.itertuples(index=False):
            label = int(event.label)
            if label not in range(10):
                raise RuntimeError(f"{run_dir.name}: invalid digit label {event.label}")
            start_s = max(float(times[0]), float(event.start_s) - context_s)
            end_s = min(float(times[-1]) + MODEL_HOP_MS / 1000.0, float(event.end_s) + context_s)
            mask = (times >= start_s) & (times < end_s)
            if mask.sum() < 10:
                raise RuntimeError(f"{run_dir.name}: digit {label} has fewer than 10 model steps")
            repetition = int(getattr(event, "source_repetition"))
            examples.append(
                DigitExample(
                    values=features[mask],
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
                    "input_steps_100hz": int(mask.sum()),
                    "features_per_step": len(FEATURE_NAMES),
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
        print(f"loaded {split} [{run_index:03d}/{len(run_dirs):03d}] {run_dir.name}")
    if not seen_speakers.issubset(allowed_speakers):
        raise RuntimeError(f"Unexpected {split} speakers: {sorted(seen_speakers - allowed_speakers)}")
    return examples, pd.DataFrame(audit_rows)


def fit_normalizer(examples: list[DigitExample]) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("Cannot fit a normalizer without training examples")
    count = 0
    total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    total_square = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for example in examples:
        values = example.values.astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class AudioMnistDataset(Dataset):
    def __init__(self, examples: list[DigitExample], mean: np.ndarray, std: np.ndarray) -> None:
        self.examples = examples
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        example = self.examples[index]
        values = (example.values - self.mean) / self.std
        return torch.from_numpy(values.astype(np.float32)), example.label, index


def collate_batch(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels, indices = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = torch.zeros(len(sequences), int(lengths.max()), sequences[0].shape[1])
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return (
        padded,
        lengths,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
    )


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.convolution(functional.pad(values, (self.left_padding, 0)))


class CausalDigitClassifier(nn.Module):
    def __init__(
        self,
        input_features: int = len(FEATURE_NAMES),
        hidden_size: int = 96,
        layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            CausalConv1d(input_features, 64, kernel_size=5),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(64, 96, kernel_size=5),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.recurrent = nn.GRU(
            96,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 10)

    def encode(self, values: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.frontend(values.transpose(1, 2)).transpose(1, 2)
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.recurrent(packed)
        return hidden[-1], features

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        representation, _ = self.encode(values, lengths)
        return self.classifier(representation)


def prediction_rows(
    model: nn.Module,
    dataset: AudioMnistDataset,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for values, lengths, labels, indices in loader:
            logits = model(values.to(device), lengths)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for label, prediction, probability, example_index in zip(
                labels.numpy(), predictions, probabilities, indices.numpy()
            ):
                example = dataset.examples[int(example_index)]
                rows.append(
                    {
                        "split": "",
                        "run": example.run,
                        "recording_id": example.recording_id,
                        "speaker_id": example.speaker_id,
                        "repetition": example.repetition,
                        "true_digit": int(label),
                        "predicted_digit": int(prediction),
                        "correct": int(label) == int(prediction),
                        "confidence": float(probability[int(prediction)]),
                        "input_steps_100hz": len(example.values),
                    }
                )
    return pd.DataFrame(rows)
