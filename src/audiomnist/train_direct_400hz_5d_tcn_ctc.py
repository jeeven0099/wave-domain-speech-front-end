#!/usr/bin/env python3
"""Train the direct-400 Hz five-resonator causal TCN-CTC on AudioMNIST."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
ALPHABET_CODE = HERE.parent / "alphabet_daq_experiment"
if str(ALPHABET_CODE) not in sys.path:
    sys.path.insert(0, str(ALPHABET_CODE))

from calibrate_resonator_ctc_decoder import apply_blank_penalty  # noqa: E402
from direct_400hz_ctc import (  # noqa: E402
    DIRECT_INPUT_FEATURES,
    DIRECT_INPUT_HOP_MS,
    DIRECT_INPUT_RATE_HZ,
    Direct400CausalConvTcnCtc,
    load_direct_capture,
)
from resonator_streaming_ctc import alignment_counts, edit_alignment  # noqa: E402


CAPTURE_ROOT = HERE / "captures"
STIMULI_ROOT = HERE / "stimuli"
OUTPUT_ROOT = HERE / "models" / "direct_400hz_5d_tcn_ctc"
SENSORS = ("R300", "R3500", "R7500", "R1800", "R4500")
FEATURE_NAMES = tuple(f"{sensor}_abs_direct_ac_v" for sensor in SENSORS)
FEATURE_CONTRACT = "audiomnist_direct_magnitude_5d_400hz_no_filter_causal_tcn_ctc_v1"

BLANK = 0
DIGIT_TO_INDEX = {str(digit): digit + 1 for digit in range(10)}
INDEX_TO_DIGIT = {index: digit for digit, index in DIGIT_TO_INDEX.items()}
VOCABULARY_SIZE = 11
FRONTEND_STRIDE = 4
MODEL_OUTPUT_RATE_HZ = DIRECT_INPUT_RATE_HZ / FRONTEND_STRIDE

# This is the c32/fk32/long model selected in the alphabet experiment.
MODEL_CHANNELS = 32
FRONTEND_KERNEL_SIZE = 32
TCN_KERNEL_SIZE = 3
DILATIONS = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class DirectDigitExample:
    values: np.ndarray
    label: int
    speaker_id: str
    repetition: int
    run: str
    recording_id: int
    start_s: float
    end_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--stimuli-root", type=Path, default=STIMULI_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--context-ms", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--blank-penalty", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.minimum_epochs > args.epochs:
        raise ValueError("--minimum-epochs cannot exceed --epochs")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("Batch size and epochs must be positive")
    if args.context_ms < 0:
        raise ValueError("--context-ms cannot be negative")
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def allowed_speakers(protocol_path: Path, split: str) -> set[str]:
    protocol = json.loads(protocol_path.read_text(encoding="ascii"))
    return {str(value).zfill(2) for value in protocol["speaker_splits"][split]}


def expected_recording_ids(manifest_path: Path) -> set[int]:
    manifest = pd.read_csv(manifest_path, usecols=["recording_id"])
    return {int(value) for value in manifest["recording_id"]}


def capture_directories(
    capture_root: Path,
    manifest_path: Path,
    require_complete: bool,
) -> list[Path]:
    expected = expected_recording_ids(manifest_path)
    by_recording: dict[int, Path] = {}
    for metadata_path in sorted(capture_root.rglob("metadata.json")):
        run_dir = metadata_path.parent
        if not (run_dir / "raw_capture.npz").is_file():
            continue
        if not (run_dir / "aligned_events.csv").is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        raw_id = metadata.get("manifest_recording_id")
        if raw_id is None or int(raw_id) not in expected:
            continue
        recording_id = int(raw_id)
        if recording_id in by_recording:
            raise RuntimeError(
                f"Recording {recording_id} has duplicate captures: "
                f"{by_recording[recording_id]} and {run_dir}"
            )
        by_recording[recording_id] = run_dir
    missing = sorted(expected - set(by_recording))
    if require_complete and missing:
        preview = ", ".join(map(str, missing[:12]))
        suffix = "..." if len(missing) > 12 else ""
        raise RuntimeError(f"Missing {len(missing)} captures: {preview}{suffix}")
    return [by_recording[key] for key in sorted(by_recording)]


def load_cached_direct_capture(
    run_dir: Path,
    cache_root: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
    recording_id = int(metadata["manifest_recording_id"])
    raw_path = run_dir / "raw_capture.npz"
    raw_stat = raw_path.stat()
    cache_dir = cache_root / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{recording_id:04d}_{run_dir.name}_direct400.npz"

    if cache_path.is_file():
        try:
            with np.load(cache_path) as cached:
                size_ok = int(cached["raw_size_bytes"]) == raw_stat.st_size
                time_ok = int(cached["raw_mtime_ns"]) == raw_stat.st_mtime_ns
                if size_ok and time_ok:
                    times = np.asarray(cached["times_s"], dtype=np.float64)
                    values = np.asarray(cached["magnitude_v"], dtype=np.float32)
                    events = pd.read_csv(run_dir / "aligned_events.csv")
                    return times, values, events, metadata
        except (OSError, ValueError, KeyError):
            pass

    times, signed_values, events, metadata = load_direct_capture(run_dir)
    magnitude = np.abs(signed_values).astype(np.float32)
    np.savez_compressed(
        cache_path,
        times_s=times,
        magnitude_v=magnitude,
        raw_size_bytes=np.int64(raw_stat.st_size),
        raw_mtime_ns=np.int64(raw_stat.st_mtime_ns),
    )
    return times, magnitude, events, metadata


def build_split(
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
        times, values, events, metadata = load_cached_direct_capture(
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
                float(times[-1]) + 1.0 / DIRECT_INPUT_RATE_HZ,
                float(event.end_s) + context_s,
            )
            first = int(np.searchsorted(times, start_s, side="left"))
            last = int(np.searchsorted(times, end_s, side="left"))
            segment = values[first:last].copy()
            if len(segment) < FRONTEND_KERNEL_SIZE:
                raise RuntimeError(
                    f"{run_dir.name}: digit {label} has only {len(segment)} direct samples"
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
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
        print(
            f"direct {split} [{run_index:03d}/{len(run_dirs):03d}] {run_dir.name}",
            flush=True,
        )

    if seen_speakers != permitted_speakers and require_complete:
        missing = sorted(permitted_speakers - seen_speakers)
        raise RuntimeError(f"Missing {split} speakers: {missing}")
    return examples, pd.DataFrame(audit_rows), len(run_dirs)


def fit_normalizer(examples: list[DirectDigitExample]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(DIRECT_INPUT_FEATURES, dtype=np.float64)
    total_square = np.zeros(DIRECT_INPUT_FEATURES, dtype=np.float64)
    for example in examples:
        values = example.values.astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
    if count == 0:
        raise ValueError("Cannot fit normalization without training examples")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-10)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class DirectDigitDataset(Dataset):
    def __init__(
        self,
        examples: list[DirectDigitExample],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.examples = examples
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        values = (example.values - self.mean) / self.std
        return torch.from_numpy(values.astype(np.float32)), example.label, index


def collate_batch(batch):
    sequences, labels, indices = zip(*batch)
    input_lengths = torch.tensor([len(item) for item in sequences], dtype=torch.long)
    padded = torch.zeros(
        len(sequences), int(input_lengths.max()), DIRECT_INPUT_FEATURES
    )
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    targets = torch.tensor(
        [DIGIT_TO_INDEX[str(label)] for label in labels], dtype=torch.long
    )
    target_lengths = torch.ones(len(labels), dtype=torch.long)
    return (
        padded,
        targets,
        input_lengths,
        target_lengths,
        torch.tensor(indices, dtype=torch.long),
    )


class Direct400DigitTcnCtc(Direct400CausalConvTcnCtc):
    """Alphabet-selected direct-400 Hz TCN with a digit CTC output head."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(
            channels=MODEL_CHANNELS,
            frontend_kernel_size=FRONTEND_KERNEL_SIZE,
            frontend_stride=FRONTEND_STRIDE,
            tcn_kernel_size=TCN_KERNEL_SIZE,
            dilations=DILATIONS,
            dropout=dropout,
        )
        self.classifier = nn.Conv1d(MODEL_CHANNELS, VOCABULARY_SIZE, 1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def learned_macs_per_output_frame() -> int:
    frontend = (
        DIRECT_INPUT_FEATURES * FRONTEND_KERNEL_SIZE
        + DIRECT_INPUT_FEATURES * MODEL_CHANNELS
    )
    blocks = len(DILATIONS) * (
        MODEL_CHANNELS * TCN_KERNEL_SIZE + MODEL_CHANNELS * MODEL_CHANNELS
    )
    classifier = MODEL_CHANNELS * VOCABULARY_SIZE
    return int(frontend + blocks + classifier)


def greedy_decode(log_probs: np.ndarray, length: int) -> str:
    indices = np.asarray(log_probs[:length]).argmax(axis=1)
    output: list[str] = []
    previous = BLANK
    for raw_index in indices:
        index = int(raw_index)
        if index != BLANK and index != previous:
            output.append(INDEX_TO_DIGIT[index])
        previous = index
    return "".join(output)


def add_beam_score(
    beams: dict[str, tuple[float, float]],
    prefix: str,
    blank_score: float | None = None,
    nonblank_score: float | None = None,
) -> None:
    old_blank, old_nonblank = beams.get(prefix, (-math.inf, -math.inf))
    if blank_score is not None:
        old_blank = float(np.logaddexp(old_blank, blank_score))
    if nonblank_score is not None:
        old_nonblank = float(np.logaddexp(old_nonblank, nonblank_score))
    beams[prefix] = old_blank, old_nonblank


def ctc_prefix_beam_decode(
    log_probs: np.ndarray,
    length: int,
    beam_width: int,
) -> str:
    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}
    for frame in np.asarray(log_probs[:length], dtype=np.float64):
        next_beams: dict[str, tuple[float, float]] = {}
        for prefix, (prob_blank, prob_nonblank) in beams.items():
            total = float(np.logaddexp(prob_blank, prob_nonblank))
            add_beam_score(
                next_beams, prefix, blank_score=total + float(frame[BLANK])
            )
            final_digit = prefix[-1] if prefix else None
            for index in range(1, len(frame)):
                digit = INDEX_TO_DIGIT[index]
                digit_score = float(frame[index])
                if digit == final_digit:
                    add_beam_score(
                        next_beams,
                        prefix,
                        nonblank_score=prob_nonblank + digit_score,
                    )
                    add_beam_score(
                        next_beams,
                        prefix + digit,
                        nonblank_score=prob_blank + digit_score,
                    )
                else:
                    add_beam_score(
                        next_beams,
                        prefix + digit,
                        nonblank_score=total + digit_score,
                    )
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: float(np.logaddexp(item[1][0], item[1][1])),
                reverse=True,
            )[:beam_width]
        )
    return max(
        beams.items(),
        key=lambda item: float(np.logaddexp(item[1][0], item[1][1])),
    )[0]


def score_prediction(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    return alignment_counts(edit_alignment(reference, hypothesis))


def run_loss_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.CTCLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for values, targets, input_lengths, target_lengths, _ in loader:
        values = values.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        log_probs, output_lengths = model(values, input_lengths)
        loss = loss_function(
            log_probs.float().transpose(0, 1),
            targets,
            output_lengths,
            target_lengths,
        )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = len(values)
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_decoder(
    model: nn.Module,
    dataset: DirectDigitDataset,
    loader: DataLoader,
    device: torch.device,
    decoder: str,
    beam_width: int = 20,
    blank_penalty: float = 0.25,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, object]] = []
    correct = substitutions = deletions = insertions = 0
    for values, _, input_lengths, _, indices in loader:
        log_probs, output_lengths = model(values.to(device), input_lengths)
        probabilities = log_probs.cpu().numpy()
        for batch_index, example_index in enumerate(indices.tolist()):
            example = dataset.examples[example_index]
            length = int(output_lengths[batch_index])
            frame_probs = probabilities[batch_index, :length]
            if decoder == "beam":
                frame_probs = apply_blank_penalty(frame_probs, blank_penalty)
                hypothesis = ctc_prefix_beam_decode(
                    frame_probs, length, beam_width=beam_width
                )
            elif decoder == "greedy":
                hypothesis = greedy_decode(frame_probs, length)
            else:
                raise ValueError(f"Unknown decoder: {decoder}")
            reference = str(example.label)
            match, substitution, deletion, insertion = score_prediction(
                reference, hypothesis
            )
            correct += int(hypothesis == reference)
            substitutions += substitution
            deletions += deletion
            insertions += insertion
            rows.append(
                {
                    "decoder": decoder,
                    "run": example.run,
                    "recording_id": example.recording_id,
                    "speaker_id": example.speaker_id,
                    "repetition": example.repetition,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "exact_match": hypothesis == reference,
                    "substitutions": substitution,
                    "deletions": deletion,
                    "insertions": insertion,
                }
            )
    count = len(dataset)
    errors = substitutions + deletions + insertions
    metrics: dict[str, float | int] = {
        "examples": count,
        "exact_digit_accuracy": correct / max(count, 1),
        "digit_error_rate": errors / max(count, 1),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
    return metrics, pd.DataFrame(rows)


def validate_model_code() -> None:
    model = Direct400DigitTcnCtc()
    values = torch.randn(2, 401, DIRECT_INPUT_FEATURES)
    lengths = torch.tensor([401, 377], dtype=torch.long)
    log_probs, output_lengths = model(values, lengths)
    expected_lengths = torch.tensor([101, 95], dtype=torch.long)
    if log_probs.shape != (2, 101, VOCABULARY_SIZE):
        raise RuntimeError(f"Unexpected model output shape: {tuple(log_probs.shape)}")
    if not torch.equal(output_lengths, expected_lengths):
        raise RuntimeError(f"Unexpected output lengths: {output_lengths.tolist()}")
    print(
        json.dumps(
            {
                "status": "ok",
                "input_shape": list(values.shape),
                "output_shape": list(log_probs.shape),
                "output_lengths": output_lengths.tolist(),
                "parameters": parameter_count(model),
                "model_macs_per_audio_second": (
                    learned_macs_per_output_frame() * int(MODEL_OUTPUT_RATE_HZ)
                ),
                "causal": True,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.validate_code:
        validate_model_code()
        return

    set_reproducible(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output_root.resolve()
    summary_path = output_root / "frozen_training_validation_summary.json"
    if summary_path.is_file() and not args.force:
        print("A frozen result already exists. Use --force only for a deliberate retrain.")
        print(summary_path.read_text(encoding="ascii"))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "direct_400hz_capture_cache"

    print("Loading AudioMNIST training and validation; test remains unopened.")
    training, training_audit, training_captures = build_split(
        "train",
        args.capture_root.resolve(),
        args.stimuli_root.resolve(),
        cache_root,
        args.context_ms,
        require_complete=not args.allow_incomplete,
    )
    validation, validation_audit, validation_captures = build_split(
        "validation",
        args.capture_root.resolve(),
        args.stimuli_root.resolve(),
        cache_root,
        args.context_ms,
        require_complete=not args.allow_incomplete,
    )
    mean, std = fit_normalizer(training)
    np.savez(output_root / "train_only_normalizer.npz", mean=mean, std=std)
    pd.concat([training_audit, validation_audit], ignore_index=True).to_csv(
        output_root / "training_validation_example_audit.csv", index=False
    )

    preparation = {
        "analysis_role": "training and validation only; held-out test not opened",
        "feature_contract": FEATURE_CONTRACT,
        "source_capture_rate_hz": 48000.0,
        "emulated_direct_sampling_rate_hz_per_sensor": DIRECT_INPUT_RATE_HZ,
        "physical_sensor_count": DIRECT_INPUT_FEATURES,
        "input_features_per_2p5ms_step": DIRECT_INPUT_FEATURES,
        "input_scalar_rate_per_second": int(
            DIRECT_INPUT_FEATURES * DIRECT_INPUT_RATE_HZ
        ),
        "feature_names": FEATURE_NAMES,
        "magnitude_operation": True,
        "fixed_rms_or_average_used": False,
        "anti_alias_filter_used": False,
        "analog_envelope_detector_used": False,
        "training_captures": training_captures,
        "validation_captures": validation_captures,
        "training_examples": len(training),
        "validation_examples": len(validation),
        "training_speakers": len(set(training_audit["speaker_id"].astype(str))),
        "validation_speakers": len(set(validation_audit["speaker_id"].astype(str))),
        "training_digit_counts": {
            str(key): int(value)
            for key, value in training_audit["digit"].value_counts().sort_index().items()
        },
        "validation_digit_counts": {
            str(key): int(value)
            for key, value in validation_audit["digit"].value_counts().sort_index().items()
        },
        "normalizer_source": "training frames only",
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "known_digit_boundaries_used_to_form_isolated_examples": True,
        "heldout_test_accessed": False,
    }
    (output_root / "input_preparation_summary.json").write_text(
        json.dumps(preparation, indent=2) + "\n", encoding="ascii"
    )
    if args.prepare_only:
        print(json.dumps(preparation, indent=2))
        return

    training_dataset = DirectDigitDataset(training, mean, std)
    validation_dataset = DirectDigitDataset(validation, mean, std)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = Direct400DigitTcnCtc(args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    loss_function = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    checkpoint_path = output_root / "best_direct_400hz_5d_tcn_ctc.pt"
    history: list[dict[str, object]] = []
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    parameters = parameter_count(model)
    macs_per_frame = learned_macs_per_output_frame()
    started = time.time()
    print(
        f"Training 5D direct-400 Hz TCN-CTC on {device}; "
        f"parameters={parameters:,}; model MAC/s="
        f"{macs_per_frame * int(MODEL_OUTPUT_RATE_HZ):,}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        training_loss = run_loss_epoch(
            model, training_loader, loss_function, device, optimizer
        )
        validation_loss = run_loss_epoch(
            model, validation_loader, loss_function, device
        )
        greedy_metrics, greedy_predictions = evaluate_decoder(
            model, validation_dataset, validation_loader, device, decoder="greedy"
        )
        validation_accuracy = float(greedy_metrics["exact_digit_accuracy"])
        scheduler.step(validation_loss)
        improved = validation_accuracy > best_accuracy + 1e-8 or (
            math.isclose(validation_accuracy, best_accuracy, abs_tol=1e-8)
            and validation_loss < best_loss
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "direct_400hz_c32_fk32_long_tcn_digit_ctc",
                    "model_config": {
                        "channels": MODEL_CHANNELS,
                        "frontend_kernel_size": FRONTEND_KERNEL_SIZE,
                        "frontend_stride": FRONTEND_STRIDE,
                        "tcn_kernel_size": TCN_KERNEL_SIZE,
                        "dilations": DILATIONS,
                        "dropout": args.dropout,
                        "classes": VOCABULARY_SIZE,
                        "blank_index": BLANK,
                        "causal": True,
                    },
                    "feature_contract": FEATURE_CONTRACT,
                    "feature_names": FEATURE_NAMES,
                    "input_mean": mean,
                    "input_std": std,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_greedy_accuracy": validation_accuracy,
                    "test_accessed": False,
                },
                checkpoint_path,
            )
            greedy_predictions.to_csv(
                output_root / "best_validation_greedy_predictions.csv", index=False
            )
        elif epoch >= args.minimum_epochs:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "training_ctc_loss": training_loss,
                "validation_ctc_loss": validation_loss,
                "validation_greedy_exact_digit_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected_checkpoint": improved,
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
        )
        pd.DataFrame(history).to_csv(output_root / "training_history.csv", index=False)
        print(
            f"epoch {epoch:03d}: train={training_loss:.4f} "
            f"valid={validation_loss:.4f} "
            f"greedy_accuracy={100.0 * validation_accuracy:.2f}%",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            print(f"Early stopping; selected epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    beam_metrics, beam_predictions = evaluate_decoder(
        model,
        validation_dataset,
        validation_loader,
        device,
        decoder="beam",
        beam_width=args.beam_width,
        blank_penalty=args.blank_penalty,
    )
    beam_predictions.to_csv(
        output_root / "validation_beam_predictions.csv", index=False
    )

    summary = {
        **preparation,
        "model": (
            "causal depthwise Conv1D stride 4 plus seven-block causal "
            "depthwise TCN plus digit CTC"
        ),
        "model_channels": MODEL_CHANNELS,
        "frontend_kernel_size_direct_samples": FRONTEND_KERNEL_SIZE,
        "frontend_causal_history_ms": (
            (FRONTEND_KERNEL_SIZE - 1) * DIRECT_INPUT_HOP_MS
        ),
        "frontend_stride": FRONTEND_STRIDE,
        "model_output_rate_hz": MODEL_OUTPUT_RATE_HZ,
        "tcn_kernel_size": TCN_KERNEL_SIZE,
        "dilations": DILATIONS,
        "temporal_receptive_field_direct_samples": model.receptive_field_direct_samples,
        "temporal_receptive_field_ms": (
            model.receptive_field_direct_samples * DIRECT_INPUT_HOP_MS
        ),
        "future_lookahead_ms": 0.0,
        "parameters": parameters,
        "learned_macs_per_10ms_output_frame": macs_per_frame,
        "learned_macs_per_audio_second": (
            macs_per_frame * int(MODEL_OUTPUT_RATE_HZ)
        ),
        "best_epoch": best_epoch,
        "best_validation_ctc_loss": best_loss,
        "best_validation_greedy_exact_digit_accuracy": best_accuracy,
        "validation_beam_width": args.beam_width,
        "validation_blank_penalty": args.blank_penalty,
        "validation_beam_exact_digit_accuracy": beam_metrics[
            "exact_digit_accuracy"
        ],
        "validation_beam_digit_error_rate": beam_metrics["digit_error_rate"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_size_mib": checkpoint_path.stat().st_size / (1024**2),
        "checkpoint_selection": "highest validation greedy accuracy, then CTC loss",
        "decoder": (
            f"CTC prefix beam width {args.beam_width}, blank penalty "
            f"{args.blank_penalty:g}, no language model"
        ),
        "test_status": "not opened by this trainer",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print("\nFrozen training/validation result:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
