#!/usr/bin/env python3
"""Train the fabricated Pooled TCN on direct one-channel 400 Hz AudioMNIST waveforms."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import wavfile
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from train_direct_400hz_5d_pooled_tcn_classifier import (
    CLASSES,
    DILATIONS,
    FRONTEND_KERNEL_SIZE,
    FRONTEND_STRIDE,
    MODEL_CHANNELS,
    MODEL_OUTPUT_RATE_HZ,
    TCN_KERNEL_SIZE,
    Direct400PooledTcnClassifier,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "AudioMNIST"
DEFAULT_PROTOCOL = HERE / "stimuli" / "protocol.json"
DEFAULT_OUTPUT_ROOT = HERE / "models" / "raw_waveform_400hz_pooled_tcn"

INPUT_RATE_HZ = 400
INPUT_SAMPLES = 400
SOURCE_RATE_HZ = 48_000
TARGET_PEAK_DBFS = -9.0
INPUT_FEATURES = 1
CACHE_VERSION = "raw_signed_waveform_400hz_antialiased_fixed_1s_peak_minus9dbfs_v1"
ARCHITECTURE_ID = "raw400_1ch_c32_fk32_long_tcn_stats_pool_softmax"
CHECKPOINT_FILENAME = "best_raw_waveform_400hz_pooled_tcn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=100)
    parser.add_argument("--latency-repeats", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.minimum_epochs > args.epochs:
        raise ValueError("--minimum-epochs cannot exceed --epochs")
    if min(args.batch_size, args.epochs, args.patience) <= 0:
        raise ValueError("Batch size, epochs, and patience must be positive")
    if args.num_workers < 0 or args.latency_warmup < 0 or args.latency_repeats <= 0:
        raise ValueError("Invalid worker or latency-repeat count")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def protocol_splits(path: Path) -> dict[int, str]:
    protocol = json.loads(path.read_text(encoding="ascii"))
    mapping: dict[int, str] = {}
    for split, speakers in protocol["speaker_splits"].items():
        for speaker in speakers:
            speaker_id = int(speaker)
            if speaker_id in mapping:
                raise RuntimeError(f"Speaker {speaker_id} occurs in multiple splits")
            mapping[speaker_id] = split
    if set(mapping) != set(range(1, 61)):
        raise RuntimeError("Protocol must assign all 60 AudioMNIST speakers exactly once")
    return mapping


def inventory(dataset_root: Path, protocol_path: Path) -> pd.DataFrame:
    mapping = protocol_splits(protocol_path)
    rows = []
    for path in sorted((dataset_root / "data").rglob("*.wav")):
        try:
            digit_text, speaker_text, repetition_text = path.stem.split("_")
            digit = int(digit_text)
            speaker = int(speaker_text)
            repetition = int(repetition_text)
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"Unexpected AudioMNIST filename: {path}") from error
        rows.append(
            {
                "path": str(path.resolve()),
                "digit": digit,
                "speaker": speaker,
                "repetition": repetition,
                "split": mapping[speaker],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["split", "speaker", "digit", "repetition"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != 30_000:
        raise RuntimeError(f"Expected 30,000 AudioMNIST WAVs, found {len(frame):,}")
    if not frame.groupby(["speaker", "digit"]).size().eq(50).all():
        raise RuntimeError("Expected 50 repetitions per speaker and digit")
    expected = {"train": 18_000, "validation": 6_000, "test": 6_000}
    if frame.groupby("split").size().to_dict() != expected:
        raise RuntimeError("AudioMNIST split counts do not match the 36/12/12 protocol")
    return frame


def _pcm_to_float(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        values = values.astype(np.float64).mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        scale = float(max(abs(info.min), info.max))
        return values.astype(np.float32) / scale
    return values.astype(np.float32)


def prepare_waveform(path: Path) -> np.ndarray:
    source_rate, values = wavfile.read(path)
    if int(source_rate) != SOURCE_RATE_HZ:
        raise RuntimeError(f"{path}: expected {SOURCE_RATE_HZ} Hz, found {source_rate} Hz")
    values = _pcm_to_float(values)
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    if peak > 1e-12:
        values = values * (10.0 ** (TARGET_PEAK_DBFS / 20.0) / peak)

    divisor = math.gcd(int(source_rate), INPUT_RATE_HZ)
    values = signal.resample_poly(
        values,
        INPUT_RATE_HZ // divisor,
        int(source_rate) // divisor,
    ).astype(np.float32)
    if len(values) > INPUT_SAMPLES:
        start = (len(values) - INPUT_SAMPLES) // 2
        values = values[start : start + INPUT_SAMPLES]
    elif len(values) < INPUT_SAMPLES:
        missing = INPUT_SAMPLES - len(values)
        left = missing // 2
        values = np.pad(values, (left, missing - left))
    if values.shape != (INPUT_SAMPLES,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid prepared waveform: {path}")
    return values


def _protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_or_build_split(
    records: pd.DataFrame,
    split: str,
    cache_root: Path,
    protocol_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = records.loc[records["split"].eq(split)].reset_index(drop=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{split}_{CACHE_VERSION}.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if (
                    str(cached["cache_version"].item()) == CACHE_VERSION
                    and str(cached["protocol_sha256"].item()) == _protocol_hash(protocol_path)
                    and int(cached["examples"].item()) == len(subset)
                ):
                    return (
                        np.asarray(cached["waveforms"], dtype=np.float32),
                        np.asarray(cached["labels"], dtype=np.int64),
                        np.asarray(cached["speakers"], dtype=np.int64),
                        np.asarray(cached["repetitions"], dtype=np.int64),
                    )
        except (OSError, ValueError, KeyError):
            pass

    waveforms = np.empty((len(subset), INPUT_SAMPLES), dtype=np.float32)
    for index, row in enumerate(subset.itertuples(index=False), start=0):
        waveforms[index] = prepare_waveform(Path(row.path))
        if (index + 1) % 1000 == 0 or index + 1 == len(subset):
            print(f"Preparing {split}: {index + 1:,}/{len(subset):,}", flush=True)
    labels = subset["digit"].to_numpy(np.int64)
    speakers = subset["speaker"].to_numpy(np.int64)
    repetitions = subset["repetition"].to_numpy(np.int64)
    np.savez_compressed(
        cache_path,
        waveforms=waveforms,
        labels=labels,
        speakers=speakers,
        repetitions=repetitions,
        cache_version=np.asarray(CACHE_VERSION),
        protocol_sha256=np.asarray(_protocol_hash(protocol_path)),
        examples=np.int64(len(subset)),
    )
    return waveforms, labels, speakers, repetitions


class WaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray, mean: float, std: float):
        self.waveforms = waveforms
        self.labels = labels
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        values = ((self.waveforms[index] - self.mean) / self.std).astype(np.float32)
        return torch.from_numpy(values[:, None]), int(self.labels[index]), index


class Raw400PooledTcnClassifier(Direct400PooledTcnClassifier):
    """The fabricated Pooled TCN with a one-channel input stem."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(dropout=dropout)
        self.frontend_depthwise = nn.Conv1d(
            INPUT_FEATURES,
            INPUT_FEATURES,
            FRONTEND_KERNEL_SIZE,
            stride=FRONTEND_STRIDE,
            groups=INPUT_FEATURES,
        )
        self.frontend_pointwise = nn.Conv1d(INPUT_FEATURES, MODEL_CHANNELS, 1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def analytical_macs_per_one_second() -> int:
    encoder_per_output = (
        INPUT_FEATURES * FRONTEND_KERNEL_SIZE
        + INPUT_FEATURES * MODEL_CHANNELS
        + len(DILATIONS)
        * (MODEL_CHANNELS * TCN_KERNEL_SIZE + MODEL_CHANNELS * MODEL_CHANNELS)
    )
    return int(encoder_per_output * MODEL_OUTPUT_RATE_HZ + 2 * MODEL_CHANNELS * CLASSES)


@torch.inference_mode()
def profiled_macs_per_one_second(model: nn.Module) -> int:
    total = 0
    handles = []

    def conv_hook(module: nn.Conv1d, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        kernel_work = module.kernel_size[0] * module.in_channels // module.groups
        total += output.numel() * kernel_work

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        model.eval()(torch.zeros(1, INPUT_SAMPLES, 1), torch.tensor([INPUT_SAMPLES]))
    finally:
        for handle in handles:
            handle.remove()
    return int(total)


def collate_batch(batch):
    values, labels, indices = zip(*batch)
    return (
        torch.stack(values),
        torch.full((len(values),), INPUT_SAMPLES, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_top3 = total_examples = 0
    for values, lengths, labels, _ in loader:
        values = values.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(values, lengths)
        loss = F.cross_entropy(logits, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = len(labels)
        total_loss += float(loss.detach().cpu()) * count
        total_correct += int(logits.argmax(1).eq(labels).sum().detach().cpu())
        total_top3 += int(
            logits.topk(3, dim=1).indices.eq(labels[:, None]).any(dim=1).sum().detach().cpu()
        )
        total_examples += count
    denominator = max(total_examples, 1)
    return total_loss / denominator, total_correct / denominator, total_top3 / denominator


@torch.inference_mode()
def evaluate_predictions(
    model: nn.Module,
    dataset: WaveformDataset,
    loader: DataLoader,
    speakers: np.ndarray,
    repetitions: np.ndarray,
    device: torch.device,
) -> tuple[float, float, float, pd.DataFrame]:
    loss, accuracy, top3_accuracy = run_epoch(model, loader, device)
    model.eval()
    rows = []
    for values, lengths, labels, indices in loader:
        probabilities = torch.softmax(model(values.to(device), lengths.to(device)), dim=1).cpu()
        predictions = probabilities.argmax(1)
        top3 = probabilities.topk(3, dim=1).indices
        for label, prediction, choices, confidence, index in zip(
            labels.tolist(), predictions.tolist(), top3.tolist(),
            probabilities.max(1).values.tolist(), indices.tolist()
        ):
            rows.append(
                {
                    "speaker": int(speakers[index]),
                    "repetition": int(repetitions[index]),
                    "reference": label,
                    "prediction": prediction,
                    "top1_correct": prediction == label,
                    "top3_correct": label in choices,
                    "top3_predictions": " ".join(map(str, choices)),
                    "prediction_confidence": confidence,
                }
            )
    return loss, accuracy, top3_accuracy, pd.DataFrame(rows)


@torch.inference_mode()
def measure_cpu_latency(
    model: nn.Module, warmup: int, repeats: int
) -> tuple[float, float, str]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model.cpu().eval()
    values = torch.zeros(1, INPUT_SAMPLES, INPUT_FEATURES)
    lengths = torch.tensor([INPUT_SAMPLES])
    for _ in range(warmup):
        model(values, lengths)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = time.perf_counter_ns()
        model(values, lengths)
        samples[index] = (time.perf_counter_ns() - start) / 1e6
    torch.set_num_threads(previous_threads)
    device_name = platform.processor() or platform.machine()
    return float(np.median(samples)), float(np.percentile(samples, 95)), device_name


def serialized_state_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def validate_model_code() -> dict[str, object]:
    model = Raw400PooledTcnClassifier().eval()
    sample = torch.randn(2, INPUT_SAMPLES, INPUT_FEATURES)
    lengths = torch.tensor([INPUT_SAMPLES, 377])
    with torch.inference_mode():
        logits = model(sample, lengths)
        frames, output_lengths = model.encode_frames(sample, lengths)
        changed = sample.clone()
        changed[:, 200:] += 5.0 * torch.randn_like(changed[:, 200:])
        changed_frames, _ = model.encode_frames(changed, lengths)
    unaffected = math.ceil(200 / FRONTEND_STRIDE)
    causal_error = float(
        (frames[:, :, :unaffected] - changed_frames[:, :, :unaffected]).abs().max()
    )
    profiled_macs = profiled_macs_per_one_second(model)
    analytical_macs = analytical_macs_per_one_second()
    if logits.shape != (2, CLASSES) or output_lengths.tolist() != [100, 95]:
        raise RuntimeError("Raw-waveform TCN shape audit failed")
    if causal_error > 1e-5:
        raise RuntimeError(f"Raw-waveform TCN causality audit failed: {causal_error}")
    if profiled_macs != analytical_macs:
        raise RuntimeError(f"MAC audit failed: profiled={profiled_macs}, analytical={analytical_macs}")
    return {
        "status": "ok",
        "input_shape": [1, INPUT_SAMPLES, INPUT_FEATURES],
        "output_shape": [1, CLASSES],
        "parameters": parameter_count(model),
        "fp32_parameter_bytes": 4 * parameter_count(model),
        "profiled_macs_per_one_second": profiled_macs,
        "analytical_macs_per_one_second": analytical_macs,
        "causal_future_invariance_max_abs_error": causal_error,
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    audit = validate_model_code()
    if args.validate_code:
        print(json.dumps(audit, indent=2))
        return

    set_reproducible(args.seed)
    device = select_device(args.device)
    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    summary_path = output_root / "heldout_test_summary.json"
    if summary_path.is_file() and not args.force:
        print("A completed result already exists. Use --force only for a deliberate retrain.")
        print(summary_path.read_text(encoding="ascii"))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "cache"
    records = inventory(dataset_root, protocol_path)

    print("Preparing training and validation waveforms; held-out test audio remains unopened.")
    train_wave, train_labels, train_speakers, train_repetitions = load_or_build_split(
        records, "train", cache_root, protocol_path
    )
    valid_wave, valid_labels, valid_speakers, valid_repetitions = load_or_build_split(
        records, "validation", cache_root, protocol_path
    )
    if args.prepare_only:
        print("Training and validation caches are ready; test remains unopened.")
        return

    normalizer_mean = float(train_wave.mean(dtype=np.float64))
    normalizer_std = float(train_wave.std(dtype=np.float64))
    if normalizer_std < 1e-8:
        raise RuntimeError("Training waveform standard deviation is too small")
    np.savez(
        output_root / "train_only_normalizer.npz",
        mean=np.float32(normalizer_mean),
        std=np.float32(normalizer_std),
    )
    training_dataset = WaveformDataset(train_wave, train_labels, normalizer_mean, normalizer_std)
    validation_dataset = WaveformDataset(valid_wave, valid_labels, normalizer_mean, normalizer_std)
    training_loader = make_loader(
        training_dataset, args.batch_size, True, device, args.num_workers, args.seed
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, device, args.num_workers, args.seed
    )

    model = Raw400PooledTcnClassifier(args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    checkpoint_path = output_root / CHECKPOINT_FILENAME
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.time()
    print(
        f"Training raw 400 Hz Pooled TCN on {device}; parameters={parameter_count(model):,}; "
        f"MAC/s={analytical_macs_per_one_second():,}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_top3 = run_epoch(
            model, training_loader, device, optimizer
        )
        valid_loss, valid_accuracy, valid_top3 = run_epoch(model, validation_loader, device)
        scheduler.step(valid_loss)
        improved = valid_accuracy > best_accuracy + 1e-8 or (
            math.isclose(valid_accuracy, best_accuracy, abs_tol=1e-8) and valid_loss < best_loss
        )
        if improved:
            best_accuracy = valid_accuracy
            best_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": ARCHITECTURE_ID,
                    "model_config": {
                        "input_features": INPUT_FEATURES,
                        "channels": MODEL_CHANNELS,
                        "frontend_kernel_size": FRONTEND_KERNEL_SIZE,
                        "frontend_stride": FRONTEND_STRIDE,
                        "tcn_kernel_size": TCN_KERNEL_SIZE,
                        "dilations": DILATIONS,
                        "dropout": args.dropout,
                        "classes": CLASSES,
                        "pooling": "masked running mean plus running maximum",
                    },
                    "feature_contract": CACHE_VERSION,
                    "input_mean": normalizer_mean,
                    "input_std": normalizer_std,
                    "epoch": epoch,
                    "validation_loss": valid_loss,
                    "validation_top1_accuracy": valid_accuracy,
                    "validation_top3_accuracy": valid_top3,
                    "test_accessed": False,
                },
                checkpoint_path,
            )
        elif epoch >= args.minimum_epochs:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": train_loss,
                "training_top1_accuracy": train_accuracy,
                "training_top3_accuracy": train_top3,
                "validation_loss": valid_loss,
                "validation_top1_accuracy": valid_accuracy,
                "validation_top3_accuracy": valid_top3,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected_checkpoint": improved,
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
        )
        pd.DataFrame(history).to_csv(output_root / "training_history.csv", index=False)
        print(
            f"epoch {epoch:03d}: train={train_loss:.4f} top1={100*train_accuracy:.2f}% "
            f"top3={100*train_top3:.2f}% | valid={valid_loss:.4f} "
            f"top1={100*valid_accuracy:.2f}% top3={100*valid_top3:.2f}%",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            print(f"Early stopping; selected epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    print("Checkpoint frozen. Opening the held-out AudioMNIST test speakers now.", flush=True)
    test_wave, test_labels, test_speakers, test_repetitions = load_or_build_split(
        records, "test", cache_root, protocol_path
    )
    test_dataset = WaveformDataset(test_wave, test_labels, normalizer_mean, normalizer_std)
    test_loader = make_loader(
        test_dataset, args.batch_size, False, device, args.num_workers, args.seed
    )
    test_loss, test_accuracy, test_top3, predictions = evaluate_predictions(
        model, test_dataset, test_loader, test_speakers, test_repetitions, device
    )
    predictions.to_csv(output_root / "heldout_test_predictions.csv", index=False)
    predictions.groupby("reference", as_index=False).agg(
        examples=("top1_correct", "size"),
        top1_accuracy=("top1_correct", "mean"),
        top3_accuracy=("top3_correct", "mean"),
    ).to_csv(output_root / "heldout_test_per_digit.csv", index=False)
    predictions.groupby("speaker", as_index=False).agg(
        examples=("top1_correct", "size"),
        top1_accuracy=("top1_correct", "mean"),
        top3_accuracy=("top3_correct", "mean"),
    ).to_csv(output_root / "heldout_test_per_speaker.csv", index=False)

    model = model.cpu().eval()
    profiled_macs = profiled_macs_per_one_second(model)
    median_latency, p95_latency, latency_cpu = measure_cpu_latency(
        model, args.latency_warmup, args.latency_repeats
    )
    state_size = serialized_state_size(model)
    summary = {
        "analysis_role": "frozen speaker-disjoint held-out AudioMNIST test",
        "experiment": "fabricated Pooled TCN without resonators on direct raw 400 Hz waveform",
        "input_representation": "one signed anti-aliased 400 Hz waveform channel",
        "source_rate_hz": SOURCE_RATE_HZ,
        "input_rate_hz": INPUT_RATE_HZ,
        "fixed_input_seconds": 1.0,
        "offline_experimental_resampling": "scipy.signal.resample_poly from 48 kHz to 400 Hz",
        "deployment_interpretation": "a real 400 Hz ADC path requires an analog anti-alias filter",
        "peak_normalization_dbfs": TARGET_PEAK_DBFS,
        "normalizer_source": "training speakers only",
        "speaker_split": "same fixed 36 train / 12 validation / 12 test speakers as fabricated experiment",
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset),
        "selected_epoch": int(checkpoint["epoch"]),
        "best_validation_top1_accuracy": float(checkpoint["validation_top1_accuracy"]),
        "test_loss": test_loss,
        "test_top1_accuracy": test_accuracy,
        "test_top1_accuracy_percent": 100.0 * test_accuracy,
        "test_top3_accuracy": test_top3,
        "parameters": parameter_count(model),
        "fp32_parameter_size_bytes": 4 * parameter_count(model),
        "fp32_parameter_size_kib": 4 * parameter_count(model) / 1024.0,
        "serialized_state_dict_size_bytes": state_size,
        "serialized_state_dict_size_kib": state_size / 1024.0,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_size_kib": checkpoint_path.stat().st_size / 1024.0,
        "profiled_macs_per_one_second_utterance": profiled_macs,
        "analytical_macs_per_one_second_utterance": analytical_macs_per_one_second(),
        "learned_flops_per_one_second_utterance_2_per_mac": 2 * profiled_macs,
        "model_latency_ms_median_cpu_one_thread_batch1": median_latency,
        "model_latency_ms_p95_cpu_one_thread_batch1": p95_latency,
        "latency_device": latency_cpu,
        "latency_torch_threads": 1,
        "latency_warmup": args.latency_warmup,
        "latency_repeats": args.latency_repeats,
        "latency_file_io_resampling_and_normalization_included": False,
        "torch_version": torch.__version__,
        "architecture_difference_from_fabricated_pooled_tcn": (
            "input stem changed from five channels to one; all channels, kernels, strides, "
            "dilations, causal blocks, pooling, and classifier dimensions retained"
        ),
        "future_lookahead_ms": 0.0,
        "test_used_for_checkpoint_selection": False,
        "checkpoint": str(checkpoint_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print("\nHeld-out result and deployment metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
