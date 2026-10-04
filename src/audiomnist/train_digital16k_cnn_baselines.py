#!/usr/bin/env python3
"""Train each of the three basic/minimal CNN baselines (cnn_one_fstride4,
tiny_conv, cnn_trad_fpool3 -- see cnn_baseline_models.py) on the 16 kHz
digital_16k log-mel representation, using the exact frontend spec frozen in
benchmark_audiomnist_raspberry_pi.py (LogMelFrontend / preprocess_logmel):
48 kHz -> 16 kHz polyphase resample, fp16 waveform round-trip, peak-normalize
to -9 dBFS, center crop/pad to 1.0 s, torchaudio MelSpectrogram(n_fft=800,
win_length=800, hop_length=160, n_mels=40, htk, f_min=20/f_max=8000),
log-compress, fp16 feature round-trip, per-mel-bin train-only normalization.

No standalone digital_16k trainer existed before this file -- the six
existing digital_16k baselines were only ever trained via the Colab notebook
generator (scripts/create_audiomnist_compact_model_benchmark_colab.py). This
script is written from scratch, reusing only the frozen frontend constants
and the speaker split in stimuli/protocol.json, so results stay comparable.

Speakers come from protocol.json's speaker_splits; only train/validation are
ever loaded here (test_policy: "Do not capture or evaluate the test split
until the model and decoder are frozen" -- respected by never touching the
"test" speaker list in this file).

One architecture is trained per invocation, selected with --model-name from
{cnn_one_fstride4, tiny_conv, cnn_trad_fpool3}, or --model-name all.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cnn_baseline_models import ARCHITECTURE_IDS, DIGITAL16K_CNN_FACTORIES

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "stimuli" / "protocol.json"
CACHE_ROOT = HERE / "cache" / "digital16k_logmel"
OUTPUT_ROOT = HERE / "models" / "digital16k_cnn_baselines"
FEATURE_CONTRACT_TEMPLATE = (
    "audiomnist_digital_16k_logmel_40mel_96frame_noncausal_basic_cnn_"
    "{model_name}_softmax_v1"
)

SOURCE_SAMPLE_RATE = 48_000
TARGET_SAMPLE_RATE = 16_000
TARGET_SAMPLES = 16_000
TARGET_PEAK_DBFS = -9.0
N_MELS = 40
COMPACT_FRAMES = 96
CLASSES = 10
MODEL_NAMES = tuple(DIGITAL16K_CNN_FACTORIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=(*MODEL_NAMES, "all"), required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=HERE.parent.parent / "datasets" / "AudioMNIST",
    )
    parser.add_argument("--protocol-path", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--validate-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.minimum_epochs > args.epochs:
        raise ValueError("--minimum-epochs cannot exceed --epochs")
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Batch size, epochs, and patience must be positive")


def set_reproducible(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _speakers_for_split(protocol_path: Path, split_name: str) -> list[int]:
    document = json.loads(protocol_path.read_text(encoding="utf-8"))
    speakers = document["speaker_splits"][split_name]
    return sorted(int(value) for value in speakers)


def _resample_poly(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return values.astype(np.float32, copy=False)
    from scipy import signal

    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(
        values, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def _peak_normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    if peak <= 1e-12:
        return values.astype(np.float32, copy=False)
    target = 10.0 ** (TARGET_PEAK_DBFS / 20.0)
    return np.asarray(values * (target / peak), dtype=np.float32)


def _crop_or_pad(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) > target_length:
        start = (len(values) - target_length) // 2
        return np.asarray(values[start : start + target_length], dtype=np.float32)
    if len(values) < target_length:
        missing = target_length - len(values)
        return np.pad(values, (missing // 2, missing - missing // 2)).astype(np.float32)
    return values.astype(np.float32, copy=False)


def _raw_rows_for_speakers(dataset_root: Path, speakers: list[int]) -> list[dict[str, Any]]:
    speaker_set = set(speakers)
    rows = []
    for path in sorted((dataset_root / "data").glob("[0-9][0-9]/*.wav")):
        digit, speaker, repetition = (int(value) for value in path.stem.split("_"))
        if speaker in speaker_set:
            rows.append(
                {
                    "path": path,
                    "label": digit,
                    "speaker": speaker,
                    "repetition": repetition,
                }
            )
    if not rows:
        raise RuntimeError(f"No raw AudioMNIST wavs found under {dataset_root} for {speakers}")
    return rows


def _build_raw_logmel_cache(
    split_name: str, dataset_root: Path, protocol_path: Path, cache_root: Path
) -> Path:
    """Computes, caches, and returns the path to the un-normalized (but
    already fp16-round-tripped, per the frozen frontend spec) log-mel
    features for one split. Cached once per split so training all three
    architectures does not repeat 18k/6k wav loads + resamples + mel
    extractions three times over.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{split_name}_logmel_raw.npz"
    if cache_path.is_file():
        return cache_path

    import soundfile as sf
    import torchaudio

    speakers = _speakers_for_split(protocol_path, split_name)
    rows = _raw_rows_for_speakers(dataset_root, speakers)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=800,
        win_length=800,
        hop_length=160,
        f_min=20.0,
        f_max=8_000.0,
        n_mels=N_MELS,
        power=2.0,
        center=False,
        norm=None,
        mel_scale="htk",
    ).cpu().eval()

    features = np.empty((len(rows), COMPACT_FRAMES, N_MELS), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    speaker_ids = np.empty(len(rows), dtype=np.int64)
    repetitions = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        values, source_rate = sf.read(row["path"], always_2d=False, dtype="float32")
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 2:
            values = values.mean(axis=1, dtype=np.float32)
        values = _resample_poly(values, int(source_rate), TARGET_SAMPLE_RATE)
        values = values.astype(np.float16).astype(np.float32)
        values = _crop_or_pad(_peak_normalize(values), TARGET_SAMPLES)
        waveform = torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0)
        with torch.inference_mode():
            feature = torch.log(mel(waveform).clamp_min(1e-8)).transpose(1, 2)
        feature = feature.to(torch.float16).to(torch.float32)
        features[index] = feature.squeeze(0).numpy()
        labels[index] = row["label"]
        speaker_ids[index] = row["speaker"]
        repetitions[index] = row["repetition"]
        if (index + 1) % 2000 == 0:
            print(f"[{split_name}] featurized {index + 1}/{len(rows)}", flush=True)

    np.savez(
        cache_path,
        features=features,
        labels=labels,
        speakers=speaker_ids,
        repetitions=repetitions,
    )
    return cache_path


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.reshape(-1, N_MELS).mean(axis=0)
    std = features.reshape(-1, N_MELS).std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


class Digital16kDigitDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        normalized = (features - mean.reshape(1, 1, N_MELS)) / std.reshape(1, 1, N_MELS)
        self.features = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[index], self.labels[index], index


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_top3 = total_examples = 0
    for values, labels, _ in loader:
        values = values.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        loss = F.cross_entropy(logits, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = len(labels)
        total_loss += float(loss.detach().cpu()) * count
        total_correct += int(logits.argmax(dim=1).eq(labels).sum().detach().cpu())
        total_top3 += int(
            logits.topk(3, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).sum().detach().cpu()
        )
        total_examples += count
    denominator = max(total_examples, 1)
    return total_loss / denominator, total_correct / denominator, total_top3 / denominator


@torch.inference_mode()
def prediction_rows(
    model: nn.Module,
    split_table: pd.DataFrame,
    dataset: Digital16kDigitDataset,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for values, labels, indices in loader:
        logits = model(values.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu()
        top3 = probabilities.topk(3, dim=1).indices
        predictions = probabilities.argmax(dim=1)
        for label, prediction, choices, probability, example_index in zip(
            labels.tolist(), predictions.tolist(), top3.tolist(), probabilities.tolist(), indices.tolist()
        ):
            meta = split_table.iloc[example_index]
            rows.append(
                {
                    "speaker_id": int(meta["speakers"]),
                    "repetition": int(meta["repetitions"]),
                    "reference": label,
                    "prediction": prediction,
                    "top1_correct": prediction == label,
                    "top3_predictions": " ".join(map(str, choices)),
                    "top3_correct": label in choices,
                    "prediction_confidence": probability[prediction],
                }
            )
    return pd.DataFrame(rows)


def validate_model_code(model_name: str) -> None:
    model = DIGITAL16K_CNN_FACTORIES[model_name]().eval()
    dummy = torch.randn(3, COMPACT_FRAMES, N_MELS)
    with torch.inference_mode():
        logits = model(dummy)
    if logits.shape != (3, CLASSES):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    print(
        json.dumps(
            {
                "model_name": model_name,
                "status": "ok",
                "input_shape": list(dummy.shape),
                "output_shape": list(logits.shape),
                "parameters": parameter_count(model),
            },
            indent=2,
        )
    )


def train_one(model_name: str, args: argparse.Namespace) -> dict[str, object]:
    set_reproducible(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = (args.output_root / model_name).resolve()
    checkpoint_filename = f"best_digital16k_basic_cnn_{model_name}_classifier.pt"
    architecture_id = ARCHITECTURE_IDS[model_name]
    feature_contract = FEATURE_CONTRACT_TEMPLATE.format(model_name=model_name)
    summary_path = output_root / "frozen_training_validation_summary.json"
    if summary_path.is_file() and not args.force:
        print(f"[{model_name}] A frozen result already exists. Use --force only for a deliberate retrain.")
        print(summary_path.read_text(encoding="ascii"))
        return json.loads(summary_path.read_text(encoding="ascii"))
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol_path.resolve()
    cache_root = args.cache_root.resolve()
    print(f"[{model_name}] Loading AudioMNIST training and validation (16 kHz digital_16k); test remains unopened.", flush=True)
    training_cache = _build_raw_logmel_cache("train", dataset_root, protocol_path, cache_root)
    validation_cache = _build_raw_logmel_cache("validation", dataset_root, protocol_path, cache_root)
    training_archive = np.load(training_cache)
    validation_archive = np.load(validation_cache)

    mean, std = fit_normalizer(training_archive["features"])
    np.savez(output_root / "train_only_normalizer.npz", mean=mean, std=std)

    training_table = pd.DataFrame(
        {
            "speakers": training_archive["speakers"],
            "repetitions": training_archive["repetitions"],
            "labels": training_archive["labels"],
        }
    )
    validation_table = pd.DataFrame(
        {
            "speakers": validation_archive["speakers"],
            "repetitions": validation_archive["repetitions"],
            "labels": validation_archive["labels"],
        }
    )
    training_table.to_csv(output_root / "training_example_audit.csv", index=False)
    validation_table.to_csv(output_root / "validation_example_audit.csv", index=False)

    training_dataset = Digital16kDigitDataset(training_archive["features"], training_archive["labels"], mean, std)
    validation_dataset = Digital16kDigitDataset(validation_archive["features"], validation_archive["labels"], mean, std)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = DIGITAL16K_CNN_FACTORIES[model_name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)
    checkpoint_path = output_root / checkpoint_filename
    parameters = parameter_count(model)
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    started = time.time()
    print(f"[{model_name}] Training on {device}; parameters={parameters:,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        training_loss, training_accuracy, training_top3 = run_epoch(model, training_loader, device, optimizer)
        validation_loss, validation_accuracy, validation_top3 = run_epoch(model, validation_loader, device)
        scheduler.step(validation_loss)
        improved = validation_accuracy > best_accuracy + 1e-8 or (
            math.isclose(validation_accuracy, best_accuracy, abs_tol=1e-8) and validation_loss < best_loss
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": architecture_id,
                    "model_config": {
                        "model_name": model_name,
                        "classes": CLASSES,
                        "causal_encoder": False,
                        "pooling": "global average pool over time and mel frequency",
                    },
                    "feature_contract": feature_contract,
                    "input_mean": mean,
                    "input_std": std,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_top1_accuracy": validation_accuracy,
                    "validation_top3_accuracy": validation_top3,
                    "test_accessed": False,
                },
                checkpoint_path,
            )
        elif epoch >= args.minimum_epochs:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "training_top1_accuracy": training_accuracy,
                "training_top3_accuracy": training_top3,
                "validation_loss": validation_loss,
                "validation_top1_accuracy": validation_accuracy,
                "validation_top3_accuracy": validation_top3,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected_checkpoint": improved,
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
        )
        pd.DataFrame(history).to_csv(output_root / "training_history.csv", index=False)
        print(
            f"[{model_name}] epoch {epoch:03d}: train={training_loss:.4f} "
            f"top1={100*training_accuracy:.2f}% top3={100*training_top3:.2f}% | "
            f"valid={validation_loss:.4f} top1={100*validation_accuracy:.2f}% top3={100*validation_top3:.2f}%",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            print(f"[{model_name}] Early stopping; selected epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    predictions = prediction_rows(model, validation_table, validation_dataset, validation_loader, device)
    predictions.to_csv(output_root / "validation_predictions.csv", index=False)
    per_digit = (
        predictions.groupby("reference", as_index=False)
        .agg(examples=("top1_correct", "size"), top1_accuracy=("top1_correct", "mean"), top3_accuracy=("top3_correct", "mean"))
        .sort_values("reference")
    )
    per_digit.to_csv(output_root / "validation_per_digit.csv", index=False)

    summary = {
        "model_name": model_name,
        "architecture": architecture_id,
        "analysis_role": "training and validation only; held-out test not opened",
        "feature_contract": feature_contract,
        "representation": "digital_16k",
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "n_mels": N_MELS,
        "frames": COMPACT_FRAMES,
        "causal_encoder": False,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "training_speakers": len(set(training_table["speakers"])),
        "validation_speakers": len(set(validation_table["speakers"])),
        "normalizer_source": "training frames only",
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "heldout_test_accessed": False,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_top1_accuracy": best_accuracy,
        "best_validation_top3_accuracy": float(checkpoint["validation_top3_accuracy"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_size_mib": checkpoint_path.stat().st_size / (1024**2),
        "checkpoint_selection": "highest validation top-1 accuracy, then loss",
        "test_status": "not opened by this trainer",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(f"\n[{model_name}] Frozen training/validation result:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_names = MODEL_NAMES if args.model_name == "all" else (args.model_name,)
    if args.validate_code:
        for model_name in model_names:
            validate_model_code(model_name)
        return
    for model_name in model_names:
        train_one(model_name, args)


if __name__ == "__main__":
    main()
