#!/usr/bin/env python3
"""Spectral-channel ablation: train the three literature-minimal CNN baselines
on the single broadband 400 Hz RMS envelope, completing the nine-architecture
set next to the six digital_envelope_400hz models.

The data path and training recipe are copied from
scripts/create_audiomnist_400hz_compact_colab_notebook.py, which produced the
six existing broadband checkpoints: peak normalization to -9 dBFS, centre
crop/pad to 1 s at 48 kHz, 400 trailing 10 ms RMS values (2.5 ms hop), one
training-split scalar mean/std, seed 310, batch 256, AdamW (1e-3, wd 1e-4),
ReduceLROnPlateau (0.5, patience 3, min 1e-5), gradient clipping 5, up to 50
epochs with early stopping after 8 stale epochs once 20 have run, checkpoint
selection by validation accuracy then loss.

Stages:
    --stage train      build the envelope cache (train + validation only) and train
    --stage export-pi  copy the frozen checkpoints and the Pi runner into
                       raspberry_pi_audiomnist_benchmark_bundle; the held-out test
                       runs once on the Pi via run_broadband_cnn.sh
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from broadband_cnn_models import BROADBAND_CNN_FACTORIES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATASET_ROOT = REPO / "datasets" / "AudioMNIST" / "data"
OUTPUT_ROOT = HERE / "models" / "broadband_envelope_400hz_cnn_baselines"
PI_BUNDLE_ROOT = REPO / "raspberry_pi_audiomnist_benchmark_bundle"
PI_MODEL_DIR = PI_BUNDLE_ROOT / "models" / "digital_envelope_400hz"
REFERENCE_CHECKPOINT = PI_MODEL_DIR / "tc_resnet8_1d" / "best_validation_checkpoint.pt"

SEED = 310
INPUT_RATE_HZ = 400
INPUT_SAMPLES = 400
SOURCE_RATE_HZ = 48_000
WINDOW_SAMPLES = 480
TARGET_PEAK_DBFS = -9.0
BATCH_SIZE = 256
EPOCHS = 50
MINIMUM_EPOCHS = 20
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SPLIT_SPEAKERS = {
    "train": {28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2,
              8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55},
    "validation": {12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("train", "export-pi"), required=True)
    parser.add_argument("--model-name", choices=(*BROADBAND_CNN_FACTORIES, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rms_envelope_400hz(path: Path) -> np.ndarray:
    values, source_rate = sf.read(path, always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if source_rate != SOURCE_RATE_HZ:
        raise ValueError(f"Expected {SOURCE_RATE_HZ} Hz WAV, got {source_rate} Hz: {path}")
    peak = float(np.max(np.abs(values)))
    if peak > 1e-12:
        values = (values * (10.0 ** (TARGET_PEAK_DBFS / 20.0) / peak)).astype(np.float32)
    target = SOURCE_RATE_HZ
    if len(values) > target:
        start = (len(values) - target) // 2
        fixed = values[start:start + target]
    else:
        left = (target - len(values)) // 2
        fixed = np.pad(values, (left, target - len(values) - left))
    end_indices = np.rint(
        (np.arange(INPUT_SAMPLES, dtype=np.float64) + 1.0) * SOURCE_RATE_HZ / INPUT_RATE_HZ
    ).astype(np.int64)
    end_indices = np.clip(end_indices, 1, len(fixed))
    start_indices = np.maximum(0, end_indices - WINDOW_SAMPLES)
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(fixed, dtype=np.float64))))
    energy = cumulative[end_indices] - cumulative[start_indices]
    return np.sqrt(np.maximum(energy, 0.0) / WINDOW_SAMPLES).astype(np.float32)


def load_envelopes() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Train and validation envelopes only; test WAVs are never read here."""
    cache_path = OUTPUT_ROOT / "envelope_cache" / "train_validation_rms10ms_hop2p5ms_v1.npz"
    rows = []
    for wav_path in sorted(DATASET_ROOT.glob("*/*.wav")):
        digit, speaker, repetition = (int(v) for v in wav_path.stem.split("_"))
        split = next((n for n, members in SPLIT_SPEAKERS.items() if speaker in members), None)
        if split is not None:
            rows.append((wav_path, digit, speaker, split))
    counts = pd.Series([r[3] for r in rows]).value_counts()
    if counts.get("train") != 18_000 or counts.get("validation") != 6_000:
        raise RuntimeError(f"unexpected split sizes: {counts.to_dict()}")
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            envelopes = np.asarray(cached["envelopes"], dtype=np.float32)
    else:
        envelopes = np.empty((len(rows), INPUT_SAMPLES), dtype=np.float32)
        for index, row in enumerate(rows):
            envelopes[index] = rms_envelope_400hz(row[0])
            if (index + 1) % 2000 == 0:
                print(f"envelopes {index + 1}/{len(rows)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, envelopes=envelopes)
    if envelopes.shape != (len(rows), INPUT_SAMPLES) or not np.isfinite(envelopes).all():
        raise RuntimeError("invalid envelope cache")
    labels = np.asarray([r[1] for r in rows], dtype=np.int64)
    splits = np.asarray([r[3] for r in rows])
    indices = {name: np.flatnonzero(splits == name) for name in SPLIT_SPEAKERS}
    return envelopes, labels, indices


class EnvelopeDataset(Dataset):
    def __init__(self, envelopes, labels, indices, mean, std):
        self.envelopes, self.labels, self.indices = envelopes, labels, indices
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        values = (self.envelopes[index] - self.mean) / self.std
        return torch.from_numpy(values[:, None].astype(np.float32)), int(self.labels[index])


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = correct = top3 = count = 0
    for values, labels in loader:
        values, labels = values.to(device), labels.to(device)
        logits = model(values)
        loss += float(criterion(logits, labels))
        correct += int(logits.argmax(1).eq(labels).sum())
        top3 += int((logits.topk(3, 1).indices == labels[:, None]).any(1).sum())
        count += len(labels)
    return loss / count, correct / count, top3 / count


def macs_one_second(model: nn.Module) -> int:
    macs = 0

    def hook(module, _inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            macs += int(output.numel() * math.prod(module.kernel_size) * module.in_channels / module.groups)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, INPUT_SAMPLES, 1, device=next(model.parameters()).device))
    for handle in handles:
        handle.remove()
    return macs


def train_one(name, envelopes, labels, indices, contract, force):
    model_dir = OUTPUT_ROOT / f"{name}_1d"
    summary_path = model_dir / "training_summary.json"
    if summary_path.is_file() and not force:
        print(f"[{name}] already frozen; skipping (use --force to retrain)")
        return json.loads(summary_path.read_text())
    model_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mean = np.float32(envelopes[indices["train"]].mean())
    train_std = np.float32(max(envelopes[indices["train"]].std(), 1e-6))
    train_loader = DataLoader(
        EnvelopeDataset(envelopes, labels, indices["train"], train_mean, train_std),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        EnvelopeDataset(envelopes, labels, indices["validation"], train_mean, train_std),
        batch_size=BATCH_SIZE, pin_memory=device.type == "cuda",
    )
    model = BROADBAND_CNN_FACTORIES[name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = model_dir / "best_validation_checkpoint.pt"
    best_accuracy, best_loss, best_epoch, stale = -1.0, math.inf, 0, 0
    history = []
    started = time.time()
    print(f"[{name}] training on {device}; params={sum(p.numel() for p in model.parameters()):,}; "
          f"train mean/std={float(train_mean):.9f}/{float(train_std):.9f}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_count = 0
        for values, batch_labels in train_loader:
            values, batch_labels = values.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss += float(loss.detach()) * len(batch_labels)
            tr_correct += int(logits.argmax(1).eq(batch_labels).sum())
            tr_count += len(batch_labels)
        va_loss, va_acc, va_top3 = evaluate(model, validation_loader, device)
        scheduler.step(va_loss)
        improved = va_acc > best_accuracy + 1e-8 or (
            math.isclose(va_acc, best_accuracy, abs_tol=1e-8) and va_loss < best_loss
        )
        if improved:
            best_accuracy, best_loss, best_epoch, stale = va_acc, va_loss, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": f"{name}_1d",
                "epoch": epoch,
                "validation_accuracy": best_accuracy,
                "validation_loss": best_loss,
                "train_mean": train_mean,
                "train_std": train_std,
                "experiment_contract": contract,
                "test_accessed": False,
            }, checkpoint_path)
        elif epoch >= MINIMUM_EPOCHS:
            stale += 1
        history.append({
            "epoch": epoch, "training_loss": tr_loss / tr_count,
            "training_accuracy": tr_correct / tr_count, "validation_loss": va_loss,
            "validation_accuracy": va_acc, "validation_top3_accuracy": va_top3,
            "learning_rate": optimizer.param_groups[0]["lr"], "selected_checkpoint": improved,
        })
        pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
        print(f"[{name}] epoch {epoch:03d}: train={tr_loss / tr_count:.4f} valid={va_loss:.4f} "
              f"top1={100 * va_acc:.2f}% top3={100 * va_top3:.2f}%", flush=True)
        if epoch >= MINIMUM_EPOCHS and stale >= PATIENCE:
            break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BROADBAND_CNN_FACTORIES[name]()
    model.load_state_dict(checkpoint["model_state"])
    summary = {
        "model": f"{name}_1d",
        "analysis_role": "spectral-channel ablation; training and validation model selection only",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs_per_1s_input": macs_one_second(model),
        "input_rate_hz": INPUT_RATE_HZ,
        "input_features_per_step": 1,
        "input_representation": "causal_10ms_rms_envelope_at_400hz",
        "causal": True,
        "test_accessed": False,
        "training_minutes": (time.time() - started) / 60.0,
        "recipe": "scripts/create_audiomnist_400hz_compact_colab_notebook.py (seed 310)",
        "checkpoint": str(checkpoint_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{name}] frozen: epoch {best_epoch}, val top1 {100 * best_accuracy:.2f}%", flush=True)
    return summary


def export_pi(names) -> None:
    for name in names:
        source = OUTPUT_ROOT / f"{name}_1d"
        if not (source / "training_summary.json").is_file():
            print(f"[export] NOT exported (not trained): {name}")
            continue
        target = PI_MODEL_DIR / f"{name}_1d"
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("best_validation_checkpoint.pt", "training_summary.json"):
            shutil.copy2(source / filename, target / filename)
        print(f"[export] {name} -> {target}")
    for filename in ("broadband_cnn_models.py", "benchmark_broadband_cnn_stage.py", "run_broadband_cnn.sh"):
        shutil.copy2(HERE / filename, PI_BUNDLE_ROOT / filename)
        print(f"[export] {filename}")


def main() -> None:
    args = parse_args()
    names = tuple(BROADBAND_CNN_FACTORIES) if args.model_name == "all" else (args.model_name,)
    if args.stage == "export-pi":
        export_pi(names)
        return
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = reference["experiment_contract"]
    envelopes, labels, indices = load_envelopes()
    train_mean = float(envelopes[indices["train"]].mean())
    # The six Colab models were normalized with this same training-split mean;
    # a match confirms the local envelopes reproduce their input exactly.
    print(f"train mean {train_mean:.9f} vs reference {float(reference['train_mean']):.9f}; "
          f"std {float(envelopes[indices['train']].std()):.9f} vs {float(reference['train_std']):.9f}",
          flush=True)
    if not math.isclose(train_mean, float(reference["train_mean"]), rel_tol=1e-4):
        raise RuntimeError("local broadband envelopes do not match the six reference models")
    for name in names:
        train_one(name, envelopes, labels, indices, contract, args.force)


if __name__ == "__main__":
    main()
