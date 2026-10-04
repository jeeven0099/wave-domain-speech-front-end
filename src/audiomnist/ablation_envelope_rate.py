#!/usr/bin/env python3
"""Envelope-rate ablation: retrain the nine fabricated-front-end classifiers on
five-channel AC-RMS envelopes sampled at 100 Hz instead of 400 Hz.

Only the envelope hop changes. The 10 ms AC-RMS window (480 raw samples at
48007 Hz), the 48 kHz resonator captures, the AudioMNIST split, the nine
architectures, and every training hyperparameter are identical to
train_fabricated_400hz_5d_baseline_classifiers.py and
train_fabricated_cnn_baselines.py.

    rate    hop (raw samples)   hop (ms)   window overlap
    100 Hz  480                 10.0       0 %
    400 Hz  120                  2.5      75 %   (existing frozen models)

Architectures are unchanged, so kernel lengths and strides are fixed in
envelope steps; their receptive fields in seconds scale with 1/rate.

Stages (run in order):
    --stage build      train/validation envelopes from raw_capture.npz into a cache
    --stage train      train + validation-based checkpoint selection
    --stage export-pi  write 100 Hz test envelopes and the frozen checkpoints into
                       raspberry_pi_audiomnist_benchmark_bundle; the held-out test
                       runs once on the Pi via run_ablation_100hz.sh
    --stage verify     check that the cache builder reproduces the stored 400 Hz files
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cnn_baseline_models import ARCHITECTURE_IDS as CNN_ARCHITECTURE_IDS
from cnn_baseline_models import FABRICATED_CNN_FACTORIES
from fabricated_400hz_5d_baseline_models import ARCHITECTURE_IDS as BASE_ARCHITECTURE_IDS
from fabricated_400hz_5d_baseline_models import FABRICATED_400_FACTORIES, INPUT_CHANNELS
from train_direct_400hz_5d_pooled_tcn_classifier import classifier_collate
from train_direct_400hz_5d_tcn_ctc import (
    HERE,
    STIMULI_ROOT,
    DirectDigitDataset,
    DirectDigitExample,
    allowed_speakers,
    capture_directories,
    fit_normalizer,
    set_reproducible,
)
from train_envelope_400hz_5d_pooled_tcn_classifier import (
    CACHE_ROOT as STORED_400_CACHE_ROOT,
    FEATURE_NAMES,
    load_cached_envelope_capture,
)
from train_fabricated_400hz_5d_baseline_classifiers import prediction_rows, run_epoch

CAPTURE_ROOT = HERE / "captures"
ABLATION_ROOT = HERE / "models" / "ablation_envelope_rate"
RMS_WINDOW_MS = 10.0
# Shortest admissible segment, in seconds: 32 steps at 400 Hz, as in the
# original 400 Hz loader, kept constant in time across rates.
MIN_SEGMENT_S = 32 / 400.0
ABLATION_RATES = (100,)
CLASSES = 10

FACTORIES = {**FABRICATED_400_FACTORIES, **FABRICATED_CNN_FACTORIES}
MODEL_NAMES = tuple(FACTORIES)
PI_BUNDLE_ROOT = HERE.parents[1] / "raspberry_pi_audiomnist_benchmark_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("build", "train", "export-pi", "verify"), required=True)
    parser.add_argument(
        "--rates", type=int, nargs="+", choices=ABLATION_RATES, default=list(ABLATION_RATES)
    )
    parser.add_argument("--model-name", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--stimuli-root", type=Path, default=STIMULI_ROOT)
    parser.add_argument("--output-root", type=Path, default=ABLATION_ROOT)
    parser.add_argument("--context-ms", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--pi-bundle-root", type=Path, default=PI_BUNDLE_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.minimum_epochs > args.epochs:
        raise ValueError("--minimum-epochs cannot exceed --epochs")
    return args


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


def frame_ac_rms(values: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Identical to alphabet_daq_experiment/build_400hz_envelopes.frame_rms,
    with the hop passed in instead of fixed at 400 Hz."""
    if len(values) < frame_length:
        return np.empty(0, dtype=np.float32)
    starts = np.arange(0, len(values) - frame_length + 1, hop_length, dtype=np.int64)
    ends = starts + frame_length
    values64 = np.asarray(values, dtype=np.float64)
    cumulative = np.empty(len(values64) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values64, out=cumulative[1:])
    sums = cumulative[ends] - cumulative[starts]
    np.square(values64, out=values64)
    np.cumsum(values64, out=cumulative[1:])
    sums_square = cumulative[ends] - cumulative[starts]
    variance = np.maximum(
        sums_square / frame_length - np.square(sums / frame_length), 0.0
    )
    return np.sqrt(variance).astype(np.float32)


def envelope_geometry(sample_rate_hz: float, rate_hz: int) -> tuple[int, int]:
    frame_length = int(round(sample_rate_hz * RMS_WINDOW_MS / 1000.0))
    hop_length = int(round(sample_rate_hz / rate_hz))
    return frame_length, hop_length


def cache_path_for(cache_root: Path, split: str, run_dir: Path, recording_id: int, rate_hz: int) -> Path:
    return cache_root / split / f"{recording_id:04d}_{run_dir.name}_rms_envelope{rate_hz}.npz"


def load_or_build_envelopes(
    run_dir: Path, cache_root: Path, split: str, rate_hz: int, build: bool
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Return (availability_times_s, envelope_v[N,5], events, metadata)."""
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
    recording_id = int(metadata["manifest_recording_id"])
    raw_path = run_dir / "raw_capture.npz"
    raw_stat = raw_path.stat()
    cache_path = cache_path_for(cache_root, split, run_dir, recording_id, rate_hz)
    events = pd.read_csv(run_dir / "aligned_events.csv")
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            if (
                int(cached["source_size_bytes"]) == raw_stat.st_size
                and int(cached["source_mtime_ns"]) == raw_stat.st_mtime_ns
                and int(cached["envelope_rate_hz"]) == rate_hz
            ):
                return (
                    np.asarray(cached["availability_times_s"], dtype=np.float64),
                    np.asarray(cached["envelope_v"], dtype=np.float32),
                    events,
                    metadata,
                )
    if not build:
        raise FileNotFoundError(
            f"{cache_path} is missing or stale; run --stage build first"
        )
    with np.load(raw_path, allow_pickle=False) as raw:
        volts = np.asarray(raw["volts_v"], dtype=np.float32)
        sample_rate_hz = float(raw["sample_rate_hz"])
    if volts.ndim != 2 or volts.shape[1] != INPUT_CHANNELS:
        raise RuntimeError(f"{run_dir.name}: expected five channels, found {volts.shape}")
    frame_length, hop_length = envelope_geometry(sample_rate_hz, rate_hz)
    values = np.column_stack(
        [frame_ac_rms(volts[:, c], frame_length, hop_length) for c in range(INPUT_CHANNELS)]
    )
    centers = (
        np.arange(len(values), dtype=np.float64) * hop_length + frame_length / 2.0
    ) / sample_rate_hz
    # A causal system can use each value only once its trailing 10 ms window
    # has ended, 5 ms after the window center (same rule as the 400 Hz loader).
    times = centers + RMS_WINDOW_MS / 2000.0
    if not np.isfinite(values).all() or values.min() < 0.0:
        raise RuntimeError(f"{run_dir.name}: invalid AC-RMS values")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        availability_times_s=times,
        envelope_v=values,
        envelope_rate_hz=np.int64(rate_hz),
        raw_samples_per_rms_window=np.int64(frame_length),
        raw_samples_per_envelope_hop=np.int64(hop_length),
        source_sample_rate_hz=np.float64(sample_rate_hz),
        source_size_bytes=np.int64(raw_stat.st_size),
        source_mtime_ns=np.int64(raw_stat.st_mtime_ns),
    )
    return times, values, events, metadata


def build_split(
    split: str,
    rate_hz: int,
    args: argparse.Namespace,
    build: bool = False,
) -> tuple[list[DirectDigitExample], pd.DataFrame, int]:
    """Segment isolated digits exactly as build_envelope_split does at 400 Hz."""
    stimuli_root = args.stimuli_root.resolve()
    permitted = allowed_speakers(stimuli_root / "protocol.json", split)
    run_dirs = capture_directories(
        args.capture_root.resolve() / split,
        stimuli_root / f"audiomnist_{split}_manifest.csv",
        require_complete=not args.allow_incomplete,
    )
    cache_root = args.output_root.resolve() / f"{rate_hz}hz" / "envelope_cache"
    context_s = args.context_ms / 1000.0
    min_steps = max(1, math.ceil(MIN_SEGMENT_S * rate_hz))
    examples: list[DirectDigitExample] = []
    audit_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for run_index, run_dir in enumerate(run_dirs, start=1):
        times, values, events, metadata = load_or_build_envelopes(
            run_dir, cache_root, split, rate_hz, build
        )
        if str(metadata.get("dataset_split", split)) != split:
            raise RuntimeError(f"{run_dir.name}: split mismatch")
        speaker_id = str(metadata.get("dataset_speaker_id", "")).zfill(2)
        if speaker_id not in permitted:
            raise RuntimeError(f"{run_dir.name}: speaker {speaker_id} not in {split}")
        seen.add(speaker_id)
        recording_id = int(metadata["manifest_recording_id"])
        digits = events.loc[events["event_type"].eq("letter")].sort_values("start_s")
        if digits.empty:
            raise RuntimeError(f"{run_dir.name}: no digit events")
        for event in digits.itertuples(index=False):
            label = int(event.label)
            if label not in range(CLASSES):
                raise RuntimeError(f"{run_dir.name}: invalid digit label {event.label}")
            start_s = max(float(times[0]), float(event.start_s) - context_s)
            end_s = min(float(times[-1]) + 1.0 / rate_hz, float(event.end_s) + context_s)
            first = int(np.searchsorted(times, start_s, side="left"))
            last = int(np.searchsorted(times, end_s, side="left"))
            segment = values[first:last].copy()
            if len(segment) < min_steps:
                raise RuntimeError(
                    f"{run_dir.name}: digit {label} has only {len(segment)} steps at {rate_hz} Hz"
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
                    "envelope_rate_hz": rate_hz,
                    "input_steps": len(segment),
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
        print(f"[{rate_hz} Hz] {split} [{run_index:03d}/{len(run_dirs):03d}] {run_dir.name}", flush=True)
    if seen != permitted and not args.allow_incomplete:
        raise RuntimeError(f"Missing {split} speakers: {sorted(permitted - seen)}")
    return examples, pd.DataFrame(audit_rows), len(run_dirs)


# ---------------------------------------------------------------------------
# Compute accounting
# ---------------------------------------------------------------------------


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def macs_one_second(model: nn.Module, rate_hz: int) -> int:
    """Learned MACs for a 1 s input, using the Raspberry Pi benchmark's
    hook definition (conv: out.numel * kernel * in/groups; linear: out * in)."""
    macs = 0

    def hook(module: nn.Module, _inputs, output) -> None:
        nonlocal macs
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            macs += int(tensor.numel() * math.prod(module.kernel_size) * module.in_channels / module.groups)
        elif isinstance(module, nn.Linear):
            macs += int(tensor.numel() * module.in_features)

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            model(
                torch.zeros(1, rate_hz, INPUT_CHANNELS, device=device),
                torch.tensor([rate_hz], dtype=torch.long),
            )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return macs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def architecture_id(model_name: str) -> str:
    return {**BASE_ARCHITECTURE_IDS, **CNN_ARCHITECTURE_IDS}[model_name]


def make_loader(dataset: DirectDigitDataset, args, device, shuffle: bool) -> DataLoader:
    kwargs = {}
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=classifier_collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
        **kwargs,
    )


def train_one(
    model_name: str,
    rate_hz: int,
    args: argparse.Namespace,
    training: list[DirectDigitExample],
    validation: list[DirectDigitExample],
    audit: pd.DataFrame,
) -> dict[str, object]:
    output_root = (args.output_root / f"{rate_hz}hz" / model_name).resolve()
    summary_path = output_root / "frozen_training_validation_summary.json"
    if summary_path.is_file() and not args.force:
        print(f"[{rate_hz} Hz/{model_name}] already frozen; skipping (use --force to retrain)")
        return json.loads(summary_path.read_text(encoding="ascii"))
    output_root.mkdir(parents=True, exist_ok=True)
    set_reproducible(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = fit_normalizer(training)
    np.savez(output_root / "train_only_normalizer.npz", mean=mean, std=std)
    audit.to_csv(output_root / "training_validation_example_audit.csv", index=False)
    training_dataset = DirectDigitDataset(training, mean, std)
    validation_dataset = DirectDigitDataset(validation, mean, std)
    training_loader = make_loader(training_dataset, args, device, shuffle=True)
    validation_loader = make_loader(validation_dataset, args, device, shuffle=False)

    model = FACTORIES[model_name]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    checkpoint_path = output_root / f"best_ablation_{rate_hz}hz_5d_{model_name}_classifier.pt"
    parameters = parameter_count(model)
    feature_contract = (
        f"audiomnist_ac_rms_envelope_5d_{rate_hz}hz_10ms_window_"
        f"{1000.0 / rate_hz:g}ms_hop_causal_{model_name}_masked_mean_softmax_ablation_v1"
    )
    best_accuracy, best_loss, best_epoch, stale = -math.inf, math.inf, 0, 0
    history: list[dict[str, object]] = []
    started = time.time()
    print(
        f"[{rate_hz} Hz/{model_name}] Training on {device}; parameters={parameters:,}; "
        f"total_stride={model.total_stride}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_top3 = run_epoch(model, training_loader, device, optimizer)
        va_loss, va_acc, va_top3 = run_epoch(model, validation_loader, device)
        scheduler.step(va_loss)
        improved = va_acc > best_accuracy + 1e-8 or (
            math.isclose(va_acc, best_accuracy, abs_tol=1e-8) and va_loss < best_loss
        )
        if improved:
            best_accuracy, best_loss, best_epoch, stale = va_acc, va_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": architecture_id(model_name),
                    "model_config": {
                        "model_name": model_name,
                        "classes": CLASSES,
                        "total_stride": model.total_stride,
                        "causal_encoder": True,
                        "pooling": "masked mean over per-frame class logits",
                    },
                    "envelope_rate_hz": rate_hz,
                    "feature_contract": feature_contract,
                    "feature_names": FEATURE_NAMES,
                    "input_mean": mean,
                    "input_std": std,
                    "context_ms": args.context_ms,
                    "epoch": epoch,
                    "validation_loss": va_loss,
                    "validation_top1_accuracy": va_acc,
                    "validation_top3_accuracy": va_top3,
                    "test_accessed": False,
                },
                checkpoint_path,
            )
        elif epoch >= args.minimum_epochs:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": tr_loss,
                "training_top1_accuracy": tr_acc,
                "training_top3_accuracy": tr_top3,
                "validation_loss": va_loss,
                "validation_top1_accuracy": va_acc,
                "validation_top3_accuracy": va_top3,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected_checkpoint": improved,
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
        )
        pd.DataFrame(history).to_csv(output_root / "training_history.csv", index=False)
        print(
            f"[{rate_hz} Hz/{model_name}] epoch {epoch:03d}: train={tr_loss:.4f} "
            f"top1={100 * tr_acc:.2f}% | valid={va_loss:.4f} top1={100 * va_acc:.2f}%",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale >= args.patience:
            print(f"[{rate_hz} Hz/{model_name}] Early stopping; selected epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    predictions = prediction_rows(model, validation_dataset, validation_loader, device)
    predictions.to_csv(output_root / "validation_predictions.csv", index=False)
    summary = {
        "model_name": model_name,
        "architecture": architecture_id(model_name),
        "analysis_role": "envelope-rate ablation; training and validation only",
        "feature_contract": feature_contract,
        "envelope_rate_hz_per_sensor": rate_hz,
        "rms_window_ms": RMS_WINDOW_MS,
        "envelope_hop_ms": 1000.0 / rate_hz,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "heldout_test_accessed": False,
        "total_stride": model.total_stride,
        "parameters": parameters,
        "macs_per_1s_input": macs_one_second(model, rate_hz),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_minutes": (time.time() - started) / 60.0,
        "best_validation_loss": best_loss,
        "best_validation_top1_accuracy": best_accuracy,
        "best_validation_top3_accuracy": float(checkpoint["validation_top3_accuracy"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": "highest validation top-1 accuracy, then loss",
        "hyperparameters": {
            k: getattr(args, k)
            for k in (
                "context_ms", "batch_size", "epochs", "minimum_epochs",
                "patience", "learning_rate", "weight_decay", "seed",
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(
        f"[{rate_hz} Hz/{model_name}] frozen: epoch {best_epoch}, "
        f"val top1 {100 * best_accuracy:.2f}%",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Raspberry Pi export (the held-out test is run on the Pi, never here)
# ---------------------------------------------------------------------------

PI_ENVELOPE_NPZ = "envelopes_100Hz_ablation_capture.npz"
PI_ENVELOPE_METADATA = "envelopes_100Hz_ablation_metadata.json"
PI_MODEL_SUBDIR = Path("models") / "ablation_envelope_100hz"


def export_pi(args: argparse.Namespace, model_names: tuple[str, ...]) -> None:
    """Write 100 Hz test envelopes (same layout as envelopes_400Hz_capture.npz)
    into the Pi bundle, and copy every frozen 100 Hz checkpoint next to them.
    Envelope extraction only transforms test waveforms; no test label is read
    and no model is evaluated here."""
    import shutil

    rate_hz = 100
    bundle = args.pi_bundle_root.resolve()
    bundle_test = bundle / "data" / "fabricated" / "captures" / "test"
    stimuli_root = args.stimuli_root.resolve()
    run_dirs = capture_directories(
        args.capture_root.resolve() / "test",
        stimuli_root / "audiomnist_test_manifest.csv",
        require_complete=True,
    )
    for index, run_dir in enumerate(run_dirs, start=1):
        target = bundle_test / run_dir.name
        if not target.is_dir():
            raise FileNotFoundError(f"Pi bundle has no test capture {target}")
        with np.load(run_dir / "raw_capture.npz", allow_pickle=False) as raw:
            volts = np.asarray(raw["volts_v"], dtype=np.float32)
            sample_rate_hz = float(raw["sample_rate_hz"])
            channel_names = np.asarray(raw["channel_names"])
        frame_length, hop_length = envelope_geometry(sample_rate_hz, rate_hz)
        values = np.column_stack(
            [frame_ac_rms(volts[:, c], frame_length, hop_length) for c in range(INPUT_CHANNELS)]
        )
        # Window-center timestamps, exactly as in envelopes_400Hz_capture.npz;
        # the Pi loader adds the 5 ms causal availability delay itself.
        centers = (
            np.arange(len(values), dtype=np.float64) * hop_length + frame_length / 2.0
        ) / sample_rate_hz
        np.savez(
            target / PI_ENVELOPE_NPZ,
            envelope_v=values,
            sample_rate_hz=np.float64(rate_hz),
            channel_names=channel_names,
            time_s=centers,
        )
        metadata = {
            "envelope_rate_hz": float(rate_hz),
            "rms_window_duration_ms": RMS_WINDOW_MS,
            "envelope_hop_duration_ms": 1000.0 / rate_hz,
            "window_overlap_fraction": 1.0 - hop_length / frame_length,
            "source_sample_rate_hz": sample_rate_hz,
            "raw_samples_per_rms_window": frame_length,
            "raw_samples_per_envelope_hop": hop_length,
            "envelope_rows": int(len(values)),
            "channel_count": INPUT_CHANNELS,
            "output_file": PI_ENVELOPE_NPZ,
            "purpose": "envelope-rate ablation (100 Hz); built by ablation_envelope_rate.py",
        }
        (target / PI_ENVELOPE_METADATA).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="ascii"
        )
        print(f"[export] test envelopes [{index:03d}/{len(run_dirs):03d}] {run_dir.name}", flush=True)

    missing = []
    for name in model_names:
        source_dir = args.output_root.resolve() / f"{rate_hz}hz" / name
        if not (source_dir / "frozen_training_validation_summary.json").is_file():
            missing.append(name)
            continue
        target_dir = bundle / PI_MODEL_SUBDIR / name
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            f"best_ablation_{rate_hz}hz_5d_{name}_classifier.pt",
            "frozen_training_validation_summary.json",
        ):
            shutil.copy2(source_dir / filename, target_dir / filename)
        print(f"[export] checkpoint {name}", flush=True)
    for filename in ("benchmark_ablation_100hz_stage.py", "run_ablation_100hz.sh"):
        shutil.copy2(HERE / filename, bundle / filename)
    if missing:
        print(f"[export] NOT exported (still training): {', '.join(missing)}")
    else:
        print(f"[export] all {len(model_names)} checkpoints exported to {bundle / PI_MODEL_SUBDIR}")


# ---------------------------------------------------------------------------


def verify_400(args: argparse.Namespace) -> None:
    """The builder at 400 Hz must reproduce the stored 400 Hz envelopes."""
    stimuli_root = args.stimuli_root.resolve()
    run_dir = capture_directories(
        args.capture_root.resolve() / "train",
        stimuli_root / "audiomnist_train_manifest.csv",
        require_complete=False,
    )[0]
    stored_t, stored_v, _, _ = load_cached_envelope_capture(run_dir, STORED_400_CACHE_ROOT, "train")
    with np.load(run_dir / "raw_capture.npz") as raw:
        volts = np.asarray(raw["volts_v"], dtype=np.float32)
        fs = float(raw["sample_rate_hz"])
    frame_length, hop_length = envelope_geometry(fs, 400)
    ours = np.column_stack([frame_ac_rms(volts[:, c], frame_length, hop_length) for c in range(5)])
    ours_t = (np.arange(len(ours)) * hop_length + frame_length / 2.0) / fs + RMS_WINDOW_MS / 2000.0
    rel = np.abs(ours - stored_v).max() / max(float(np.abs(stored_v).max()), 1e-12)
    print(
        json.dumps(
            {
                "run": run_dir.name,
                "rows_equal": len(ours) == len(stored_v),
                "max_time_error_s": float(np.abs(ours_t - stored_t).max()),
                "max_relative_value_error": float(rel),
                "geometry_by_rate": {
                    r: dict(zip(("window_samples", "hop_samples"), envelope_geometry(fs, r)))
                    for r in (100, 400)
                },
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    model_names = MODEL_NAMES if args.model_name == "all" else (args.model_name,)
    if args.stage == "verify":
        verify_400(args)
        return
    if args.stage == "export-pi":
        export_pi(args, model_names)
        return
    for rate_hz in args.rates:
        # Build/load train and validation once per rate; test is never touched here.
        training, training_audit, _ = build_split("train", rate_hz, args, build=True)
        validation, validation_audit, _ = build_split("validation", rate_hz, args, build=True)
        steps = training_audit["input_steps"]
        print(
            f"[{rate_hz} Hz] train={len(training)} validation={len(validation)} "
            f"steps/utterance median={int(steps.median())} max={int(steps.max())}",
            flush=True,
        )
        if args.stage == "build":
            continue
        audit = pd.concat([training_audit, validation_audit], ignore_index=True)
        for name in model_names:
            train_one(name, rate_hz, args, training, validation, audit)


if __name__ == "__main__":
    main()
