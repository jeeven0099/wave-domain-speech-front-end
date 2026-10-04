#!/usr/bin/env python3
"""Train each of the three basic/minimal CNN baselines (cnn_one_fstride4,
tiny_conv, cnn_trad_fpool3 -- see cnn_baseline_models.py) on the same
five-channel 400 Hz AC-RMS envelope data used for the six already-frozen
compact architectures in train_fabricated_400hz_5d_baseline_classifiers.py.

This is a near-literal copy of that trainer with only the model source
swapped (cnn_baseline_models instead of fabricated_400hz_5d_baseline_models)
and a separate output root, so training/validation/checkpointing/early
stopping behave identically and the results stay directly comparable.
Held-out test is never opened here; only evaluate_cnn_baselines.py does.

One architecture is trained per invocation, selected with --model-name from
{cnn_one_fstride4, tiny_conv, cnn_trad_fpool3}, or --model-name all.
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

from cnn_baseline_models import ARCHITECTURE_IDS, FABRICATED_CNN_FACTORIES
from fabricated_400hz_5d_baseline_models import INPUT_CHANNELS
from train_direct_400hz_5d_pooled_tcn_classifier import classifier_collate
from train_direct_400hz_5d_tcn_ctc import (
    DIRECT_INPUT_FEATURES,
    HERE,
    STIMULI_ROOT,
    DirectDigitDataset,
    fit_normalizer,
    set_reproducible,
)
from train_envelope_400hz_5d_pooled_tcn_classifier import (
    CACHE_ROOT as ENVELOPE_CACHE_ROOT,
    ENVELOPE_HOP_MS,
    ENVELOPE_RATE_HZ,
    FEATURE_NAMES,
    RMS_WINDOW_MS,
    build_envelope_split,
)

assert DIRECT_INPUT_FEATURES == INPUT_CHANNELS

CAPTURE_ROOT = HERE / "captures"
OUTPUT_ROOT = HERE / "models" / "fabricated_cnn_baselines"
FEATURE_CONTRACT_TEMPLATE = (
    "audiomnist_ac_rms_envelope_5d_400hz_10ms_window_2p5ms_hop_"
    "causal_basic_cnn_{model_name}_masked_mean_softmax_v1"
)
CLASSES = 10
MODEL_NAMES = tuple(FABRICATED_CNN_FACTORIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name", choices=(*MODEL_NAMES, "all"), required=True
    )
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--stimuli-root", type=Path, default=STIMULI_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ENVELOPE_CACHE_ROOT)
    parser.add_argument("--context-ms", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--validate-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.minimum_epochs > args.epochs:
        raise ValueError("--minimum-epochs cannot exceed --epochs")
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Batch size, epochs, and patience must be positive")
    if args.context_ms < 0:
        raise ValueError("--context-ms cannot be negative")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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
        values = values.to(device)
        labels = labels.to(device)
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
        total_correct += int(logits.argmax(dim=1).eq(labels).sum().detach().cpu())
        total_top3 += int(
            logits.topk(3, dim=1)
            .indices.eq(labels.unsqueeze(1))
            .any(dim=1)
            .sum()
            .detach()
            .cpu()
        )
        total_examples += count
    denominator = max(total_examples, 1)
    return (
        total_loss / denominator,
        total_correct / denominator,
        total_top3 / denominator,
    )


@torch.inference_mode()
def prediction_rows(
    model: nn.Module,
    dataset: DirectDigitDataset,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for values, lengths, labels, indices in loader:
        logits = model(values.to(device), lengths)
        probabilities = torch.softmax(logits, dim=1).cpu()
        top3 = probabilities.topk(3, dim=1).indices
        predictions = probabilities.argmax(dim=1)
        for label, prediction, choices, probability, example_index in zip(
            labels.tolist(),
            predictions.tolist(),
            top3.tolist(),
            probabilities.tolist(),
            indices.tolist(),
        ):
            example = dataset.examples[example_index]
            rows.append(
                {
                    "run": example.run,
                    "recording_id": example.recording_id,
                    "speaker_id": example.speaker_id,
                    "repetition": example.repetition,
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
    """Structural checks mirroring
    train_fabricated_400hz_5d_baseline_classifiers.validate_model_code:
    padding invariance and no leakage past `lengths` for the masked mean
    pool. The causal building blocks themselves (CausalConv1d, ConvBNAct1d)
    are imported unchanged from fabricated_400hz_5d_baseline_models.py, so
    they are not re-audited here.
    """
    model = FABRICATED_CNN_FACTORIES[model_name]().eval()
    total_stride = model.total_stride
    batch, long_len, short_len = 2, 120, 77
    values = torch.randn(batch, long_len, INPUT_CHANNELS)
    lengths = torch.tensor([long_len, short_len], dtype=torch.long)
    with torch.inference_mode():
        logits = model(values, lengths)
        padded_values = torch.cat(
            [values, torch.randn(batch, 40, INPUT_CHANNELS)], dim=1
        )
        padded_logits = model(padded_values, lengths)
        padding_error = float((logits - padded_logits).abs().max())
        garbled_values = values.clone()
        garbled_values[1, short_len:] = 1000.0 * torch.randn_like(
            garbled_values[1, short_len:]
        )
        garbled_logits = model(garbled_values, lengths)
        garbling_error = float((logits[1] - garbled_logits[1]).abs().max())
    if logits.shape != (batch, CLASSES):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if padding_error > 1e-4:
        raise RuntimeError(
            f"{model_name}: masked pooling is not padding-invariant "
            f"(max abs diff {padding_error})"
        )
    if garbling_error > 1e-4:
        raise RuntimeError(
            f"{model_name}: masked pooling leaked content past `lengths` "
            f"(max abs diff {garbling_error})"
        )
    expected_lengths = [
        math.ceil(long_len / total_stride),
        math.ceil(short_len / total_stride),
    ]
    frame_logits = model.encode_frames(values)
    output_steps = frame_logits.shape[1]
    if output_steps < max(expected_lengths):
        raise RuntimeError(
            f"{model_name}: encoder produced {output_steps} frames, fewer than "
            f"the {max(expected_lengths)} implied by total_stride={total_stride}"
        )
    print(
        json.dumps(
            {
                "model_name": model_name,
                "status": "ok",
                "total_stride": total_stride,
                "input_shape": list(values.shape),
                "frame_shape": list(frame_logits.shape),
                "output_shape": list(logits.shape),
                "parameters": parameter_count(model),
                "padding_invariance_max_abs_error": padding_error,
                "padding_content_leak_max_abs_error": garbling_error,
            },
            indent=2,
        )
    )


def train_one(model_name: str, args: argparse.Namespace) -> dict[str, object]:
    set_reproducible(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = (args.output_root / model_name).resolve()
    checkpoint_filename = f"best_fabricated_basic_cnn_{model_name}_classifier.pt"
    architecture_id = ARCHITECTURE_IDS[model_name]
    feature_contract = FEATURE_CONTRACT_TEMPLATE.format(model_name=model_name)
    summary_path = output_root / "frozen_training_validation_summary.json"
    if summary_path.is_file() and not args.force:
        print(
            f"[{model_name}] A frozen result already exists. "
            "Use --force only for a deliberate retrain."
        )
        print(summary_path.read_text(encoding="ascii"))
        return json.loads(summary_path.read_text(encoding="ascii"))
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[{model_name}] Loading AudioMNIST training and validation; "
        "test remains unopened.",
        flush=True,
    )
    training, training_audit, training_captures = build_envelope_split(
        "train",
        args.capture_root.resolve(),
        args.stimuli_root.resolve(),
        args.cache_root.resolve(),
        args.context_ms,
        require_complete=not args.allow_incomplete,
    )
    validation, validation_audit, validation_captures = build_envelope_split(
        "validation",
        args.capture_root.resolve(),
        args.stimuli_root.resolve(),
        args.cache_root.resolve(),
        args.context_ms,
        require_complete=not args.allow_incomplete,
    )
    mean, std = fit_normalizer(training)
    np.savez(output_root / "train_only_normalizer.npz", mean=mean, std=std)
    pd.concat([training_audit, validation_audit], ignore_index=True).to_csv(
        output_root / "training_validation_example_audit.csv", index=False
    )

    training_dataset = DirectDigitDataset(training, mean, std)
    validation_dataset = DirectDigitDataset(validation, mean, std)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=classifier_collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=classifier_collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = FABRICATED_CNN_FACTORIES[model_name]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    checkpoint_path = output_root / checkpoint_filename
    parameters = parameter_count(model)
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    started = time.time()
    print(
        f"[{model_name}] Training on {device}; parameters={parameters:,}; "
        f"total_stride={model.total_stride}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        training_loss, training_accuracy, training_top3 = run_epoch(
            model, training_loader, device, optimizer
        )
        validation_loss, validation_accuracy, validation_top3 = run_epoch(
            model, validation_loader, device
        )
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
                    "architecture": architecture_id,
                    "model_config": {
                        "model_name": model_name,
                        "classes": CLASSES,
                        "total_stride": model.total_stride,
                        "causal_encoder": True,
                        "pooling": "masked mean over per-frame class logits",
                    },
                    "feature_contract": feature_contract,
                    "feature_names": FEATURE_NAMES,
                    "input_mean": mean,
                    "input_std": std,
                    "context_ms": args.context_ms,
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
            f"valid={validation_loss:.4f} top1={100*validation_accuracy:.2f}% "
            f"top3={100*validation_top3:.2f}%",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            print(f"[{model_name}] Early stopping; selected epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    predictions = prediction_rows(model, validation_dataset, validation_loader, device)
    predictions.to_csv(output_root / "validation_predictions.csv", index=False)
    per_digit = (
        predictions.groupby("reference", as_index=False)
        .agg(
            examples=("top1_correct", "size"),
            top1_accuracy=("top1_correct", "mean"),
            top3_accuracy=("top3_correct", "mean"),
        )
        .sort_values("reference")
    )
    per_digit.to_csv(output_root / "validation_per_digit.csv", index=False)

    summary = {
        "model_name": model_name,
        "architecture": architecture_id,
        "analysis_role": "training and validation only; held-out test not opened",
        "feature_contract": feature_contract,
        "feature_names": FEATURE_NAMES,
        "source_capture_rate_hz_nominal": 48_000.0,
        "envelope_rate_hz_per_sensor": ENVELOPE_RATE_HZ,
        "physical_sensor_count": DIRECT_INPUT_FEATURES,
        "rms_window_ms": RMS_WINDOW_MS,
        "envelope_hop_ms": ENVELOPE_HOP_MS,
        "feature_availability_latency_ms": RMS_WINDOW_MS,
        "training_captures": training_captures,
        "validation_captures": validation_captures,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "training_speakers": len(set(training_audit["speaker_id"].astype(str))),
        "validation_speakers": len(set(validation_audit["speaker_id"].astype(str))),
        "normalizer_source": "training frames only",
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "known_digit_boundaries_used_to_form_isolated_examples": True,
        "heldout_test_accessed": False,
        "total_stride": model.total_stride,
        "future_lookahead_ms": 0.0,
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
