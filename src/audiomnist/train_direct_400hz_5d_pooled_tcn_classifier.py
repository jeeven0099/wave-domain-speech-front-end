#!/usr/bin/env python3
"""Train a causal pooled 10-class model on direct-400 Hz resonator data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_direct_400hz_5d_tcn_ctc import (
    DIRECT_INPUT_FEATURES,
    DIRECT_INPUT_HOP_MS,
    DIRECT_INPUT_RATE_HZ,
    FEATURE_NAMES,
    FRONTEND_KERNEL_SIZE,
    FRONTEND_STRIDE,
    HERE,
    MODEL_CHANNELS,
    MODEL_OUTPUT_RATE_HZ,
    STIMULI_ROOT,
    TCN_KERNEL_SIZE,
    DILATIONS,
    Direct400CausalConvTcnCtc,
    DirectDigitDataset,
    build_split,
    fit_normalizer,
)


CAPTURE_ROOT = HERE / "captures"
OUTPUT_ROOT = HERE / "models" / "direct_400hz_5d_pooled_tcn_classifier"
SHARED_CACHE_ROOT = (
    HERE / "models" / "direct_400hz_5d_tcn_ctc" / "direct_400hz_capture_cache"
)
FEATURE_CONTRACT = (
    "audiomnist_direct_magnitude_5d_400hz_no_filter_"
    "causal_tcn_stats_pool_softmax_v1"
)
CHECKPOINT_FILENAME = "best_direct_400hz_5d_pooled_tcn_classifier.pt"
ARCHITECTURE_ID = "direct_400hz_c32_fk32_long_tcn_stats_pool_softmax"
REPRESENTATION_DESCRIPTION = "direct-400 Hz"
REPRESENTATION_SUMMARY = {
    "source_capture_rate_hz": 48_000.0,
    "emulated_direct_sampling_rate_hz_per_sensor": DIRECT_INPUT_RATE_HZ,
    "physical_sensor_count": DIRECT_INPUT_FEATURES,
    "input_features_per_2p5ms_step": DIRECT_INPUT_FEATURES,
    "input_scalar_rate_per_second": int(
        DIRECT_INPUT_RATE_HZ * DIRECT_INPUT_FEATURES
    ),
    "feature_names": FEATURE_NAMES,
    "magnitude_operation": True,
    "fixed_rms_or_average_used": False,
    "anti_alias_filter_used": False,
    "analog_envelope_detector_used": False,
}
CLASSES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--stimuli-root", type=Path, default=STIMULI_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=SHARED_CACHE_ROOT)
    parser.add_argument("--context-ms", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
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
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def classifier_collate(batch):
    sequences, labels, indices = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = torch.zeros(
        len(sequences), int(lengths.max()), DIRECT_INPUT_FEATURES, dtype=torch.float32
    )
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return (
        padded,
        lengths,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
    )


class Direct400PooledTcnClassifier(Direct400CausalConvTcnCtc):
    """Causal frame encoder with running mean/max statistics at the endpoint."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(
            channels=MODEL_CHANNELS,
            frontend_kernel_size=FRONTEND_KERNEL_SIZE,
            frontend_stride=FRONTEND_STRIDE,
            tcn_kernel_size=TCN_KERNEL_SIZE,
            dilations=DILATIONS,
            dropout=dropout,
        )
        del self.classifier
        self.pool_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * MODEL_CHANNELS, CLASSES)

    def encode_frames(
        self, values: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = values.transpose(1, 2)
        output = F.pad(output, (self.frontend_kernel_size - 1, 0))
        output = self.frontend_pointwise(self.frontend_depthwise(output))
        output = self.frontend_normalization(output.transpose(1, 2)).transpose(1, 2)
        output = self.frontend_dropout(F.silu(output))
        for block in self.blocks:
            output = block(output)
        output_lengths = torch.div(
            lengths + self.frontend_stride - 1,
            self.frontend_stride,
            rounding_mode="floor",
        )
        return output, output_lengths

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frames, output_lengths = self.encode_frames(values, lengths)
        steps = torch.arange(frames.shape[-1], device=frames.device)
        valid = steps.unsqueeze(0) < output_lengths.to(frames.device).unsqueeze(1)
        valid_float = valid.unsqueeze(1).to(frames.dtype)
        mean = (frames * valid_float).sum(dim=-1) / output_lengths.to(
            frames.device, frames.dtype
        ).clamp_min(1).unsqueeze(1)
        maximum = frames.masked_fill(~valid.unsqueeze(1), -torch.inf).amax(dim=-1)
        pooled = torch.cat([mean, maximum], dim=1)
        return self.classifier(self.pool_dropout(pooled))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def learned_macs_per_one_second() -> int:
    encoder_per_output = (
        DIRECT_INPUT_FEATURES * FRONTEND_KERNEL_SIZE
        + DIRECT_INPUT_FEATURES * MODEL_CHANNELS
        + len(DILATIONS)
        * (MODEL_CHANNELS * TCN_KERNEL_SIZE + MODEL_CHANNELS * MODEL_CHANNELS)
    )
    endpoint_head = 2 * MODEL_CHANNELS * CLASSES
    return int(encoder_per_output * MODEL_OUTPUT_RATE_HZ + endpoint_head)


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
            logits.topk(3, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).sum().detach().cpu()
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


def validate_model_code() -> None:
    model = Direct400PooledTcnClassifier().eval()
    values = torch.randn(2, 401, DIRECT_INPUT_FEATURES)
    lengths = torch.tensor([401, 377], dtype=torch.long)
    with torch.inference_mode():
        logits = model(values, lengths)
        frames, output_lengths = model.encode_frames(values, lengths)
        changed = values.clone()
        changed[:, 200:] += 5.0 * torch.randn_like(changed[:, 200:])
        changed_frames, _ = model.encode_frames(changed, lengths)
    if logits.shape != (2, CLASSES):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if output_lengths.tolist() != [101, 95]:
        raise RuntimeError(f"Unexpected output lengths: {output_lengths.tolist()}")
    unaffected = math.ceil(200 / FRONTEND_STRIDE)
    causal_error = float(
        (frames[:, :, :unaffected] - changed_frames[:, :, :unaffected]).abs().max()
    )
    if causal_error > 1e-5:
        raise RuntimeError(f"Encoder failed causality audit: {causal_error}")
    print(
        json.dumps(
            {
                "status": "ok",
                "input_shape": list(values.shape),
                "frame_shape": list(frames.shape),
                "output_shape": list(logits.shape),
                "parameters": parameter_count(model),
                "learned_macs_per_audio_second": learned_macs_per_one_second(),
                "learned_flops_per_audio_second_2_per_mac": (
                    2 * learned_macs_per_one_second()
                ),
                "encoder_future_invariance_max_abs_error": causal_error,
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

    print("Loading AudioMNIST training and validation; test remains unopened.")
    training, training_audit, training_captures = build_split(
        "train",
        args.capture_root.resolve(),
        args.stimuli_root.resolve(),
        args.cache_root.resolve(),
        args.context_ms,
        require_complete=not args.allow_incomplete,
    )
    validation, validation_audit, validation_captures = build_split(
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

    model = Direct400PooledTcnClassifier(args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    checkpoint_path = output_root / CHECKPOINT_FILENAME
    parameters = parameter_count(model)
    model_macs = learned_macs_per_one_second()
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    started = time.time()
    print(
        f"Training {REPRESENTATION_DESCRIPTION} pooled TCN classifier on {device}; "
        f"parameters={parameters:,}; learned MAC/s={model_macs:,}",
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
                    "architecture": ARCHITECTURE_ID,
                    "model_config": {
                        "channels": MODEL_CHANNELS,
                        "frontend_kernel_size": FRONTEND_KERNEL_SIZE,
                        "frontend_stride": FRONTEND_STRIDE,
                        "tcn_kernel_size": TCN_KERNEL_SIZE,
                        "dilations": DILATIONS,
                        "dropout": args.dropout,
                        "classes": CLASSES,
                        "causal_encoder": True,
                        "pooling": "masked running mean plus running maximum",
                    },
                    "feature_contract": FEATURE_CONTRACT,
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
            f"epoch {epoch:03d}: train={training_loss:.4f} "
            f"top1={100*training_accuracy:.2f}% top3={100*training_top3:.2f}% | "
            f"valid={validation_loss:.4f} top1={100*validation_accuracy:.2f}% "
            f"top3={100*validation_top3:.2f}%",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            print(f"Early stopping; selected epoch {best_epoch}", flush=True)
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
        "analysis_role": "training and validation only; held-out test not opened",
        "feature_contract": FEATURE_CONTRACT,
        **REPRESENTATION_SUMMARY,
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
        "model": (
            "causal stride-4 depthwise Conv1D plus seven causal depthwise TCN "
            "blocks plus masked running mean/max pooling plus 10-class linear head"
        ),
        "pooling_streamability": (
            "mean and maximum can be updated causally; decision emitted at endpoint"
        ),
        "model_channels": MODEL_CHANNELS,
        "frontend_kernel_size_input_steps": FRONTEND_KERNEL_SIZE,
        "frontend_causal_history_ms": (
            (FRONTEND_KERNEL_SIZE - 1) * DIRECT_INPUT_HOP_MS
        ),
        "frontend_stride": FRONTEND_STRIDE,
        "encoder_output_rate_hz": MODEL_OUTPUT_RATE_HZ,
        "temporal_receptive_field_input_steps": model.receptive_field_direct_samples,
        "temporal_receptive_field_ms": (
            model.receptive_field_direct_samples * DIRECT_INPUT_HOP_MS
        ),
        "future_lookahead_ms": 0.0,
        "parameters": parameters,
        "learned_macs_per_one_second_utterance": model_macs,
        "learned_flops_per_one_second_utterance_2_per_mac": 2 * model_macs,
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
    print("\nFrozen training/validation result:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
