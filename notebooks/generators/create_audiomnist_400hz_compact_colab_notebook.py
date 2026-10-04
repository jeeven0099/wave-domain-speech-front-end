from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audiomnist_400hz_digital_compact_models_colab.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
        # AudioMNIST digital compact models at 400 Hz

        This notebook trains, validates, freezes, and then tests six compact
        causal model families on a **single-channel 400 Hz RMS-envelope stream**:

        1. MatchboxNet 3x1x64, causal 1D adaptation
        2. TC-ResNet8, causal 1D adaptation
        3. MLPerf Tiny DS-CNN, causal 1D adaptation
        4. BC-ResNet-1, causal 1D adaptation
        5. BC-ResNet-2, causal 1D adaptation
        6. EdgeSpeechNet-A, causal 1D adaptation

        The fixed AudioMNIST protocol is speaker-disjoint: 36 training speakers
        (18,000 utterances), 12 validation speakers (6,000), and 12 test speakers
        (6,000). All models receive the same 400 envelope values per one-second
        example.

        Each value is a causal RMS amplitude computed from the full-rate waveform
        using a 10 ms trailing window and a 2.5 ms hop. There is no FFT or Mel
        transform. This preserves broadband amplitude modulation before reducing
        the representation to 400 values/s; it is **not** direct 400 Hz waveform
        sampling and it is **not** a native 400 Hz ADC measurement.

        These are architecture-family adaptations for a 1D waveform. They are not
        official reproduced scores from the original model publications. The
        notebook reports top-1/top-3 accuracy, parameters, FP32 state size, learned
        MACs, compute cost in FLOPs, total estimated FLOPs including input
        normalization, latency, causality, and confusion matrices.
        """
    ),
    markdown(
        """
        ## 1. Colab setup

        Select **Runtime > Change runtime type > T4 GPU**, then run all cells.
        Checkpoints are saved to Google Drive and reused after interruptions.
        """
    ),
    code(
        r"""
        %pip install -q soundfile
        """
    ),
    code(
        r"""
        import copy
        import io
        import json
        import math
        import os
        import random
        import subprocess
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import soundfile as sf
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from IPython.display import display
        from sklearn.metrics import confusion_matrix
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        SEED = 310
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    markdown("## 2. Configuration and frozen protocol"),
    code(
        r"""
        IN_COLAB = "google.colab" in str(get_ipython())
        USE_GOOGLE_DRIVE = True
        REUSE_CHECKPOINTS = True
        QUICK_SMOKE_TEST = False

        if IN_COLAB and USE_GOOGLE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")

        DATASET_ROOT = Path("/content/AudioMNIST") if IN_COLAB else Path("datasets/AudioMNIST")
        CACHE_ROOT = Path("/content/audiomnist_rms_envelope_400hz") if IN_COLAB else Path(".cache/audiomnist_rms_envelope_400hz")
        if IN_COLAB and USE_GOOGLE_DRIVE:
            OUTPUT_ROOT = Path(
                "/content/drive/MyDrive/audiomnist_digital_compact_400hz_rms_envelope"
            )
        else:
            OUTPUT_ROOT = Path("audiomnist_digital_compact_400hz_rms_envelope_outputs")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        INPUT_RATE_HZ = 400
        FIXED_SECONDS = 1.0
        INPUT_SAMPLES = int(INPUT_RATE_HZ * FIXED_SECONDS)
        INPUT_FEATURES = 1
        INPUT_SCALARS_PER_SECOND = INPUT_RATE_HZ * INPUT_FEATURES
        ENVELOPE_WINDOW_MS = 10.0
        ENVELOPE_HOP_MS = 1000.0 / INPUT_RATE_HZ
        EXPECTED_SOURCE_RATE_HZ = 48_000
        # Streaming running RMS: square, add, and remove one source sample.
        ENVELOPE_FRONT_END_ARITHMETIC_OPS_PER_SECOND = 3 * EXPECTED_SOURCE_RATE_HZ
        ENVELOPE_SQRTS_PER_SECOND = INPUT_RATE_HZ
        TARGET_PEAK_DBFS = -9.0
        BATCH_SIZE = 256
        EPOCHS = 50
        MINIMUM_EPOCHS = 20
        PATIENCE = 8
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 1e-4
        NUM_WORKERS = 2 if IN_COLAB else 0
        LATENCY_WARMUP = 20
        LATENCY_REPEATS = 100

        if QUICK_SMOKE_TEST:
            EPOCHS = 2
            MINIMUM_EPOCHS = 1
            PATIENCE = 2

        SPLIT_SPEAKERS = {
            "train": {28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2,
                      8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55},
            "validation": {12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50},
            "test": {26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51},
        }
        assert not (SPLIT_SPEAKERS["train"] & SPLIT_SPEAKERS["validation"])
        assert not (SPLIT_SPEAKERS["train"] & SPLIT_SPEAKERS["test"])
        assert not (SPLIT_SPEAKERS["validation"] & SPLIT_SPEAKERS["test"])

        EXPERIMENT_CONTRACT = {
            "task": "AudioMNIST isolated spoken-digit classification",
            "input": "one causal broadband RMS-envelope value per 2.5 ms",
            "input_rate_hz": INPUT_RATE_HZ,
            "input_features_per_step": INPUT_FEATURES,
            "input_scalar_rate_per_second": INPUT_SCALARS_PER_SECOND,
            "fixed_duration_s": FIXED_SECONDS,
            "source_audio_rate_hz": EXPECTED_SOURCE_RATE_HZ,
            "envelope_window_ms": ENVELOPE_WINDOW_MS,
            "envelope_hop_ms": ENVELOPE_HOP_MS,
            "rms_envelope_used": True,
            "native_400hz_adc_claim": False,
            "fft_or_mel_used": False,
            "decoder": "10-class endpoint softmax; top-1 and top-3",
            "model_causality": "past-only temporal convolution; no future padding",
            "split": "fixed 36/12/12 speaker-disjoint AudioMNIST protocol",
            "selection": "validation accuracy only; test evaluated after all checkpoints freeze",
            "flop_convention": "one multiply-accumulate equals two FLOPs",
        }
        (OUTPUT_ROOT / "experiment_contract.json").write_text(
            json.dumps(EXPERIMENT_CONTRACT, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(EXPERIMENT_CONTRACT, indent=2))
        """
    ),
    markdown("## 3. Download and inventory AudioMNIST"),
    code(
        r"""
        if not (DATASET_ROOT / "data").is_dir():
            if DATASET_ROOT.exists():
                raise RuntimeError(f"{DATASET_ROOT} exists but does not contain data/")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/soerenab/AudioMNIST.git", str(DATASET_ROOT)],
                check=True,
            )

        inventory_rows = []
        for wav_path in sorted((DATASET_ROOT / "data").glob("*/*.wav")):
            digit_text, speaker_text, repetition_text = wav_path.stem.split("_")
            speaker = int(speaker_text)
            split_name = next(
                name for name, members in SPLIT_SPEAKERS.items() if speaker in members
            )
            inventory_rows.append({
                "path": str(wav_path),
                "digit": int(digit_text),
                "speaker": speaker,
                "repetition": int(repetition_text),
                "split": split_name,
            })
        inventory = pd.DataFrame(inventory_rows)
        split_summary = inventory.groupby("split").agg(
            speakers=("speaker", "nunique"), utterances=("path", "size")
        ).reindex(["train", "validation", "test"])
        display(split_summary)
        assert split_summary.loc["train", "utterances"] == 18_000
        assert split_summary.loc["validation", "utterances"] == 6_000
        assert split_summary.loc["test", "utterances"] == 6_000
        inventory.to_csv(OUTPUT_ROOT / "dataset_inventory.csv", index=False)
        """
    ),
    markdown(
        """
        ## 4. Build the 400 Hz RMS-envelope cache

        Peak normalization is performed first. Every output value is the RMS of a
        trailing 10 ms source-waveform window, emitted every 2.5 ms. The window is
        causal and zero-padded before the start of the one-second example. Every
        output contains exactly 400 nonnegative envelope values. Test inputs are
        cached here, but test labels are not used for training or model selection.
        """
    ),
    code(
        r"""
        def read_mono(path):
            values, sample_rate = sf.read(path, always_2d=False)
            values = np.asarray(values, dtype=np.float32)
            if values.ndim == 2:
                values = values.mean(axis=1)
            return values, int(sample_rate)


        def peak_normalize(values, peak_dbfs=TARGET_PEAK_DBFS):
            peak = float(np.max(np.abs(values)))
            if peak <= 1e-12:
                return values.astype(np.float32)
            target = 10.0 ** (peak_dbfs / 20.0)
            return (values * (target / peak)).astype(np.float32)


        def crop_or_pad(values, target_length):
            if len(values) > target_length:
                start = (len(values) - target_length) // 2
                return values[start:start + target_length]
            left = (target_length - len(values)) // 2
            right = target_length - len(values) - left
            return np.pad(values, (left, right))


        def rms_envelope_400hz(path):
            values, source_rate = read_mono(path)
            if source_rate != EXPECTED_SOURCE_RATE_HZ:
                raise ValueError(
                    f"Expected {EXPECTED_SOURCE_RATE_HZ} Hz AudioMNIST WAV, "
                    f"got {source_rate} Hz: {path}"
                )
            values = peak_normalize(values)
            fixed = crop_or_pad(values, int(round(source_rate * FIXED_SECONDS)))
            window_samples = int(round(source_rate * ENVELOPE_WINDOW_MS / 1000.0))
            end_indices = np.rint(
                (np.arange(INPUT_SAMPLES, dtype=np.float64) + 1.0)
                * source_rate / INPUT_RATE_HZ
            ).astype(np.int64)
            end_indices = np.clip(end_indices, 1, len(fixed))
            start_indices = np.maximum(0, end_indices - window_samples)
            squared = np.square(fixed, dtype=np.float64)
            cumulative_energy = np.concatenate(([0.0], np.cumsum(squared)))
            energy = cumulative_energy[end_indices] - cumulative_energy[start_indices]
            # Divide by the full window length: unavailable pre-roll is zero padding.
            result = np.sqrt(np.maximum(energy, 0.0) / window_samples)
            return np.asarray(result, dtype=np.float32)


        CACHE_PATH = CACHE_ROOT / "audiomnist_400hz_rms10ms_hop2p5ms_v1.npz"
        if CACHE_PATH.is_file():
            with np.load(CACHE_PATH) as cached:
                ENVELOPES = np.asarray(cached["envelopes"], dtype=np.float32)
                LABELS = np.asarray(cached["labels"], dtype=np.int64)
                SPEAKERS = np.asarray(cached["speakers"], dtype=np.int64)
                REPETITIONS = np.asarray(cached["repetitions"], dtype=np.int64)
            print("Loaded cache:", CACHE_PATH)
        else:
            ENVELOPES = np.empty((len(inventory), INPUT_SAMPLES), dtype=np.float32)
            for index, path in enumerate(tqdm(inventory["path"], desc="Computing RMS envelopes")):
                ENVELOPES[index] = rms_envelope_400hz(path)
            LABELS = inventory["digit"].to_numpy(np.int64)
            SPEAKERS = inventory["speaker"].to_numpy(np.int64)
            REPETITIONS = inventory["repetition"].to_numpy(np.int64)
            np.savez_compressed(
                CACHE_PATH,
                envelopes=ENVELOPES,
                labels=LABELS,
                speakers=SPEAKERS,
                repetitions=REPETITIONS,
            )
            print("Saved cache:", CACHE_PATH)

        assert ENVELOPES.shape == (30_000, INPUT_SAMPLES)
        assert np.isfinite(ENVELOPES).all()
        assert np.min(ENVELOPES) >= 0.0
        INDICES = {
            split: inventory.index[inventory["split"].eq(split)].to_numpy(np.int64)
            for split in ("train", "validation", "test")
        }
        print("Envelope tensor:", ENVELOPES.shape)
        """
    ),
    markdown("## 5. Train-only normalization and data loaders"),
    code(
        r"""
        TRAIN_MEAN = np.float32(ENVELOPES[INDICES["train"]].mean())
        TRAIN_STD = np.float32(max(ENVELOPES[INDICES["train"]].std(), 1e-6))
        print("Training-only mean/std:", float(TRAIN_MEAN), float(TRAIN_STD))


        class EnvelopeDataset(Dataset):
            def __init__(self, indices):
                self.indices = np.asarray(indices, dtype=np.int64)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                index = int(self.indices[item])
                values = (ENVELOPES[index] - TRAIN_MEAN) / TRAIN_STD
                return (
                    torch.from_numpy(values[:, None].astype(np.float32)),
                    int(LABELS[index]),
                    index,
                )


        def make_loader(split, shuffle=False):
            return DataLoader(
                EnvelopeDataset(INDICES[split]),
                batch_size=BATCH_SIZE,
                shuffle=shuffle,
                num_workers=NUM_WORKERS,
                pin_memory=DEVICE.type == "cuda",
                persistent_workers=NUM_WORKERS > 0,
            )


        TRAIN_LOADER = make_loader("train", shuffle=True)
        VALIDATION_LOADER = make_loader("validation")
        # The test loader is intentionally created only in the held-out test cell.
        """
    ),
    markdown(
        """
        ## 6. Causal 1D compact architectures

        All temporal convolutions pad only on the left. Each model emits frame
        logits and averages the available past frame logits for the endpoint
        decision. Because the input is a 1D waveform rather than a time-frequency
        image, DS-CNN, BC-ResNet, and EdgeSpeechNet are explicitly 1D adaptations.
        """
    ),
    code(
        r"""
        class CausalConv1d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         dilation=1, groups=1, bias=False):
                super().__init__()
                self.left_context = dilation * (kernel_size - 1)
                self.conv = nn.Conv1d(
                    in_channels, out_channels, kernel_size, stride=stride,
                    dilation=dilation, groups=groups, bias=bias,
                )

            def forward(self, values):
                return self.conv(F.pad(values, (self.left_context, 0)))


        class ConvBNAct(nn.Sequential):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         dilation=1, groups=1):
                super().__init__(
                    CausalConv1d(
                        in_channels, out_channels, kernel_size, stride=stride,
                        dilation=dilation, groups=groups, bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )


        class TCSConv(nn.Sequential):
            def __init__(self, in_channels, out_channels, kernel_size,
                         dilation=1, activation=True):
                layers = [
                    CausalConv1d(
                        in_channels, in_channels, kernel_size, dilation=dilation,
                        groups=in_channels, bias=False,
                    ),
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                ]
                if activation:
                    layers.append(nn.ReLU(inplace=True))
                super().__init__(*layers)


        class EndpointClassifier(nn.Module):
            temporal_stride = 1
            temporal_receptive_field_samples = 1

            def forward(self, values):
                return self.encode_frames(values).mean(dim=1)


        class MatchboxResidual(nn.Module):
            def __init__(self, in_channels, channels, kernel_size):
                super().__init__()
                self.main = TCSConv(in_channels, channels, kernel_size, activation=False)
                self.skip = (
                    nn.Sequential(
                        nn.Conv1d(in_channels, channels, 1, bias=False),
                        nn.BatchNorm1d(channels),
                    )
                    if in_channels != channels else nn.Identity()
                )

            def forward(self, values):
                return F.relu(self.main(values) + self.skip(values), inplace=True)


        class MatchboxNet3x1x64(EndpointClassifier):
            temporal_stride = 1
            temporal_receptive_field_samples = 109

            def __init__(self):
                super().__init__()
                self.prologue = TCSConv(1, 128, 11)
                self.blocks = nn.Sequential(
                    MatchboxResidual(128, 64, 13),
                    MatchboxResidual(64, 64, 15),
                    MatchboxResidual(64, 64, 17),
                )
                self.epilogue = nn.Sequential(
                    TCSConv(64, 128, 29, dilation=2),
                    ConvBNAct(128, 128, 1),
                )
                self.classifier = nn.Linear(128, 10)

            def encode_frames(self, values):
                values = self.epilogue(
                    self.blocks(self.prologue(values.transpose(1, 2)))
                )
                return self.classifier(values.transpose(1, 2))


        class TCResidual(nn.Module):
            def __init__(self, in_channels, out_channels, stride):
                super().__init__()
                self.main = nn.Sequential(
                    ConvBNAct(in_channels, out_channels, 3, stride=stride),
                    CausalConv1d(out_channels, out_channels, 3, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                self.skip = (
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                        nn.BatchNorm1d(out_channels),
                    )
                    if in_channels != out_channels or stride != 1 else nn.Identity()
                )

            def forward(self, values):
                return F.relu(self.main(values) + self.skip(values), inplace=True)


        class TCResNet8(EndpointClassifier):
            temporal_stride = 8
            temporal_receptive_field_samples = 45

            def __init__(self):
                super().__init__()
                self.stem = ConvBNAct(1, 16, 3)
                self.blocks = nn.Sequential(
                    TCResidual(16, 24, 2),
                    TCResidual(24, 32, 2),
                    TCResidual(32, 48, 2),
                )
                self.classifier = nn.Linear(48, 10)

            def encode_frames(self, values):
                values = self.blocks(self.stem(values.transpose(1, 2)))
                return self.classifier(values.transpose(1, 2))


        class DSConvBlock(nn.Sequential):
            def __init__(self, channels):
                super().__init__(
                    CausalConv1d(channels, channels, 3, groups=channels, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels, 1, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )


        class MLPerfTinyDSCNN(EndpointClassifier):
            temporal_stride = 2
            temporal_receptive_field_samples = 26

            def __init__(self):
                super().__init__()
                self.stem = ConvBNAct(1, 64, 10, stride=2)
                self.blocks = nn.Sequential(*[DSConvBlock(64) for _ in range(4)])
                self.classifier = nn.Linear(64, 10)

            def encode_frames(self, values):
                values = self.blocks(self.stem(values.transpose(1, 2)))
                return self.classifier(values.transpose(1, 2))


        class BCResidual1d(nn.Module):
            def __init__(self, in_channels, out_channels, dilation, stride=1):
                super().__init__()
                self.main = nn.Sequential(
                    CausalConv1d(
                        in_channels, in_channels, 3, stride=stride,
                        dilation=dilation, groups=in_channels, bias=False,
                    ),
                    nn.BatchNorm1d(in_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout(0.1),
                )
                self.skip = (
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                        nn.BatchNorm1d(out_channels),
                    )
                    if in_channels != out_channels or stride != 1 else nn.Identity()
                )

            def forward(self, values):
                return F.relu(self.main(values) + self.skip(values), inplace=True)


        class BCResNet1d(EndpointClassifier):
            temporal_stride = 8
            temporal_receptive_field_samples = 117

            def __init__(self, scale=1):
                super().__init__()
                channels = [16 * scale, 8 * scale, 12 * scale, 16 * scale, 20 * scale]
                self.stem = ConvBNAct(1, channels[0], 5, stride=2)
                stages = []
                in_channels = channels[0]
                for stage_index, (out_channels, repeats) in enumerate(
                    zip(channels[1:], (2, 2, 4, 4))
                ):
                    blocks = []
                    for block_index in range(repeats):
                        stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                        blocks.append(
                            BCResidual1d(
                                in_channels, out_channels,
                                dilation=2 ** stage_index, stride=stride,
                            )
                        )
                        in_channels = out_channels
                    stages.append(nn.Sequential(*blocks))
                self.stages = nn.Sequential(*stages)
                self.final = ConvBNAct(channels[-1], 32 * scale, 5)
                self.classifier = nn.Linear(32 * scale, 10)

            def encode_frames(self, values):
                values = self.final(self.stages(self.stem(values.transpose(1, 2))))
                return self.classifier(values.transpose(1, 2))


        class EdgeResidual1d(nn.Module):
            def __init__(self, middle_channels):
                super().__init__()
                self.layers = nn.Sequential(
                    ConvBNAct(39, middle_channels, 3),
                    CausalConv1d(middle_channels, 39, 3, bias=False),
                    nn.BatchNorm1d(39),
                )

            def forward(self, values):
                return F.relu(values + self.layers(values), inplace=True)


        class EdgeSpeechNetA1d(EndpointClassifier):
            temporal_stride = 1
            temporal_receptive_field_samples = 29

            def __init__(self):
                super().__init__()
                self.stem = ConvBNAct(1, 39, 3)
                self.blocks = nn.Sequential(*[
                    EdgeResidual1d(width) for width in (20, 15, 25, 22, 22, 25)
                ])
                self.final = ConvBNAct(39, 45, 3)
                self.classifier = nn.Linear(45, 10)

            def encode_frames(self, values):
                values = self.final(self.blocks(self.stem(values.transpose(1, 2))))
                return self.classifier(values.transpose(1, 2))


        MODEL_FACTORIES = {
            "matchboxnet_3x1x64_1d": MatchboxNet3x1x64,
            "tc_resnet8_1d": TCResNet8,
            "mlperf_tiny_dscnn_1d": MLPerfTinyDSCNN,
            "bc_resnet_1_1d": lambda: BCResNet1d(scale=1),
            "bc_resnet_2_1d": lambda: BCResNet1d(scale=2),
            "edgespeechnet_a_1d": EdgeSpeechNetA1d,
        }

        # Shape and strict past-only audit.
        dummy = torch.randn(2, INPUT_SAMPLES, 1)
        changed = dummy.clone()
        change_start = INPUT_SAMPLES // 2
        changed[:, change_start:] += torch.randn_like(changed[:, change_start:]) * 5.0
        audit_rows = []
        for name, factory in MODEL_FACTORIES.items():
            model = factory().eval()
            with torch.inference_mode():
                logits = model(dummy)
                original_frames = model.encode_frames(dummy)
                changed_frames = model.encode_frames(changed)
            assert logits.shape == (2, 10), (name, logits.shape)
            unaffected = math.ceil(change_start / model.temporal_stride)
            future_error = float(
                (original_frames[:, :unaffected] - changed_frames[:, :unaffected]).abs().max()
            )
            if future_error > 1e-5:
                raise RuntimeError(f"{name} failed causality audit: {future_error}")
            audit_rows.append({
                "model": name,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "frame_logits_shape": str(tuple(original_frames.shape)),
                "temporal_stride": model.temporal_stride,
                "output_update_interval_ms": 1000.0 * model.temporal_stride / INPUT_RATE_HZ,
                "nominal_receptive_field_ms": (
                    1000.0 * model.temporal_receptive_field_samples / INPUT_RATE_HZ
                ),
                "future_lookahead_ms": 0.0,
                "future_invariance_max_abs_error": future_error,
            })
        architecture_audit = pd.DataFrame(audit_rows)
        architecture_audit.to_csv(OUTPUT_ROOT / "causal_architecture_audit.csv", index=False)
        display(architecture_audit)
        """
    ),
    markdown("## 7. Train and select checkpoints using validation only"),
    code(
        r"""
        @torch.inference_mode()
        def evaluate(model, loader):
            model.eval()
            criterion = nn.CrossEntropyLoss(reduction="sum")
            total_loss = 0.0
            labels_all, predictions_all, indices_all = [], [], []
            top3_correct = 0
            for values, labels, indices in loader:
                values = values.to(DEVICE, non_blocking=True)
                labels_device = labels.to(DEVICE, non_blocking=True)
                logits = model(values)
                total_loss += float(criterion(logits, labels_device).cpu())
                predictions = logits.argmax(dim=1)
                top3 = logits.topk(3, dim=1).indices
                top3_correct += int((top3 == labels_device[:, None]).any(dim=1).sum())
                labels_all.extend(labels.tolist())
                predictions_all.extend(predictions.cpu().tolist())
                indices_all.extend(indices.tolist())
            labels_array = np.asarray(labels_all)
            predictions_array = np.asarray(predictions_all)
            return {
                "loss": total_loss / len(labels_array),
                "accuracy": float((labels_array == predictions_array).mean()),
                "top3_accuracy": top3_correct / len(labels_array),
                "labels": labels_array,
                "predictions": predictions_array,
                "indices": np.asarray(indices_all),
            }


        def train_one(name, factory):
            model_dir = OUTPUT_ROOT / name
            model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = model_dir / "best_validation_checkpoint.pt"
            summary_path = model_dir / "training_summary.json"
            if REUSE_CHECKPOINTS and checkpoint_path.is_file() and summary_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model = factory()
                model.load_state_dict(checkpoint["model_state"])
                print("Reused:", name)
                return model, json.loads(summary_path.read_text())

            model = factory().to(DEVICE)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
            )
            criterion = nn.CrossEntropyLoss()
            history = []
            best_accuracy = -1.0
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            for epoch in range(1, EPOCHS + 1):
                model.train()
                train_loss = train_correct = train_count = 0
                for values, labels, _ in TRAIN_LOADER:
                    values = values.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(values)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    train_loss += float(loss.detach()) * len(labels)
                    train_correct += int(logits.argmax(dim=1).eq(labels).sum())
                    train_count += len(labels)
                validation = evaluate(model, VALIDATION_LOADER)
                scheduler.step(validation["loss"])
                improved = validation["accuracy"] > best_accuracy + 1e-8 or (
                    math.isclose(validation["accuracy"], best_accuracy, abs_tol=1e-8)
                    and validation["loss"] < best_loss
                )
                if improved:
                    best_accuracy = validation["accuracy"]
                    best_loss = validation["loss"]
                    best_epoch = epoch
                    stale = 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "model_name": name,
                        "epoch": epoch,
                        "validation_accuracy": best_accuracy,
                        "validation_loss": best_loss,
                        "train_mean": TRAIN_MEAN,
                        "train_std": TRAIN_STD,
                        "experiment_contract": EXPERIMENT_CONTRACT,
                        "test_accessed": False,
                    }, checkpoint_path)
                elif epoch >= MINIMUM_EPOCHS:
                    stale += 1
                row = {
                    "epoch": epoch,
                    "training_loss": train_loss / train_count,
                    "training_accuracy": train_correct / train_count,
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                    "validation_top3_accuracy": validation["top3_accuracy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "selected_checkpoint": improved,
                }
                history.append(row)
                pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
                print(
                    f"{name} epoch {epoch:03d}: train={row['training_loss']:.4f} "
                    f"valid={row['validation_loss']:.4f} "
                    f"top1={100 * row['validation_accuracy']:.2f}% "
                    f"top3={100 * row['validation_top3_accuracy']:.2f}%"
                )
                if epoch >= MINIMUM_EPOCHS and stale >= PATIENCE:
                    break

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = factory()
            model.load_state_dict(checkpoint["model_state"])
            summary = {
                "model": name,
                "analysis_role": "training and validation model selection only",
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "best_validation_loss": best_loss,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "input_rate_hz": INPUT_RATE_HZ,
                "input_features_per_step": INPUT_FEATURES,
                "input_representation": "causal_10ms_rms_envelope_at_400hz",
                "causal": True,
                "test_accessed": False,
                "checkpoint": str(checkpoint_path),
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            return model, summary


        TRAINED = {}
        TRAINING_SUMMARIES = {}
        for model_name, model_factory in MODEL_FACTORIES.items():
            print(f"\n=== {model_name} ===")
            trained_model, training_summary = train_one(model_name, model_factory)
            TRAINED[model_name] = trained_model
            TRAINING_SUMMARIES[model_name] = training_summary

        validation_table = pd.DataFrame(TRAINING_SUMMARIES.values()).sort_values(
            "best_validation_accuracy", ascending=False
        )
        validation_table.to_csv(OUTPUT_ROOT / "validation_model_selection.csv", index=False)
        display(validation_table)
        print("All model choices are frozen. The next cell opens test labels.")
        """
    ),
    markdown(
        """
        ## 8. Held-out test evaluation

        Run this cell only after all six validation checkpoints are frozen. Do not
        tune architectures, epochs, or preprocessing after inspecting these test
        results.
        """
    ),
    code(
        r"""
        TEST_LOADER = make_loader("test")
        test_rows = []
        prediction_frames = []
        for name, model in TRAINED.items():
            model = model.to(DEVICE)
            result = evaluate(model, TEST_LOADER)
            test_rows.append({
                "model": name,
                "test_accuracy": result["accuracy"],
                "test_top3_accuracy": result["top3_accuracy"],
                "test_error_rate": 1.0 - result["accuracy"],
            })
            prediction_frames.append(pd.DataFrame({
                "model": name,
                "dataset_index": result["indices"],
                "speaker": SPEAKERS[result["indices"]],
                "repetition": REPETITIONS[result["indices"]],
                "reference": result["labels"],
                "prediction": result["predictions"],
            }))
            print(
                f"{name}: test top1={100 * result['accuracy']:.2f}% "
                f"top3={100 * result['top3_accuracy']:.2f}%"
            )

        TEST_RESULTS = pd.DataFrame(test_rows)
        PREDICTIONS = pd.concat(prediction_frames, ignore_index=True)
        TEST_RESULTS.to_csv(OUTPUT_ROOT / "heldout_test_accuracy.csv", index=False)
        PREDICTIONS.to_csv(OUTPUT_ROOT / "heldout_test_predictions.csv", index=False)
        display(TEST_RESULTS.sort_values("test_accuracy", ascending=False))
        """
    ),
    markdown(
        """
        ## 9. Parameters, MACs, FLOPs, size, and latency

        Learned MACs count executed `Conv1d` and `Linear` multiplication-accumulation
        operations for one one-second input. Compute cost uses the conventional
        approximation **2 FLOPs per MAC**. Batch normalization and nonlinearities
        are not included in learned MACs. Total estimated FLOPs add two arithmetic
        operations per envelope value for frozen mean subtraction and scaling,
        plus an explicit streaming RMS-envelope estimate (square, running-window
        add, and running-window subtract at 48 kHz). Square roots are reported
        separately because they are not meaningfully represented by one FLOP.
        """
    ),
    code(
        r"""
        def count_learned_macs(model, sample):
            total = 0
            handles = []

            def conv_hook(module, inputs, output):
                nonlocal total
                kernel = int(np.prod(module.kernel_size))
                multiplications = module.in_channels // module.groups * kernel
                total += int(output.numel() * multiplications)

            def linear_hook(module, inputs, output):
                nonlocal total
                total += int(output.numel() * module.in_features)

            for module in model.modules():
                if isinstance(module, nn.Conv1d):
                    handles.append(module.register_forward_hook(conv_hook))
                elif isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(linear_hook))
            model.eval()
            with torch.inference_mode():
                model(sample)
            for handle in handles:
                handle.remove()
            return total


        def fp32_state_size_bytes(model):
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            return len(buffer.getvalue())


        def benchmark_cpu_ms(model, sample):
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            local_model = copy.deepcopy(model).cpu().eval()
            local_sample = sample.cpu()
            with torch.inference_mode():
                for _ in range(LATENCY_WARMUP):
                    local_model(local_sample)
                timings = []
                for _ in range(LATENCY_REPEATS):
                    start = time.perf_counter()
                    local_model(local_sample)
                    timings.append(1000.0 * (time.perf_counter() - start))
            torch.set_num_threads(previous_threads)
            return float(np.median(timings)), float(np.percentile(timings, 95))


        def benchmark_gpu_ms(model, sample):
            if DEVICE.type != "cuda":
                return math.nan, math.nan
            local_model = copy.deepcopy(model).to(DEVICE).eval()
            local_sample = sample.to(DEVICE)
            with torch.inference_mode():
                for _ in range(LATENCY_WARMUP):
                    local_model(local_sample)
                torch.cuda.synchronize()
                timings = []
                for _ in range(LATENCY_REPEATS):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    local_model(local_sample)
                    end.record()
                    torch.cuda.synchronize()
                    timings.append(start.elapsed_time(end))
            return float(np.median(timings)), float(np.percentile(timings, 95))


        sample = torch.zeros(1, INPUT_SAMPLES, 1)
        efficiency_rows = []
        for name, model in TRAINED.items():
            cpu_median, cpu_p95 = benchmark_cpu_ms(model, sample)
            gpu_median, gpu_p95 = benchmark_gpu_ms(model, sample)
            macs = count_learned_macs(model.cpu(), sample)
            learned_flops = 2 * macs
            input_normalization_flops = 2 * INPUT_SCALARS_PER_SECOND
            envelope_front_end_ops = ENVELOPE_FRONT_END_ARITHMETIC_OPS_PER_SECOND
            test_result = TEST_RESULTS.loc[TEST_RESULTS["model"].eq(name)].iloc[0]
            architecture = architecture_audit.loc[
                architecture_audit["model"].eq(name)
            ].iloc[0]
            efficiency_rows.append({
                "model": name,
                "test_accuracy": float(test_result["test_accuracy"]),
                "test_top3_accuracy": float(test_result["test_top3_accuracy"]),
                "input_rate_hz": INPUT_RATE_HZ,
                "input_features_per_step": INPUT_FEATURES,
                "input_scalar_rate_per_second": INPUT_SCALARS_PER_SECOND,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "fp32_state_size_mib": fp32_state_size_bytes(model) / (1024 ** 2),
                "learned_macs_per_one_second_utterance": macs,
                "learned_flops_per_one_second_utterance": learned_flops,
                "input_normalization_flops_per_second": input_normalization_flops,
                "envelope_front_end_estimated_arithmetic_ops_per_second": envelope_front_end_ops,
                "envelope_square_roots_per_second": ENVELOPE_SQRTS_PER_SECOND,
                "total_estimated_flops_per_second": (
                    learned_flops + input_normalization_flops + envelope_front_end_ops
                ),
                "cpu_latency_median_ms_one_thread": cpu_median,
                "cpu_latency_p95_ms_one_thread": cpu_p95,
                "gpu_latency_median_ms_batch1": gpu_median,
                "gpu_latency_p95_ms_batch1": gpu_p95,
                "future_lookahead_ms": 0.0,
                "output_update_interval_ms": architecture["output_update_interval_ms"],
                "nominal_receptive_field_ms": architecture["nominal_receptive_field_ms"],
                "input_representation": "causal_10ms_rms_envelope_at_400hz",
            })

        COMPARISON = pd.DataFrame(efficiency_rows).sort_values(
            "test_accuracy", ascending=False
        )
        COMPARISON.to_csv(OUTPUT_ROOT / "digital_compact_400hz_comparison.csv", index=False)
        display(COMPARISON)
        """
    ),
    markdown("## 10. Confusion matrices and efficiency plots"),
    code(
        r"""
        figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        for axis, name in zip(axes.flat, MODEL_FACTORIES):
            frame = PREDICTIONS.loc[PREDICTIONS["model"].eq(name)]
            matrix = confusion_matrix(
                frame["reference"], frame["prediction"], labels=list(range(10))
            )
            normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            sns.heatmap(
                normalized, vmin=0, vmax=1, cmap="viridis", square=True,
                xticklabels=range(10), yticklabels=range(10), ax=axis,
                cbar=name == list(MODEL_FACTORIES)[-1],
            )
            accuracy = COMPARISON.loc[
                COMPARISON["model"].eq(name), "test_accuracy"
            ].iloc[0]
            axis.set_title(f"{name}\naccuracy={100 * accuracy:.2f}%")
            axis.set_xlabel("Predicted digit")
            axis.set_ylabel("Reference digit")
        figure.suptitle(
            "AudioMNIST Digital Compact Models: 400 Hz RMS Envelope",
            fontsize=16,
        )
        figure.savefig(OUTPUT_ROOT / "heldout_test_confusion_matrices.png", dpi=180)
        plt.show()

        plot_specs = [
            ("parameters", "Parameters"),
            ("fp32_state_size_mib", "FP32 state size (MiB)"),
            ("learned_macs_per_one_second_utterance", "Learned MACs per 1 s"),
            ("learned_flops_per_one_second_utterance", "Learned FLOPs per 1 s"),
            ("total_estimated_flops_per_second", "Total estimated FLOPs/s"),
            ("cpu_latency_median_ms_one_thread", "CPU latency (ms, one thread)"),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
        for axis, (column, label) in zip(axes.flat, plot_specs):
            axis.scatter(COMPARISON[column], 100 * COMPARISON["test_accuracy"], s=75)
            for row in COMPARISON.itertuples(index=False):
                axis.annotate(
                    row.model.replace("_1d", ""),
                    (getattr(row, column), 100 * row.test_accuracy),
                    xytext=(5, 4), textcoords="offset points", fontsize=8,
                )
            axis.set_xlabel(label)
            axis.set_ylabel("Held-out test accuracy (%)")
            axis.grid(alpha=0.25)
            axis.set_xscale("linear")
        figure.suptitle("AudioMNIST 400 Hz Accuracy-Efficiency Comparison", fontsize=16)
        figure.savefig(OUTPUT_ROOT / "accuracy_efficiency_frontiers.png", dpi=180)
        plt.show()
        """
    ),
    markdown(
        """
        ## 11. Core metrics and configurable edge-MCU scenario

        The table below isolates the four requested measured or statically counted
        quantities: held-out accuracy, parameter count, learned MACs, and learned
        FLOPs. The FLOP convention is `1 MAC = 2 FLOPs`.

        Duty cycle, sleep ratio, and energy are **deployment estimates**, not
        measurements from the Colab runtime. FLOPs do not map uniquely to MCU clock
        cycles. The notebook therefore reports a sensitivity sweep at 1, 2, and 4
        cycles per MAC. Replace these assumptions with cycle-accurate measurements
        from the final compiled firmware before publication.

        The default electrical scenario uses an STM32U595-style MCU at 64 MHz and
        3.3 V. Its editable current assumptions are based on the official
        [STM32U59xxx datasheet](https://www.st.com/resource/en/datasheet/stm32u595ai.pdf),
        which reports operating-point-dependent run current in uA/MHz and Stop 3
        current. They are not guaranteed values for a particular board.
        """
    ),
    code(
        r"""
        CORE_METRICS = COMPARISON[[
            "model",
            "test_accuracy",
            "test_top3_accuracy",
            "parameters",
            "learned_macs_per_one_second_utterance",
            "learned_flops_per_one_second_utterance",
            "total_estimated_flops_per_second",
        ]].copy()
        CORE_METRICS.to_csv(OUTPUT_ROOT / "accuracy_parameters_macs_flops.csv", index=False)
        display(CORE_METRICS)

        # Editable deployment assumptions. These model CPU compute only.
        MCU_NAME = "STM32U595-style scenario"
        MCU_CLOCK_HZ = 64_000_000
        MCU_CLOCK_MHZ = MCU_CLOCK_HZ / 1e6
        MCU_VOLTAGE_V = 3.3
        MCU_ACTIVE_UA_PER_MHZ = 40.0
        MCU_STOP3_CURRENT_UA = 2.2
        CYCLES_PER_MAC_SCENARIOS = (1.0, 2.0, 4.0)
        CYCLES_PER_NONMAC_FRONTEND_OP = 1.0
        INFERENCE_INTERVAL_SECONDS = 1.0
        BATTERY_CAPACITY_MAH = 220.0

        active_current_a = MCU_ACTIVE_UA_PER_MHZ * MCU_CLOCK_MHZ * 1e-6
        sleep_current_a = MCU_STOP3_CURRENT_UA * 1e-6
        deployment_rows = []
        for row in COMPARISON.itertuples(index=False):
            for cycles_per_mac in CYCLES_PER_MAC_SCENARIOS:
                estimated_cycles = (
                    row.learned_macs_per_one_second_utterance * cycles_per_mac
                    + (
                        row.input_normalization_flops_per_second
                        + row.envelope_front_end_estimated_arithmetic_ops_per_second
                    )
                    * CYCLES_PER_NONMAC_FRONTEND_OP
                )
                unconstrained_awake_s = estimated_cycles / MCU_CLOCK_HZ
                real_time_feasible = unconstrained_awake_s <= INFERENCE_INTERVAL_SECONDS
                awake_s = min(unconstrained_awake_s, INFERENCE_INTERVAL_SECONDS)
                sleep_s = max(INFERENCE_INTERVAL_SECONDS - awake_s, 0.0)
                duty_cycle = awake_s / INFERENCE_INTERVAL_SECONDS
                sleep_to_wake = sleep_s / awake_s if awake_s > 0 else math.inf
                active_energy_j = MCU_VOLTAGE_V * active_current_a * awake_s
                sleep_energy_j = MCU_VOLTAGE_V * sleep_current_a * sleep_s
                interval_energy_j = active_energy_j + sleep_energy_j
                average_current_a = (
                    active_current_a * awake_s + sleep_current_a * sleep_s
                ) / INFERENCE_INTERVAL_SECONDS
                battery_life_hours = (
                    BATTERY_CAPACITY_MAH / (average_current_a * 1e3)
                    if average_current_a > 0 else math.inf
                )
                deployment_rows.append({
                    "model": row.model,
                    "test_accuracy": row.test_accuracy,
                    "parameters": row.parameters,
                    "model_macs_per_1s": row.learned_macs_per_one_second_utterance,
                    "model_flops_per_1s": row.learned_flops_per_one_second_utterance,
                    "assumed_cycles_per_mac": cycles_per_mac,
                    "estimated_cycles_per_1s_audio": estimated_cycles,
                    "estimated_awake_ms_per_second": 1000.0 * unconstrained_awake_s,
                    "estimated_compute_duty_cycle_percent": 100.0 * duty_cycle,
                    "estimated_cpu_sleep_percent": 100.0 * (1.0 - duty_cycle),
                    "estimated_sleep_to_wake_ratio": sleep_to_wake,
                    "real_time_feasible_at_assumed_clock": real_time_feasible,
                    "estimated_active_compute_energy_uj": active_energy_j * 1e6,
                    "estimated_mcu_interval_energy_uj": interval_energy_j * 1e6,
                    "estimated_mcu_only_battery_life_days": battery_life_hours / 24.0,
                })

        MCU_ESTIMATES = pd.DataFrame(deployment_rows)
        MCU_ESTIMATES.to_csv(OUTPUT_ROOT / "mcu_duty_cycle_energy_sensitivity.csv", index=False)
        display(MCU_ESTIMATES)

        print(
            "Important: sleep percentage is CPU compute slack, not whole-device "
            "deep-sleep time. Continuous microphones, ADC/DMA, wake-up overhead, "
            "memory traffic, and regulators are excluded. Battery life is therefore "
            "an MCU-only upper bound, not a complete-device prediction."
        )
        """
    ),
    markdown(
        """
        ## 12. Sampling, ADC, data-flow, and latency analysis

        These plots distinguish the fabricated five-resonator proposal from this
        notebook's one-channel **digital 400 Hz envelope** baseline. The notebook
        first acquires the full-rate waveform and computes RMS before emitting the
        400 Hz representation. Therefore its model input is 400 values/s, but its
        source acquisition and digital front-end costs are not 400 samples/s.

        A 400 Hz ADC cannot directly reconstruct a 3.5 or 7.5 kHz pressure waveform;
        its direct-waveform Nyquist limit is 200 Hz. The envelope retains slow
        amplitude modulation after high-rate acquisition. A fabricated resonator
        system can instead perform analog band selection, but it still needs an
        analog or otherwise pre-ADC baseband readout to justify a native 400 Hz ADC.

        The ADC power graph includes only the idealized capacitor-switching term
        `P_dynamic = C_eff * V^2 * f_s`. Real ADC reference, bias, leakage, channel
        switching, and microphone power can dominate, so replace the assumed
        effective capacitance with measured hardware power for a final paper claim.
        """
    ),
    code(
        r"""
        # Editable system assumptions.
        SPEECH_MAX_FREQUENCY_HZ = 8_000
        RESONATOR_CENTERS_HZ = np.array([300, 1_750, 3_500, 4_500, 7_500], dtype=float)
        FABRICATED_CHANNELS = 5
        FABRICATED_RATE_PER_CHANNEL_HZ = 400
        DIGITAL_CHANNELS = 1
        DIGITAL_RATE_HZ = 16_000
        BITS_PER_SAMPLE = 16

        ADC_EFFECTIVE_SWITCHED_CAPACITANCE_PF = 10.0
        ADC_VOLTAGE_V = 3.3
        DIGITAL_LOGMEL_FRONTEND_OPS_PER_SECOND = 7_262_000.0
        FABRICATED_DIGITAL_OPS_PER_INPUT_VALUE = 3.0
        NOTEBOOK_DIGITAL_OPS_PER_INPUT_VALUE = 2.0

        # Illustrative front-end timing assumptions, not measured resonator Q values.
        DIGITAL_FRAME_BUFFER_MS = 25.0
        ACOUSTIC_PATH_LENGTH_M = 0.05
        SPEED_OF_SOUND_M_PER_S = 343.0
        ASSUMED_RESONATOR_Q = 4.0

        frequency = np.linspace(0, SPEECH_MAX_FREQUENCY_HZ, 1000)
        nyquist_required_rate = 2.0 * frequency
        figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
        axis.plot(frequency, nyquist_required_rate, linewidth=2.5,
                  label="Minimum rate for direct waveform reconstruction: 2f")
        axis.axhline(400, color="#d1495b", linewidth=2.5,
                     label="400 Hz low-rate stream")
        axis.scatter(RESONATOR_CENTERS_HZ, np.full(5, 400.0), s=75,
                     color="#00798c", zorder=3, label="Intended resonator bands")
        axis.axvspan(200, SPEECH_MAX_FREQUENCY_HZ, color="#d1495b", alpha=0.08)
        axis.annotate(
            "Above 200 Hz, 400 Hz direct sampling aliases.\n"
            "Low-rate band readout requires analog/baseband transduction.",
            xy=(4_500, 400), xytext=(2_900, 5_300),
            arrowprops={"arrowstyle": "->", "color": "#444444"}, fontsize=10,
        )
        axis.set(xlabel="Acoustic frequency (Hz)", ylabel="Required sampling rate (Hz)",
                 title="Nyquist Limit and the Requirement for Analog Feature Transduction")
        axis.set_xlim(0, SPEECH_MAX_FREQUENCY_HZ)
        axis.set_ylim(0, 2 * SPEECH_MAX_FREQUENCY_HZ * 1.03)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
        figure.savefig(OUTPUT_ROOT / "nyquist_and_analog_transduction.png", dpi=180)
        plt.show()

        sampling_rates = np.geomspace(400, EXPECTED_SOURCE_RATE_HZ, 300)
        capacitance_f = ADC_EFFECTIVE_SWITCHED_CAPACITANCE_PF * 1e-12
        one_channel_power_uw = (
            capacitance_f * ADC_VOLTAGE_V ** 2 * sampling_rates * 1e6
        )
        fabricated_power_uw = (
            capacitance_f * ADC_VOLTAGE_V ** 2
            * FABRICATED_RATE_PER_CHANNEL_HZ * FABRICATED_CHANNELS * 1e6
        )
        digital_power_uw = (
            capacitance_f * ADC_VOLTAGE_V ** 2 * DIGITAL_RATE_HZ * DIGITAL_CHANNELS * 1e6
        )
        envelope_source_power_uw = (
            capacitance_f * ADC_VOLTAGE_V ** 2 * EXPECTED_SOURCE_RATE_HZ * 1e6
        )
        figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
        axis.plot(sampling_rates, one_channel_power_uw, linewidth=2.5,
                  label="One-channel ideal switching component")
        axis.scatter([400], [fabricated_power_uw], s=100, color="#00798c",
                     label="Five channels x 400 Hz")
        axis.scatter([16_000], [digital_power_uw], s=100, color="#d1495b",
                     label="One channel x 16 kHz")
        axis.scatter([EXPECTED_SOURCE_RATE_HZ], [envelope_source_power_uw], s=100,
                     color="#6a4c93", label="Notebook RMS source x 48 kHz")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(
            xlabel="Sampling rate per channel (Hz)",
            ylabel="Idealized ADC dynamic switching power (uW)",
            title="ADC Switching-Power Scenario (Excludes Static and Analog Power)",
        )
        axis.grid(alpha=0.25, which="both")
        axis.legend()
        axis.text(
            0.02, 0.04,
            f"Per channel: 40x rate reduction\n"
            f"Aggregate five-channel samples: {DIGITAL_RATE_HZ/(5*400):.1f}x reduction",
            transform=axis.transAxes,
        )
        figure.savefig(OUTPUT_ROOT / "adc_dynamic_power_scenario.png", dpi=180)
        plt.show()

        system_names = ["Digital mono\n16 kHz", "Fabricated\n5 x 400 Hz",
                        "Notebook RMS\n48k to 1 x 400 Hz"]
        aggregate_samples = np.array(
            [16_000, 5 * 400, EXPECTED_SOURCE_RATE_HZ], dtype=float
        )
        representation_values = np.array([16_000, 5 * 400, 400], dtype=float)
        payload_kb_s = aggregate_samples * BITS_PER_SAMPLE / 8.0 / 1000.0
        payload_frame = pd.DataFrame({
            "system": system_names,
            "aggregate_adc_samples_per_second": aggregate_samples.astype(int),
            "model_input_values_per_second": representation_values.astype(int),
            "data_rate_kbit_per_second": aggregate_samples * BITS_PER_SAMPLE / 1000.0,
            "payload_kB_per_second": payload_kb_s,
        })
        payload_frame.to_csv(OUTPUT_ROOT / "data_ingestion_comparison.csv", index=False)
        display(payload_frame)

        figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
        bars = axis.bar(system_names, payload_kb_s,
                        color=["#d1495b", "#00798c", "#6a4c93"])
        axis.bar_label(bars, fmt="%.1f kB/s", padding=4)
        axis.set(ylabel="16-bit payload (decimal kB/s)",
                 title="Input Data Footprint per Second")
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(OUTPUT_ROOT / "input_data_footprint.png", dpi=180)
        plt.show()

        durations = np.linspace(1, 60, 60)
        fabricated_frontend_ops_s = (
            FABRICATED_CHANNELS * FABRICATED_RATE_PER_CHANNEL_HZ
            * FABRICATED_DIGITAL_OPS_PER_INPUT_VALUE
        )
        notebook_frontend_ops_s = (
            ENVELOPE_FRONT_END_ARITHMETIC_OPS_PER_SECOND
            + INPUT_SCALARS_PER_SECOND * NOTEBOOK_DIGITAL_OPS_PER_INPUT_VALUE
        )
        figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
        axis.plot(durations, DIGITAL_LOGMEL_FRONTEND_OPS_PER_SECOND * durations,
                  linewidth=2.5, label="Digital 40-bin log-mel estimate")
        axis.plot(durations, fabricated_frontend_ops_s * durations,
                  linewidth=2.5, label="Five-channel direct feature handling")
        axis.plot(durations, notebook_frontend_ops_s * durations,
                  linewidth=2.5, label="One-channel 48 kHz RMS to 400 Hz envelope")
        axis.set_yscale("log")
        axis.set(
            xlabel="Audio duration (s)", ylabel="Cumulative estimated operations",
            title="Front-End Arithmetic Scales Linearly with Stream Duration",
        )
        axis.grid(alpha=0.25, which="both")
        axis.legend()
        figure.savefig(OUTPUT_ROOT / "frontend_compute_scaling.png", dpi=180)
        plt.show()

        acoustic_travel_ms = 1000.0 * ACOUSTIC_PATH_LENGTH_M / SPEED_OF_SOUND_M_PER_S
        adc_period_ms = 1000.0 / FABRICATED_RATE_PER_CHANNEL_HZ
        tau_ms = 1000.0 * ASSUMED_RESONATOR_Q / (np.pi * RESONATOR_CENTERS_HZ)
        latency_frame = pd.DataFrame({
            "resonator_center_hz": RESONATOR_CENTERS_HZ,
            "assumed_q": ASSUMED_RESONATOR_Q,
            "acoustic_travel_ms": acoustic_travel_ms,
            "adc_period_ms": adc_period_ms,
            "resonator_amplitude_time_constant_ms": tau_ms,
            "first_sample_bound_ms": acoustic_travel_ms + adc_period_ms,
            "approx_95pct_settling_bound_ms": acoustic_travel_ms + adc_period_ms + 3 * tau_ms,
        })
        latency_frame.to_csv(OUTPUT_ROOT / "frontend_latency_bounds.csv", index=False)
        display(latency_frame)

        figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
        labels = [f"R{int(f)}" for f in RESONATOR_CENTERS_HZ] + ["25 ms log-mel\nframe buffer"]
        values = list(latency_frame["approx_95pct_settling_bound_ms"]) + [DIGITAL_FRAME_BUFFER_MS]
        bars = axis.bar(labels, values, color=["#00798c"] * 5 + ["#d1495b"])
        axis.bar_label(bars, fmt="%.1f ms", padding=3)
        axis.set(ylabel="Illustrative first-feature latency bound (ms)",
                 title="Front-End Latency Is Low but Not Zero")
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(OUTPUT_ROOT / "frontend_latency_bounds.png", dpi=180)
        plt.show()
        """
    ),
    markdown(
        """
        ## 13. Reporting notes

        Report the following together:

        - Model input is one digital channel with 400 nonnegative RMS-envelope
          values per second, computed using a causal 10 ms window and 2.5 ms hop.
        - The envelope is derived from the full-rate AudioMNIST waveform. It is not
          direct 400 Hz waveform sampling and does not support a native 400 Hz ADC
          claim for the digital baseline.
        - The fabricated device has five physical resonator channels, so its input
          is 5D at 400 Hz, or 2,000 scalar values/s. This notebook's digital input
          is 1D at 400 Hz, or 400 scalar values/s.
        - Each model is causal internally, but AudioMNIST scoring occurs at the
          known one-second utterance endpoint.
        - FLOPs use 2 FLOPs per MAC. Batch normalization, activations, memory
          movement, and acquisition are excluded unless separately stated.
        - MCU duty cycle and energy are sensitivity estimates. Report them as
          projections until the compiled model is profiled on the selected MCU.
        - A five-channel 400 Hz system processes 2,000 ADC samples/s, which is 8x
          fewer aggregate samples than one 16 kHz channel, not 40x. The 40x figure
          applies per channel only.
        - Resonance performs analog band selection but does not by itself guarantee
          a baseband envelope. A real fabricated 400 Hz acquisition path needs an
          analog envelope/energy detector or a sensor state already varying below
          200 Hz; otherwise high-frequency microphone pressure aliases.
        - Analog resonators, microphones, ADCs, and regulators consume energy even
          when digital front-end arithmetic is small; they must be measured for a
          complete joules-per-inference or battery-life claim.
        - These are 1D architecture adaptations, not official scores for the named
          model families.
        - Do not compare these results to the fabricated alphabet accuracy. Compare
          them only with the fabricated AudioMNIST result on the same speaker split.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": OUTPUT.name,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
