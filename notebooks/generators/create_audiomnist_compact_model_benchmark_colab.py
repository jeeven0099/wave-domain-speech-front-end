from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audiomnist_causal_streaming_benchmark_colab.ipynb"


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
        # AudioMNIST causal-streaming efficiency benchmark

        This notebook trains and evaluates eight isolated-digit classifiers on the
        **same speaker-disjoint AudioMNIST split used by the fabricated resonator
        experiment**:

        | Split | Speakers | Utterances |
        |---|---:|---:|
        | Train | 36 | 18,000 |
        | Validation | 12 | 6,000 |
        | Held-out test | 12 | 6,000 |

        Models:

        1. MatchboxNet 3x1x64
        2. TC-ResNet8
        3. MLPerf Tiny reference DS-CNN
        4. BC-ResNet-1
        5. BC-ResNet-2
        6. EdgeSpeechNet-A
        7. Whisper Tiny encoder classifier
        8. Whisper Base encoder classifier

        The six compact models are trained from scratch on one shared **causal
        40-bin log-Mel, 50 ms window, 10 ms hop** representation. All temporal
        convolutions use past-only padding, and a running mean of frame logits can
        produce an updated digit decision every 10 ms. Whisper uses its native
        80-bin log-Mel transform and externally pretrained encoder weights. Its
        text decoder is deliberately removed because AudioMNIST is a ten-class
        isolated-word task. Whisper results are therefore marked as pretrained and
        must not be described as from-scratch comparisons.

        The compact implementations are faithful AudioMNIST adaptations of the
        published model families, but these are **not** official MLPerf submissions
        or reproduced paper scores. Architecture sources:
        [AudioMNIST](https://github.com/soerenab/AudioMNIST),
        [MatchboxNet](https://arxiv.org/abs/2004.08531),
        [TC-ResNet](https://arxiv.org/abs/1904.03814),
        [MLPerf Tiny DS-CNN reference](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py),
        [BC-ResNet](https://github.com/Qualcomm-AI-research/bcresnet),
        [EdgeSpeechNet](https://arxiv.org/abs/1810.08559), and
        [OpenAI Whisper](https://github.com/openai/whisper).

        AudioMNIST contains one known digit label per utterance, so the correct
        decoder is a ten-class softmax (top-1/top-3). CTC beam search is not used;
        it is reserved for the continuous alphabet experiment.

        **Reported metrics:** held-out top-1 and top-3 accuracy, total/trainable
        parameters, FP32 state size, learned multiply-accumulates (MACs), measured
        front-end and model latency, and an analytical front-end operation count.
        File I/O is excluded.
        """
    ),
    markdown(
        """
        ## 1. Colab setup

        Select **Runtime > Change runtime type > T4 GPU** before running. The
        notebook saves a best-validation checkpoint after every model, so it can
        resume after a runtime interruption.
        """
    ),
    code(
        r"""
        %pip install -q openai-whisper soundfile
        """
    ),
    code(
        r"""
        import copy
        import hashlib
        import json
        import math
        import os
        import platform
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
        import torchaudio
        import whisper
        from IPython.display import display
        from scipy import signal
        from sklearn.metrics import accuracy_score, confusion_matrix
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
        print("Training device:", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    markdown("## 2. Configuration and exact experiment contract"),
    code(
        r"""
        IN_COLAB = "google.colab" in str(get_ipython())
        USE_GOOGLE_DRIVE = True
        REUSE_CHECKPOINTS = True
        QUICK_SMOKE_TEST = False  # True: 2 epochs; never use quick-mode scores in a paper.

        if IN_COLAB and USE_GOOGLE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")

        DATASET_ROOT = Path("/content/AudioMNIST") if IN_COLAB else Path("datasets/AudioMNIST")
        CACHE_ROOT = Path("/content/audiomnist_compact_cache") if IN_COLAB else Path(".cache/audiomnist_compact")
        if IN_COLAB and USE_GOOGLE_DRIVE:
            OUTPUT_ROOT = Path(
                "/content/drive/MyDrive/audiomnist_causal_streaming_benchmark_v2"
            )
        else:
            OUTPUT_ROOT = Path("audiomnist_causal_streaming_benchmark_v2_outputs")

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        SAMPLE_RATE = 16_000
        FIXED_SECONDS = 1.0
        TARGET_SAMPLES = int(SAMPLE_RATE * FIXED_SECONDS)
        TARGET_PEAK_DBFS = -9.0
        N_MELS = 40
        N_FFT = 800
        WINDOW_MS = 50.0
        HOP_MS = 10.0
        WIN_LENGTH = int(SAMPLE_RATE * WINDOW_MS / 1000)
        HOP_LENGTH = int(SAMPLE_RATE * HOP_MS / 1000)
        COMPACT_FRAMES = (TARGET_SAMPLES - N_FFT) // HOP_LENGTH + 1
        WHISPER_MELS = 80
        WHISPER_FRAMES = 100
        WHISPER_N_FFT = 400
        WHISPER_HOP_LENGTH = 160
        CAUSAL_CONTRACT_VERSION = 2

        COMPACT_BATCH_SIZE = 128
        COMPACT_EPOCHS = 50
        COMPACT_MINIMUM_EPOCHS = 20
        COMPACT_PATIENCE = 8
        COMPACT_LR = 1e-3
        WHISPER_EPOCHS = 12
        WHISPER_MINIMUM_EPOCHS = 4
        WHISPER_PATIENCE = 3
        WHISPER_LAST_BLOCKS = 2
        WHISPER_ENCODER_LR = 1e-5
        WHISPER_HEAD_LR = 5e-4
        WHISPER_BATCH_SIZE = {"tiny": 128, "base": 64}
        NUM_WORKERS = 2 if IN_COLAB else 0
        LATENCY_WARMUP = 10
        LATENCY_REPEATS = 50

        if QUICK_SMOKE_TEST:
            COMPACT_EPOCHS = 2
            WHISPER_EPOCHS = 2
            COMPACT_MINIMUM_EPOCHS = 1
            WHISPER_MINIMUM_EPOCHS = 1
            COMPACT_PATIENCE = 2
            WHISPER_PATIENCE = 2

        EXPERIMENT_CONTRACT = {
            "seed": SEED,
            "causal_contract_version": CAUSAL_CONTRACT_VERSION,
            "task": "AudioMNIST isolated digit classification",
            "split": "same speaker-disjoint split as fabricated resonator experiment",
            "compact_frontend": {
                "sample_rate_hz": SAMPLE_RATE,
                "duration_s": FIXED_SECONDS,
                "n_mels": N_MELS,
                "window_ms": WINDOW_MS,
                "hop_ms": HOP_MS,
                "peak_normalization_dbfs": TARGET_PEAK_DBFS,
                "stft_center": False,
                "feature_rate_hz": SAMPLE_RATE / HOP_LENGTH,
            },
            "whisper_frontend": "OpenAI Whisper native 80-bin log-Mel; one-second short context",
            "decoder": "10-class softmax; no CTC because each example has one digit label",
            "decision_protocol": (
                "score the one-second utterance at its known endpoint; running-mean "
                "scores are available causally before the endpoint"
            ),
            "causal_compact_contract": (
                "past-only temporal padding; zero future lookahead; running-mean "
                "classification state can update at each encoder output"
            ),
            "test_policy": "test labels evaluated once after all validation-based model selection",
        }
        (OUTPUT_ROOT / "experiment_contract.json").write_text(
            json.dumps(EXPERIMENT_CONTRACT, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(EXPERIMENT_CONTRACT, indent=2))
        """
    ),
    markdown("## 3. AudioMNIST inventory and speaker-disjoint split"),
    code(
        r"""
        if not (DATASET_ROOT / "data").is_dir():
            if DATASET_ROOT.exists():
                raise RuntimeError(f"{DATASET_ROOT} exists but does not contain data/")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/soerenab/AudioMNIST.git", str(DATASET_ROOT)],
                check=True,
            )

        SPLIT_SPEAKERS = {
            "train": {28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2,
                      8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55},
            "validation": {12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50},
            "test": {26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51},
        }
        assert not (SPLIT_SPEAKERS["train"] & SPLIT_SPEAKERS["validation"])
        assert not (SPLIT_SPEAKERS["train"] & SPLIT_SPEAKERS["test"])
        assert not (SPLIT_SPEAKERS["validation"] & SPLIT_SPEAKERS["test"])
        assert len(set.union(*SPLIT_SPEAKERS.values())) == 60

        speaker_to_split = {
            speaker: split_name
            for split_name, speakers in SPLIT_SPEAKERS.items()
            for speaker in speakers
        }
        rows = []
        for path in sorted((DATASET_ROOT / "data").glob("[0-9][0-9]/*.wav")):
            digit_text, speaker_text, repetition_text = path.stem.split("_")
            speaker = int(speaker_text)
            rows.append({
                "path": str(path),
                "digit": int(digit_text),
                "speaker": speaker,
                "repetition": int(repetition_text),
                "split": speaker_to_split[speaker],
            })

        records = pd.DataFrame(rows).sort_values(
            ["split", "speaker", "digit", "repetition"], kind="stable"
        ).reset_index(drop=True)
        records["record_index"] = np.arange(len(records), dtype=np.int32)
        split_counts = records.groupby("split").size().reindex(["train", "validation", "test"])

        assert len(records) == 30_000, f"Expected 30,000 files, found {len(records):,}"
        assert split_counts.to_dict() == {"train": 18_000, "validation": 6_000, "test": 6_000}
        assert records.groupby(["speaker", "digit"]).size().eq(50).all()
        display(split_counts.rename("utterances").to_frame())
        display(records.groupby(["split", "digit"]).size().unstack(0))
        """
    ),
    markdown(
        """
        ## 4. Deterministic waveform and feature cache

        Every utterance is resampled to 16 kHz, peak-normalized to -9 dBFS,
        center-cropped if necessary, and symmetrically zero-padded to one second.
        Cache construction does not use labels and cannot tune a model to the test
        set. Compact features use `center=False`, so no STFT frame reads samples
        beyond its availability time. Compact and Whisper features are then
        computed in GPU batches.
        """
    ),
    code(
        r"""
        def read_mono(path):
            values, sample_rate = sf.read(path, always_2d=False)
            values = np.asarray(values, dtype=np.float32)
            if values.ndim == 2:
                values = values.mean(axis=1)
            return int(sample_rate), values


        def resample_audio(values, source_rate, target_rate=SAMPLE_RATE):
            if source_rate == target_rate:
                return values.astype(np.float32, copy=False)
            divisor = math.gcd(source_rate, target_rate)
            return signal.resample_poly(
                values, target_rate // divisor, source_rate // divisor
            ).astype(np.float32)


        def normalize_and_fix_length(values):
            peak = float(np.max(np.abs(values))) if len(values) else 0.0
            if peak > 1e-9:
                values = values * (10.0 ** (TARGET_PEAK_DBFS / 20.0) / peak)
            if len(values) > TARGET_SAMPLES:
                start = (len(values) - TARGET_SAMPLES) // 2
                values = values[start:start + TARGET_SAMPLES]
            if len(values) < TARGET_SAMPLES:
                missing = TARGET_SAMPLES - len(values)
                values = np.pad(values, (missing // 2, missing - missing // 2))
            return values.astype(np.float32)


        cache_contract = {
            **EXPERIMENT_CONTRACT["compact_frontend"],
            "whisper_mels": WHISPER_MELS,
            "record_count": len(records),
            "dataset_head": records[["path", "digit", "speaker", "repetition"]].head(10).to_dict("records"),
        }
        cache_hash = hashlib.sha256(
            json.dumps(cache_contract, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        CACHE_DIR = CACHE_ROOT / cache_hash
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        COMPLETE_FLAG = CACHE_DIR / "complete.json"


        def make_compact_mel(device):
            return torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                win_length=WIN_LENGTH,
                hop_length=HOP_LENGTH,
                f_min=20.0,
                f_max=SAMPLE_RATE / 2,
                n_mels=N_MELS,
                power=2.0,
                center=False,
                norm=None,
                mel_scale="htk",
            ).to(device)


        def whisper_logmel_batch(waveform):
            # OpenAI Whisper log-Mel transform with per-utterance dynamic range.
            window = torch.hann_window(WHISPER_N_FFT, device=waveform.device)
            spectrum = torch.stft(
                waveform, WHISPER_N_FFT, WHISPER_HOP_LENGTH,
                window=window, return_complex=True,
            )
            power = spectrum[..., :-1].abs().square()
            filters = whisper.audio.mel_filters(waveform.device, WHISPER_MELS)
            mel = filters @ power
            log_mel = mel.clamp_min(1e-10).log10()
            floor = log_mel.amax(dim=(-2, -1), keepdim=True) - 8.0
            return (torch.maximum(log_mel, floor) + 4.0) / 4.0


        def prepare_cache():
            if COMPLETE_FLAG.is_file():
                print("Using complete cache:", CACHE_DIR)
                return

            count = len(records)
            wave_path = CACHE_DIR / "waveforms.npy"
            wave_map = np.lib.format.open_memmap(
                wave_path, mode="w+", dtype=np.float16, shape=(count, TARGET_SAMPLES)
            )
            for row in tqdm(records.itertuples(index=False), total=count, desc="Reading AudioMNIST WAVs"):
                sample_rate, raw = read_mono(row.path)
                wave = normalize_and_fix_length(resample_audio(raw, sample_rate))
                wave_map[int(row.record_index)] = wave.astype(np.float16)
            wave_map.flush()
            del wave_map

            wave_map = np.load(wave_path, mmap_mode="r")
            compact_map = np.lib.format.open_memmap(
                CACHE_DIR / "logmel40.npy", mode="w+", dtype=np.float16,
                shape=(count, COMPACT_FRAMES, N_MELS),
            )
            whisper_map = np.lib.format.open_memmap(
                CACHE_DIR / "whisper80.npy", mode="w+", dtype=np.float16,
                shape=(count, WHISPER_FRAMES, WHISPER_MELS),
            )
            compact_mel = make_compact_mel(DEVICE)
            feature_batch = 256 if DEVICE.type == "cuda" else 32
            with torch.inference_mode():
                for start in tqdm(range(0, count, feature_batch), desc="Computing log-Mel caches"):
                    stop = min(count, start + feature_batch)
                    waveform = torch.from_numpy(
                        np.asarray(wave_map[start:stop], dtype=np.float32)
                    ).to(DEVICE)
                    compact = torch.log(
                        compact_mel(waveform).clamp_min(1e-8)
                    ).transpose(1, 2)
                    whisper_feature = whisper_logmel_batch(waveform).transpose(1, 2)
                    assert compact.shape[1:] == (COMPACT_FRAMES, N_MELS), compact.shape
                    assert whisper_feature.shape[1:] == (WHISPER_FRAMES, WHISPER_MELS), whisper_feature.shape
                    compact_map[start:stop] = compact.cpu().numpy().astype(np.float16)
                    whisper_map[start:stop] = whisper_feature.cpu().numpy().astype(np.float16)

            compact_map.flush()
            whisper_map.flush()
            del compact_map, whisper_map, wave_map
            records.to_csv(CACHE_DIR / "inventory.csv", index=False)
            COMPLETE_FLAG.write_text(json.dumps(cache_contract, indent=2) + "\n", encoding="utf-8")
            print("Created cache:", CACHE_DIR)


        prepare_cache()
        WAVEFORMS = np.load(CACHE_DIR / "waveforms.npy", mmap_mode="r")
        LOGMEL40 = np.load(CACHE_DIR / "logmel40.npy", mmap_mode="r")
        WHISPER80 = np.load(CACHE_DIR / "whisper80.npy", mmap_mode="r")
        LABELS = records.digit.to_numpy(np.int64)
        SPEAKERS = records.speaker.to_numpy(np.int64)
        SPLITS = records.split.to_numpy()
        INDICES = {name: np.flatnonzero(SPLITS == name) for name in ("train", "validation", "test")}
        print("Waveforms:", WAVEFORMS.shape)
        print("Compact log-Mel:", LOGMEL40.shape)
        print("Whisper log-Mel:", WHISPER80.shape)
        """
    ),
    markdown("## 5. Train-only normalization and data loaders"),
    code(
        r"""
        train_values = np.asarray(LOGMEL40[INDICES["train"]], dtype=np.float32)
        COMPACT_MEAN = train_values.mean(axis=(0, 1)).astype(np.float32)
        COMPACT_STD = np.maximum(train_values.std(axis=(0, 1)), 1e-4).astype(np.float32)
        del train_values


        class FeatureDataset(Dataset):
            def __init__(self, values, indices, mean=None, std=None):
                self.values = values
                self.indices = np.asarray(indices, dtype=np.int64)
                self.mean = mean
                self.std = std

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                index = int(self.indices[item])
                values = np.array(self.values[index], dtype=np.float32, copy=True)
                if self.mean is not None:
                    values = (values - self.mean) / self.std
                return torch.from_numpy(values), int(LABELS[index]), index


        def make_loaders(values, batch_size, mean=None, std=None, include_test=False):
            split_names = ["train", "validation"] + (["test"] if include_test else [])
            datasets = {
                split_name: FeatureDataset(values, INDICES[split_name], mean, std)
                for split_name in split_names
            }
            return {
                split_name: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=split_name == "train",
                    num_workers=NUM_WORKERS,
                    pin_memory=DEVICE.type == "cuda",
                    persistent_workers=NUM_WORKERS > 0,
                    generator=torch.Generator().manual_seed(SEED),
                )
                for split_name, dataset in datasets.items()
            }


        compact_loaders = make_loaders(
            LOGMEL40, COMPACT_BATCH_SIZE, COMPACT_MEAN, COMPACT_STD, include_test=False
        )
        np.savez(OUTPUT_ROOT / "compact_train_normalizer.npz", mean=COMPACT_MEAN, std=COMPACT_STD)
        """
    ),
    markdown(
        r"""
        ## 6. Compact architectures

        `MatchboxNet3x1x64` uses three time-channel-separable residual blocks
        (`B=3`, `R=1`, `C=64`). `TCResNet8` treats Mel bins as input channels and
        applies temporal convolutions. The DS-CNN follows the MLPerf Tiny reference
        layer pattern but changes the final class count from 12 to 10. BC-ResNet
        width multipliers 1 and 2 share the published broadcast-residual topology.
        EdgeSpeechNet-A uses the paper's bottleneck sequence
        `20, 15, 25, 22, 22, 25` around a 39-channel residual trunk.
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


        class ConvBNAct1d(nn.Sequential):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
                super().__init__(
                    CausalConv1d(in_channels, out_channels, kernel_size, stride=stride,
                                 dilation=dilation, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )


        class TCSConv1d(nn.Sequential):
            def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=True):
                layers = [
                    CausalConv1d(in_channels, in_channels, kernel_size,
                                 dilation=dilation, groups=in_channels, bias=False),
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                ]
                if activation:
                    layers.append(nn.ReLU(inplace=True))
                super().__init__(*layers)


        class MatchboxResidualBlock(nn.Module):
            def __init__(self, in_channels, channels, kernel_size):
                super().__init__()
                self.main = TCSConv1d(in_channels, channels, kernel_size, activation=False)
                self.residual = (
                    nn.Sequential(nn.Conv1d(in_channels, channels, 1, bias=False), nn.BatchNorm1d(channels))
                    if in_channels != channels else nn.Identity()
                )

            def forward(self, values):
                return F.relu(self.main(values) + self.residual(values), inplace=True)


        class MatchboxNet3x1x64(nn.Module):
            temporal_stride = 1
            temporal_receptive_field_frames = 109

            def __init__(self, num_classes=10):
                super().__init__()
                self.prologue = TCSConv1d(N_MELS, 128, 11)
                self.blocks = nn.Sequential(
                    MatchboxResidualBlock(128, 64, 13),
                    MatchboxResidualBlock(64, 64, 15),
                    MatchboxResidualBlock(64, 64, 17),
                )
                self.epilogue = nn.Sequential(
                    TCSConv1d(64, 128, 29, dilation=2),
                    ConvBNAct1d(128, 128, 1),
                )
                self.classifier = nn.Linear(128, num_classes)

            def encode_frames(self, values):
                values = values.transpose(1, 2)
                values = self.epilogue(self.blocks(self.prologue(values)))
                return self.classifier(values.transpose(1, 2))

            def forward(self, values):
                return self.encode_frames(values).mean(dim=1)


        class TCResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride):
                super().__init__()
                self.main = nn.Sequential(
                    ConvBNAct1d(in_channels, out_channels, 3, stride=stride),
                    CausalConv1d(out_channels, out_channels, 3, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                self.skip = (
                    nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                                  nn.BatchNorm1d(out_channels))
                    if in_channels != out_channels or stride != 1 else nn.Identity()
                )

            def forward(self, values):
                return F.relu(self.main(values) + self.skip(values), inplace=True)


        class TCResNet8(nn.Module):
            temporal_stride = 8
            temporal_receptive_field_frames = 45

            def __init__(self, num_classes=10):
                super().__init__()
                self.stem = ConvBNAct1d(N_MELS, 16, 3)
                self.blocks = nn.Sequential(
                    TCResidualBlock(16, 24, 2),
                    TCResidualBlock(24, 32, 2),
                    TCResidualBlock(32, 48, 2),
                )
                self.classifier = nn.Linear(48, num_classes)

            def encode_frames(self, values):
                values = self.blocks(self.stem(values.transpose(1, 2)))
                return self.classifier(values.transpose(1, 2))

            def forward(self, values):
                return self.encode_frames(values).mean(dim=1)


        class CausalTimeFirstSamePadConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
                super().__init__()
                self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
                self.stride = stride if isinstance(stride, tuple) else (stride, stride)
                self.conv = nn.Conv2d(
                    in_channels, out_channels, self.kernel_size, stride=self.stride,
                    groups=groups, bias=bias,
                )

            def forward(self, values):
                height, width = values.shape[-2:]
                out_h = math.ceil(height / self.stride[0])
                out_w = math.ceil(width / self.stride[1])
                pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - height, 0)
                pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - width, 0)
                values = F.pad(values, (pad_w // 2, pad_w - pad_w // 2,
                                        pad_h, 0))
                return self.conv(values)


        class DSConvBlock(nn.Sequential):
            def __init__(self, channels):
                super().__init__(
                    CausalTimeFirstSamePadConv2d(
                        channels, channels, 3, 1, groups=channels, bias=False
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )


        class MLPerfTinyDSCNN(nn.Module):
            temporal_stride = 2
            temporal_receptive_field_frames = 26

            def __init__(self, num_classes=10):
                super().__init__()
                self.stem = nn.Sequential(
                    CausalTimeFirstSamePadConv2d(
                        1, 64, (10, 4), (2, 2), bias=False
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.blocks = nn.Sequential(*[DSConvBlock(64) for _ in range(4)])
                self.classifier = nn.Linear(64, num_classes)

            def encode_frames(self, values):
                values = values.unsqueeze(1)  # B,T,F -> B,1,T,F
                values = self.blocks(self.stem(values)).mean(dim=-1).transpose(1, 2)
                return self.classifier(values)

            def forward(self, values):
                return self.encode_frames(values).mean(dim=1)


        class SubSpectralNorm(nn.Module):
            def __init__(self, channels, groups=5):
                super().__init__()
                self.groups = groups
                self.norm = nn.BatchNorm2d(channels * groups)

            def forward(self, values):
                batch, channels, frequency, time_steps = values.shape
                if frequency % self.groups:
                    raise RuntimeError(f"Frequency dimension {frequency} is not divisible by {self.groups}")
                values = values.reshape(batch, channels * self.groups,
                                        frequency // self.groups, time_steps)
                values = self.norm(values)
                return values.reshape(batch, channels, frequency, time_steps)


        class CausalTimeLastConv2d(nn.Module):
            # Symmetric frequency padding and past-only temporal padding.

            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         dilation=1, groups=1, bias=False, frequency_padding="same"):
                super().__init__()
                kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
                stride = stride if isinstance(stride, tuple) else (stride,) * 2
                dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 2
                frequency_context = dilation[0] * (kernel_size[0] - 1)
                if frequency_padding == "same":
                    self.frequency_padding = (
                        frequency_context // 2,
                        frequency_context - frequency_context // 2,
                    )
                else:
                    padding = int(frequency_padding)
                    self.frequency_padding = (padding, padding)
                self.time_left_context = dilation[1] * (kernel_size[1] - 1)
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride=stride,
                    dilation=dilation, groups=groups, bias=bias,
                )

            def forward(self, values):
                frequency_top, frequency_bottom = self.frequency_padding
                return self.conv(F.pad(values, (
                    self.time_left_context, 0, frequency_top, frequency_bottom
                )))


        class BCConvBNAct(nn.Module):
            def __init__(self, in_channels, out_channels, stage_index, kernel_size=3,
                         stride=1, groups=1, use_dilation=False, activation="relu",
                         batch_norm=True, subspectral_norm=False):
                super().__init__()
                kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
                stride = stride if isinstance(stride, tuple) else (stride, stride)
                dilation = 2 ** stage_index if use_dilation else 1
                layers = [CausalTimeLastConv2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    dilation=(dilation, dilation), groups=groups, bias=False,
                )]
                if subspectral_norm:
                    layers.append(SubSpectralNorm(out_channels, groups=5))
                elif batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "silu":
                    layers.append(nn.SiLU(inplace=True))
                elif activation is not None:
                    raise ValueError(activation)
                self.block = nn.Sequential(*layers)

            def forward(self, values):
                return self.block(values)


        class BCResBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stage_index, stride):
                super().__init__()
                self.transition = in_channels != out_channels
                frequency_layers = []
                if self.transition:
                    frequency_layers.append(BCConvBNAct(
                        in_channels, out_channels, stage_index, kernel_size=1
                    ))
                    in_channels = out_channels
                frequency_layers.append(BCConvBNAct(
                    in_channels, out_channels, stage_index, kernel_size=(3, 1),
                    stride=(stride, 1), groups=out_channels, activation=None,
                    subspectral_norm=True,
                ))
                self.frequency_layers = nn.Sequential(*frequency_layers)
                self.temporal = nn.Sequential(
                    BCConvBNAct(out_channels, out_channels, stage_index, kernel_size=(1, 3),
                                groups=out_channels, use_dilation=True, activation="silu"),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.Dropout2d(0.1),
                )

            def forward(self, values):
                frequency = self.frequency_layers(values)
                temporal = self.temporal(frequency.mean(dim=2, keepdim=True))
                output = frequency + temporal
                if not self.transition:
                    output = output + values
                return F.relu(output, inplace=True)


        class BCResNet(nn.Module):
            temporal_stride = 1
            temporal_receptive_field_frames = 117

            def __init__(self, scale=1, num_classes=10):
                super().__init__()
                base = 8 * scale
                channels = [16 * scale, 8 * scale, 12 * scale,
                            16 * scale, 20 * scale, 32 * scale]
                self.stem = nn.Sequential(
                    CausalTimeLastConv2d(
                        1, channels[0], 5, stride=(2, 1), bias=False
                    ),
                    nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
                )
                stages = []
                for stage_index, repeats in enumerate((2, 2, 4, 4)):
                    blocks = []
                    in_channels = channels[stage_index]
                    out_channels = channels[stage_index + 1]
                    for block_index in range(repeats):
                        stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                        blocks.append(BCResBlock(in_channels, out_channels, stage_index, stride))
                        in_channels = out_channels
                    stages.append(nn.Sequential(*blocks))
                self.stages = nn.Sequential(*stages)
                self.final_features = nn.Sequential(
                    CausalTimeLastConv2d(
                        channels[4], channels[4], (5, 5),
                        groups=channels[4], bias=False, frequency_padding=0
                    ),
                    nn.Conv2d(channels[4], channels[5], 1, bias=False),
                    nn.BatchNorm2d(channels[5]), nn.ReLU(inplace=True),
                )
                self.classifier = nn.Linear(channels[5], num_classes)

            def encode_frames(self, values):
                values = values.transpose(1, 2).unsqueeze(1)  # B,1,F,T
                values = self.stages(self.stem(values))
                values = self.final_features(values).mean(dim=2).transpose(1, 2)
                return self.classifier(values)

            def forward(self, values):
                return self.encode_frames(values).mean(dim=1)


        class EdgeResidualBottleneck(nn.Module):
            def __init__(self, middle_channels):
                super().__init__()
                self.layers = nn.Sequential(
                    CausalTimeLastConv2d(39, middle_channels, 3, bias=False),
                    nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True),
                    CausalTimeLastConv2d(middle_channels, 39, 3, bias=False),
                    nn.BatchNorm2d(39),
                )

            def forward(self, values):
                return F.relu(values + self.layers(values), inplace=True)


        class EdgeSpeechNetA(nn.Module):
            temporal_stride = 1
            temporal_receptive_field_frames = 29

            def __init__(self, num_classes=10):
                super().__init__()
                self.stem = nn.Sequential(
                    CausalTimeLastConv2d(1, 39, 3, bias=False),
                    nn.BatchNorm2d(39), nn.ReLU(inplace=True),
                )
                self.blocks = nn.Sequential(*[
                    EdgeResidualBottleneck(width) for width in (20, 15, 25, 22, 22, 25)
                ])
                self.final = nn.Sequential(
                    CausalTimeLastConv2d(39, 45, 3, bias=False),
                    nn.BatchNorm2d(45), nn.ReLU(inplace=True),
                )
                self.classifier = nn.Linear(45, num_classes)

            def encode_frames(self, values):
                values = values.transpose(1, 2).unsqueeze(1)  # B,1,F,T
                values = self.final(self.blocks(self.stem(values))).mean(dim=2).transpose(1, 2)
                return self.classifier(values)

            def forward(self, values):
                return self.encode_frames(values).mean(dim=1)


        COMPACT_FACTORIES = {
            "matchboxnet_3x1x64": MatchboxNet3x1x64,
            "tc_resnet8": TCResNet8,
            "mlperf_tiny_dscnn": MLPerfTinyDSCNN,
            "bc_resnet_1": lambda: BCResNet(scale=1),
            "bc_resnet_2": lambda: BCResNet(scale=2),
            "edgespeechnet_a": EdgeSpeechNetA,
        }

        dummy = torch.randn(2, COMPACT_FRAMES, N_MELS)
        changed_future = dummy.clone()
        perturbation_start = COMPACT_FRAMES // 2
        changed_future[:, perturbation_start:] += 7.0 * torch.randn_like(
            changed_future[:, perturbation_start:]
        )
        architecture_rows = []
        for name, factory in COMPACT_FACTORIES.items():
            model = factory().eval()
            assert model(dummy).shape == (2, 10), (name, model(dummy).shape)
            with torch.inference_mode():
                original_frames = model.encode_frames(dummy)
                changed_frames = model.encode_frames(changed_future)
            unaffected_outputs = math.ceil(
                perturbation_start / model.temporal_stride
            )
            future_error = float((
                original_frames[:, :unaffected_outputs]
                - changed_frames[:, :unaffected_outputs]
            ).abs().max())
            if future_error > 1e-5:
                raise RuntimeError(
                    f"{name} is not causal: future perturbation error {future_error}"
                )
            architecture_rows.append({
                "model": name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "output_shape": str(tuple(model(dummy).shape)),
                "frame_logits_shape": str(tuple(original_frames.shape)),
                "feature_hop_ms": HOP_MS,
                "temporal_stride": model.temporal_stride,
                "output_update_interval_ms": HOP_MS * model.temporal_stride,
                "temporal_receptive_field_ms": (
                    HOP_MS * model.temporal_receptive_field_frames
                ),
                "future_lookahead_frames": 0,
                "future_invariance_max_abs_error": future_error,
                "decision_state": "causal running mean of frame logits",
            })
        architecture_audit = pd.DataFrame(architecture_rows)
        architecture_audit.to_csv(OUTPUT_ROOT / "causal_architecture_audit.csv", index=False)
        display(architecture_audit)
        """
    ),
    markdown("## 7. Validation-only training and checkpointing"),
    code(
        r"""
        def classification_metrics(logits, labels):
            top1 = (logits.argmax(dim=1) == labels).float().mean().item()
            top3 = (logits.topk(3, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()
            return top1, top3


        @torch.inference_mode()
        def evaluate_loader(model, loader, device=DEVICE):
            model.eval()
            total_loss = 0.0
            total_top1 = 0.0
            total_top3 = 0.0
            total = 0
            all_predictions = []
            all_labels = []
            all_indices = []
            for features, labels, indices in loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(features)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                top1, top3 = classification_metrics(logits, labels)
                batch = len(labels)
                total_loss += loss.item()
                total_top1 += top1 * batch
                total_top3 += top3 * batch
                total += batch
                all_predictions.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_indices.append(indices.numpy())
            return {
                "loss": total_loss / total,
                "accuracy": total_top1 / total,
                "top3_accuracy": total_top3 / total,
                "prediction": np.concatenate(all_predictions),
                "label": np.concatenate(all_labels),
                "record_index": np.concatenate(all_indices),
            }


        TRAINED = {}


        def train_classifier(name, model, loaders, epochs, minimum_epochs, patience,
                             optimizer_factory, metadata):
            checkpoint = OUTPUT_ROOT / f"{name}_best.pt"
            history_path = OUTPUT_ROOT / f"{name}_history.csv"
            model = model.to(DEVICE)

            if REUSE_CHECKPOINTS and checkpoint.is_file():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                saved_contract = saved.get("metadata", {}).get(
                    "causal_contract_version"
                )
                if saved_contract == CAUSAL_CONTRACT_VERSION:
                    model.load_state_dict(saved["state_dict"])
                    TRAINED[name] = {
                        "model": model.cpu(), "checkpoint": checkpoint,
                        "metadata": saved["metadata"], "best_epoch": saved["best_epoch"],
                        "best_validation_loss": saved["best_validation_loss"],
                    }
                    print(f"{name}: loaded {checkpoint.name}")
                    return
                print(
                    f"{name}: ignoring checkpoint from contract {saved_contract}; "
                    f"current contract is {CAUSAL_CONTRACT_VERSION}"
                )

            optimizer = optimizer_factory(model)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
            )
            scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
            best_loss = math.inf
            best_epoch = 0
            stale_epochs = 0
            history = []

            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                progress = tqdm(loaders["train"], desc=f"{name} epoch {epoch}/{epochs}", leave=False)
                for features, labels, _ in progress:
                    features = features.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                        enabled=DEVICE.type == "cuda"):
                        logits = model(features)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item() * len(labels)
                    train_correct += (logits.argmax(1) == labels).sum().item()
                    train_total += len(labels)

                validation = evaluate_loader(model, loaders["validation"])
                scheduler.step(validation["loss"])
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss / train_total,
                    "train_accuracy": train_correct / train_total,
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                    "validation_top3_accuracy": validation["top3_accuracy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                history.append(row)
                print(row)

                if validation["loss"] < best_loss - 1e-5:
                    best_loss = validation["loss"]
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "metadata": metadata,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_loss,
                    }, checkpoint)
                elif epoch >= minimum_epochs:
                    stale_epochs += 1
                    if epoch >= minimum_epochs and stale_epochs >= patience:
                        print(f"{name}: early stopping at epoch {epoch}")
                        break

            pd.DataFrame(history).to_csv(history_path, index=False)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["state_dict"])
            TRAINED[name] = {
                "model": model.cpu(), "checkpoint": checkpoint,
                "metadata": metadata, "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
            }
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


        compact_metadata = {
            "causal_contract_version": CAUSAL_CONTRACT_VERSION,
            "frontend": (
                "causal 40-bin HTK log-Mel; 16 kHz; 50 ms window; "
                "10 ms hop; center=False; 1 s fixed input"
            ),
            "pretraining": "none; trained from scratch on AudioMNIST train speakers",
            "selection": "minimum validation cross-entropy",
            "decoder": "10-class softmax; top-1 and top-3",
            "streaming": "past-only encoder plus running mean of frame logits",
        }
        for name, factory in COMPACT_FACTORIES.items():
            train_classifier(
                name=name,
                model=factory(),
                loaders=compact_loaders,
                epochs=COMPACT_EPOCHS,
                minimum_epochs=COMPACT_MINIMUM_EPOCHS,
                patience=COMPACT_PATIENCE,
                optimizer_factory=lambda model: torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=COMPACT_LR, weight_decay=1e-4,
                ),
                metadata={**compact_metadata, "architecture": name},
            )
        """
    ),
    markdown(
        r"""
        ## 8. Whisper Tiny and Base adaptations

        OpenAI Whisper was built for sequence-to-sequence ASR with a 30-second
        context. For a fair isolated-digit deployment measurement, this notebook
        retains the pretrained **audio encoder**, uses only the first 50 encoder
        positions produced by one second of audio, mean-pools the encoder sequence,
        and adds a ten-class linear head. The last two encoder blocks and the head
        are fine-tuned; earlier encoder blocks are frozen. The language decoder is
        absent from parameter, size, MAC, and latency counts because it is not used.

        This adaptation prevents padding a one-second digit to 30 seconds, but it
        must be named "short-context Whisper encoder classifier," not full Whisper
        transcription.
        """
    ),
    code(
        r"""
        class ShortContextWhisperClassifier(nn.Module):
            def __init__(self, size, num_classes=10, unfreeze_last_blocks=2):
                super().__init__()
                source = whisper.load_model(size, device="cpu")
                self.encoder = source.encoder.float()
                self.whisper_size = size
                self.audio_state = source.dims.n_audio_state
                del source
                encoded_steps = math.ceil(WHISPER_FRAMES / 2)
                short_positions = self.encoder.positional_embedding[:encoded_steps].detach().clone()
                self.encoder.positional_embedding = nn.Parameter(short_positions, requires_grad=False)
                self.classifier = nn.Linear(self.audio_state, num_classes)

                for parameter in self.encoder.parameters():
                    parameter.requires_grad = False
                for block in self.encoder.blocks[-unfreeze_last_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for parameter in self.encoder.ln_post.parameters():
                    parameter.requires_grad = True

            def forward(self, values):
                values = values.transpose(1, 2)  # B,80,T
                values = F.gelu(self.encoder.conv1(values))
                values = F.gelu(self.encoder.conv2(values))
                values = values.permute(0, 2, 1)
                positions = self.encoder.positional_embedding[:values.shape[1]]
                values = (values + positions).to(values.dtype)
                for block in self.encoder.blocks:
                    values = block(values)
                values = self.encoder.ln_post(values)
                return self.classifier(values.mean(dim=1).float())


        whisper_loaders_by_size = {}
        for size in ("tiny", "base"):
            whisper_loaders = make_loaders(
                WHISPER80, WHISPER_BATCH_SIZE[size], include_test=False
            )
            whisper_loaders_by_size[size] = whisper_loaders
            name = f"whisper_{size}_short_context_encoder"
            model = ShortContextWhisperClassifier(
                size, unfreeze_last_blocks=WHISPER_LAST_BLOCKS
            )

            def whisper_optimizer(current_model):
                head_parameters = list(current_model.classifier.parameters())
                head_ids = {id(parameter) for parameter in head_parameters}
                encoder_parameters = [
                    parameter for parameter in current_model.parameters()
                    if parameter.requires_grad and id(parameter) not in head_ids
                ]
                return torch.optim.AdamW([
                    {"params": encoder_parameters, "lr": WHISPER_ENCODER_LR},
                    {"params": head_parameters, "lr": WHISPER_HEAD_LR},
                ], weight_decay=1e-4)

            train_classifier(
                name=name,
                model=model,
                loaders=whisper_loaders,
                epochs=WHISPER_EPOCHS,
                minimum_epochs=WHISPER_MINIMUM_EPOCHS,
                patience=WHISPER_PATIENCE,
                optimizer_factory=whisper_optimizer,
                metadata={
                    "causal_contract_version": CAUSAL_CONTRACT_VERSION,
                    "architecture": name,
                    "frontend": "OpenAI Whisper native 80-bin log-Mel; 16 kHz; one-second short context",
                    "pretraining": f"OpenAI Whisper {size} pretrained encoder",
                    "fine_tuning": f"last {WHISPER_LAST_BLOCKS} encoder blocks + layer norm + 10-class head",
                    "decoder": "removed; isolated-digit classifier",
                    "selection": "minimum validation cross-entropy",
                },
            )
        """
    ),
    markdown(
        """
        ## 9. One-shot held-out test

        This is the first cell that iterates over held-out test labels. Do not tune
        epochs, architecture, or hyperparameters after inspecting these results.
        Any later revision requires a new untouched test split.
        """
    ),
    code(
        r"""
        compact_test_loader = make_loaders(
            LOGMEL40, COMPACT_BATCH_SIZE, COMPACT_MEAN, COMPACT_STD, include_test=True
        )["test"]
        whisper_test_loader = make_loaders(
            WHISPER80, WHISPER_BATCH_SIZE["base"], include_test=True
        )["test"]

        TEST_RESULTS = {}
        prediction_frames = []
        for name, item in TRAINED.items():
            loader = whisper_test_loader if name.startswith("whisper_") else compact_test_loader
            model = item["model"].to(DEVICE)
            result = evaluate_loader(model, loader)
            frame = pd.DataFrame({
                "model": name,
                "record_index": result["record_index"],
                "label": result["label"],
                "prediction": result["prediction"],
            })
            frame["speaker"] = SPEAKERS[frame.record_index.to_numpy()]
            speaker_accuracy = frame.assign(
                correct=frame.label.eq(frame.prediction)
            ).groupby("speaker").correct.mean()
            result["macro_speaker_accuracy"] = float(speaker_accuracy.mean())
            TEST_RESULTS[name] = result
            prediction_frames.append(frame)
            item["model"] = model.cpu()
            print(name, {
                "top1": result["accuracy"],
                "top3": result["top3_accuracy"],
                "macro_speaker": result["macro_speaker_accuracy"],
            })
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(OUTPUT_ROOT / "heldout_test_predictions.csv", index=False)
        """
    ),
    markdown(
        r"""
        ## 10. Parameters, FP32 size, MACs, and latency

        Learned MACs are counted from the operations actually executed by
        `Conv1d`, `Conv2d`, and `Linear` modules. Whisper additionally includes the
        two attention matrix products, `QK^T` and `softmax(QK^T)V`, for every audio
        encoder block. Batch normalization, activations, pooling, FFT, and Mel
        filtering are not learned-model MACs. The notebook therefore reports an
        analytical STFT/Mel front-end operation estimate separately, alongside
        measured front-end and end-to-end latency.

        Latency is measured at batch size one on the Colab CPU with one PyTorch
        thread. File loading is excluded. These numbers are hardware-specific and
        should always be reported with the generated `hardware.json` file.
        """
    ),
    code(
        r"""
        def count_learned_macs(model, sample):
            total = 0
            handles = []

            def convolution_hook(module, inputs, output):
                nonlocal total
                kernel = int(np.prod(module.kernel_size))
                total += int(output.numel() * kernel * module.in_channels / module.groups)

            def linear_hook(module, inputs, output):
                nonlocal total
                total += int(output.numel() * module.in_features)

            for module in model.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    handles.append(module.register_forward_hook(convolution_hook))
                elif isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(linear_hook))

            model.eval()
            with torch.inference_mode():
                model(sample)
            for handle in handles:
                handle.remove()

            if isinstance(model, ShortContextWhisperClassifier):
                encoded_steps = math.ceil(sample.shape[1] / 2)
                attention_products = (
                    2 * len(model.encoder.blocks) * encoded_steps * encoded_steps * model.audio_state
                )
                total += int(attention_products)
            return int(total)


        def estimate_logmel_frontend_cost(n_fft, n_mels, frames):
            # Dense STFT/Mel arithmetic estimate; nonlinear calls stay separate.
            spectrum_bins = n_fft // 2 + 1
            window_ops = frames * n_fft
            fft_real_ops = frames * 5.0 * n_fft * math.log2(n_fft)
            power_ops = frames * 3.0 * spectrum_bins
            dense_mel_ops = frames * n_mels * (2.0 * spectrum_bins - 1.0)
            return {
                "frontend_estimated_arithmetic_ops": float(
                    window_ops + fft_real_ops + power_ops + dense_mel_ops
                ),
                "frontend_estimated_fft_real_ops": float(fft_real_ops),
                "frontend_estimated_dense_mel_macs": float(
                    frames * n_mels * spectrum_bins
                ),
                "frontend_log_calls": int(frames * n_mels),
                "frontend_cost_convention": (
                    "Hann multiply + 5*N*log2(N) real FFT approximation + "
                    "complex power + dense Mel multiply/add; log calls separate"
                ),
            }


        def fp32_state_size_mib(model):
            # Deployment-size estimate if every state tensor is stored as float32.
            return sum(value.numel() * 4 for value in model.state_dict().values()) / 2**20


        def latency_ms(function, warmup=LATENCY_WARMUP, repeats=LATENCY_REPEATS):
            for _ in range(warmup):
                function()
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                function()
                timings.append((time.perf_counter() - started) * 1000.0)
            return float(np.median(timings)), float(np.percentile(timings, 95))


        benchmark_index = int(INDICES["test"][len(INDICES["test"]) // 2])
        benchmark_wave = torch.from_numpy(
            np.asarray(WAVEFORMS[benchmark_index], dtype=np.float32)
        ).unsqueeze(0)
        compact_mean_tensor = torch.from_numpy(COMPACT_MEAN)[None, None, :]
        compact_std_tensor = torch.from_numpy(COMPACT_STD)[None, None, :]
        compact_mel_cpu = make_compact_mel(torch.device("cpu"))

        def compact_frontend():
            feature = torch.log(compact_mel_cpu(benchmark_wave).clamp_min(1e-8)).transpose(1, 2)
            return (feature - compact_mean_tensor) / compact_std_tensor

        def whisper_frontend():
            return whisper_logmel_batch(benchmark_wave).transpose(1, 2)

        compact_sample = compact_frontend()
        whisper_sample = whisper_frontend()
        assert compact_sample.shape == (1, COMPACT_FRAMES, N_MELS)
        assert whisper_sample.shape == (1, WHISPER_FRAMES, WHISPER_MELS)

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        benchmark_rows = []
        architecture_lookup = architecture_audit.set_index("model").to_dict("index")
        for name, item in TRAINED.items():
            model = copy.deepcopy(item["model"]).cpu().eval()
            is_whisper = name.startswith("whisper_")
            sample = whisper_sample if is_whisper else compact_sample
            frontend = whisper_frontend if is_whisper else compact_frontend
            with torch.inference_mode():
                frontend_median, frontend_p95 = latency_ms(frontend)
                model_median, model_p95 = latency_ms(lambda: model(sample))
                e2e_median, e2e_p95 = latency_ms(lambda: model(frontend()))
            total_parameters = sum(parameter.numel() for parameter in model.parameters())
            trainable_parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            frontend_cost = estimate_logmel_frontend_cost(
                WHISPER_N_FFT if is_whisper else N_FFT,
                WHISPER_MELS if is_whisper else N_MELS,
                WHISPER_FRAMES if is_whisper else COMPACT_FRAMES,
            )
            compact_architecture = None if is_whisper else architecture_lookup[name]
            benchmark_rows.append({
                "model": name,
                "input": "80-bin Whisper log-Mel" if is_whisper else "40-bin log-Mel",
                "raw_audio_sample_rate_hz": SAMPLE_RATE,
                "model_feature_rate_hz": SAMPLE_RATE / HOP_LENGTH,
                "feature_values_per_second": (
                    WHISPER_FRAMES * WHISPER_MELS
                    if is_whisper else COMPACT_FRAMES * N_MELS
                ),
                "feature_window_ms": 25.0 if is_whisper else WINDOW_MS,
                "feature_hop_ms": HOP_MS,
                "feature_availability_delay_ms": 25.0 if is_whisper else WINDOW_MS,
                "encoder_output_update_interval_ms": (
                    None if is_whisper
                    else compact_architecture["output_update_interval_ms"]
                ),
                "temporal_receptive_field_ms": (
                    None if is_whisper
                    else compact_architecture["temporal_receptive_field_ms"]
                ),
                "classification_endpoint_ms": 1000.0,
                "pretraining": item["metadata"]["pretraining"],
                "causal_streaming": not is_whisper,
                "future_lookahead_frames": None if is_whisper else 0,
                "decoder": "10-class softmax; top-1/top-3",
                "test_accuracy": TEST_RESULTS[name]["accuracy"],
                "test_top3_accuracy": TEST_RESULTS[name]["top3_accuracy"],
                "macro_speaker_accuracy": TEST_RESULTS[name]["macro_speaker_accuracy"],
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "fp32_state_size_mib": fp32_state_size_mib(model),
                "learned_macs": count_learned_macs(model, sample),
                "frontend_latency_ms_median": frontend_median,
                "frontend_latency_ms_p95": frontend_p95,
                "model_latency_ms_median": model_median,
                "model_latency_ms_p95": model_p95,
                "end_to_end_latency_ms_median": e2e_median,
                "end_to_end_latency_ms_p95": e2e_p95,
                "best_epoch": item["best_epoch"],
                "best_validation_loss": item["best_validation_loss"],
                **frontend_cost,
            })
        torch.set_num_threads(original_threads)

        results = pd.DataFrame(benchmark_rows)
        results["test_accuracy_percent"] = 100 * results.test_accuracy
        results["test_top3_accuracy_percent"] = 100 * results.test_top3_accuracy
        results["macro_speaker_accuracy_percent"] = 100 * results.macro_speaker_accuracy
        results["parameters_million"] = results.total_parameters / 1e6
        results["macs_million"] = results.learned_macs / 1e6
        results["latency_batch_size"] = 1
        results["latency_device"] = "Colab CPU; one PyTorch thread"
        results.to_csv(OUTPUT_ROOT / "audiomnist_model_benchmark.csv", index=False)

        hardware = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "training_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "latency_device": "CPU",
            "latency_torch_threads": 1,
            "latency_batch_size": 1,
            "latency_file_io_included": False,
        }
        (OUTPUT_ROOT / "hardware.json").write_text(
            json.dumps(hardware, indent=2) + "\n", encoding="utf-8"
        )
        display(results.sort_values("test_accuracy", ascending=False)[[
            "model", "pretraining", "test_accuracy_percent", "test_top3_accuracy_percent",
            "total_parameters", "trainable_parameters", "fp32_state_size_mib",
            "macs_million", "frontend_estimated_arithmetic_ops",
            "frontend_latency_ms_median", "model_latency_ms_median",
            "end_to_end_latency_ms_median",
        ]].style.format({
            "test_accuracy_percent": "{:.2f}",
            "test_top3_accuracy_percent": "{:.2f}",
            "fp32_state_size_mib": "{:.3f}",
            "macs_million": "{:.3f}",
            "frontend_estimated_arithmetic_ops": "{:.0f}",
            "frontend_latency_ms_median": "{:.3f}",
            "model_latency_ms_median": "{:.3f}",
            "end_to_end_latency_ms_median": "{:.3f}",
        }))
        """
    ),
    markdown("## 11. Fabricated-resonator reference and figures"),
    code(
        r"""
        # Frozen held-out result from the physical five-resonator AudioMNIST experiment.
        # It uses the exact same 36/12/12 speaker split, but acoustic playback through
        # the device rather than direct digital WAV input. Its latency was measured on
        # a different local CPU and must not be treated as same-hardware latency.
        fabricated_macs_per_second = 3_113_200 / 0.760
        resonator_reference = pd.DataFrame([{
            "model": "fabricated_5d_causal_conv_unigru24",
            "input": "5 physical resonator RMS-envelope values / 10 ms",
            "pretraining": "none",
            "test_accuracy_percent": 93.6167,
            "test_top3_accuracy_percent": np.nan,
            "total_parameters": 41_514,
            "trainable_parameters": 41_514,
            "fp32_state_size_mib": 41_514 * 4 / 2**20,
            "checkpoint_size_mib": 0.1633,
            "learned_macs": fabricated_macs_per_second,
            "macs_million": fabricated_macs_per_second / 1e6,
            "raw_audio_sample_rate_hz": 48_000,
            "raw_channel_count": 5,
            "intermediate_envelope_rate_hz": 400,
            "model_feature_rate_hz": 100,
            "feature_values_per_second": 500,
            "feature_window_ms": 10.0,
            "feature_hop_ms": 10.0,
            "feature_availability_delay_ms": 17.5,
            "encoder_output_update_interval_ms": 10.0,
            "temporal_receptive_field_ms": np.nan,
            "temporal_context": "unbounded causal past summarized by GRU state",
            "classification_endpoint_ms": 760.0,
            "causal_streaming": True,
            "future_lookahead_frames": 0,
            "decoder": "10-class softmax; top-1/top-3",
            "passive_resonator_frontend_learned_macs": 0,
            "frontend_estimated_arithmetic_ops": 738_000,
            "frontend_cost_convention": (
                "approximate five-channel cumulative-sum AC-RMS at 400 Hz, "
                "quiet-baseline normalization, dB conversion, and 4-to-1 mean"
            ),
            "model_latency_ms_median": 2.2152,
            "model_latency_ms_p95": 2.8363,
            "latency_note": "prior local CPU, one thread; not directly comparable to Colab latency",
            "comparison_scope": (
                "same speaker-disjoint source/test split; physical acoustic capture "
                "rather than direct digital waveform"
            ),
        }])
        display(resonator_reference)
        resonator_reference.to_csv(OUTPUT_ROOT / "fabricated_resonator_reference.csv", index=False)

        common_columns = [
            "model", "input", "pretraining", "test_accuracy_percent",
            "test_top3_accuracy_percent", "total_parameters", "trainable_parameters",
            "fp32_state_size_mib", "learned_macs", "macs_million",
            "raw_audio_sample_rate_hz", "model_feature_rate_hz",
            "feature_values_per_second", "feature_window_ms", "feature_hop_ms",
            "feature_availability_delay_ms", "encoder_output_update_interval_ms",
            "temporal_receptive_field_ms", "classification_endpoint_ms",
            "causal_streaming", "future_lookahead_frames", "decoder",
            "frontend_estimated_arithmetic_ops", "model_latency_ms_median",
            "model_latency_ms_p95",
        ]
        full_comparison = pd.concat(
            [results.reindex(columns=common_columns),
             resonator_reference.reindex(columns=common_columns)],
            ignore_index=True,
        )
        full_comparison.to_csv(
            OUTPUT_ROOT / "audiomnist_digital_and_fabricated_comparison.csv",
            index=False,
        )
        display(full_comparison.sort_values("test_accuracy_percent", ascending=False))

        sns.set_theme(style="whitegrid")
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        ordered = full_comparison.sort_values("test_accuracy_percent")
        axes[0].barh(ordered.model, ordered.test_accuracy_percent, color="#277da1")
        axes[0].set_xlabel("Held-out top-1 accuracy (%)")
        axes[0].set_xlim(max(0, ordered.test_accuracy_percent.min() - 5), 100)

        axes[1].scatter(full_comparison.macs_million, full_comparison.test_accuracy_percent,
                        s=70, color="#2a9d8f")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Learned MACs / one-second utterance (million, log scale)")
        axes[1].set_ylabel("Held-out top-1 accuracy (%)")
        for row in full_comparison.itertuples():
            axes[1].annotate(row.model, (row.macs_million, row.test_accuracy_percent),
                             xytext=(4, 3), textcoords="offset points", fontsize=8)

        axes[2].scatter(full_comparison.total_parameters,
                        full_comparison.test_accuracy_percent,
                        s=70, color="#e76f51")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("Total parameters (log scale)")
        axes[2].set_ylabel("Held-out top-1 accuracy (%)")
        for row in full_comparison.itertuples():
            axes[2].annotate(row.model, (row.total_parameters, row.test_accuracy_percent),
                             xytext=(4, 3), textcoords="offset points", fontsize=8)
        figure.suptitle("AudioMNIST accuracy-efficiency comparison")
        figure.tight_layout()
        figure.savefig(OUTPUT_ROOT / "audiomnist_accuracy_efficiency.png", dpi=220, bbox_inches="tight")
        plt.show()
        """
    ),
    code(
        r"""
        model_names = list(TRAINED)
        figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        for axis, name in zip(axes.flat, model_names):
            frame = predictions[predictions.model == name]
            matrix = confusion_matrix(frame.label, frame.prediction, labels=np.arange(10), normalize="true")
            sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", square=True,
                        cbar=False, ax=axis)
            axis.set_title(name)
            axis.set_xlabel("Predicted digit")
            axis.set_ylabel("True digit")
        figure.suptitle("Held-out speaker-normalized confusion matrices", fontsize=16)
        figure.savefig(OUTPUT_ROOT / "audiomnist_confusion_matrices.png", dpi=220, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 12. Reporting checklist

        Report all of the following with the results:

        - The fixed 36/12/12 speaker split and 18k/6k/6k utterance counts.
        - Compact front end: 16 kHz, 40 Mel bins, 50 ms window, 10 ms hop,
          `center=False`, one-second fixed input, and train-only normalization.
        - Compact encoders use past-only temporal padding and zero future
          lookahead; the classification state is a running mean of frame logits.
        - Whisper front end: native 80-bin log-Mel and one-second short-context
          encoder adaptation.
        - Which models were trained from scratch and which used external Whisper
          pretraining.
        - Validation loss selected checkpoints; held-out test was evaluated once.
        - AudioMNIST uses a softmax classifier, not CTC beam decoding, because each
          evaluated example has exactly one known digit label.
        - Learned MACs and front-end operation estimates are separate; feature
          extraction is included in measured end-to-end latency.
        - Exact CPU/GPU information from `hardware.json`.
        - The compact architectures are AudioMNIST adaptations, not official MLPerf
          or original-paper benchmark results.

        The digital table is saved as `audiomnist_model_benchmark.csv`; the merged
        digital/physical table is `audiomnist_digital_and_fabricated_comparison.csv`.
        Per-utterance predictions, histories, checkpoints, figures, the experiment
        contract, causal audit, and hardware metadata are saved beside them.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
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


OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
