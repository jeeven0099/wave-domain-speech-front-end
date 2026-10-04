#!/usr/bin/env python3
"""Create the AudioMNIST digital-baseline Colab notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audiomnist_digital_baselines_colab.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells: list[dict[str, object]] = [
    markdown(
        r"""
        # AudioMNIST digital baselines on the physical-device split

        This notebook trains and evaluates ten clean-digital baselines using the
        **same published AudioMNIST digit split 1** used by the fabricated
        five-resonator experiment:

        1. matched five-band, 20-dimensional sequence + causal Conv-GRU;
        2. 20-bin log-Mel sequence + the same causal Conv-GRU (front-end control);
        3. 40-bin log-Mel + BC-ResNet-1 (small-digital control);
        4. 40-bin log-Mel + BC-ResNet-8 (primary strong digital encoder);
        5. 120-bin log-Mel + BC-ResNet-8 (spectral-resolution ablation);
        6. 40-bin log-Mel + Transformer-CTC;
        7. 120-bin log-Mel + Transformer-CTC;
        8. temporally pooled 20-bin log-Mel + RBF SVM;
        9. AudioNet architecture with its official 8 kHz waveform preprocessing;
        10. AlexNet architecture with its official 227 x 227 STFT preprocessing.

        The final table reports held-out accuracy, trainable parameters, serialized
        size, MACs, feature-extraction latency, model latency, and end-to-end
        latency. The notebook keeps the test split out of training, normalization,
        early stopping, and hyperparameter selection.

        **Interpretation:** the first two Conv-GRU rows isolate the front end because
        they use the same classifier. BC-ResNet is the primary task-appropriate
        log-Mel encoder; the scale-1 and scale-8 rows prevent a cherry-picked model
        size comparison. AudioNet and AlexNet are PyTorch
        reimplementations of the official Caffe architecture files and are external
        architecture references, not latency- or feature-matched models. Clean
        source audio is an ideal digital ceiling; it is not a hardware-matched open
        microphone control.
        """
    ),
    code(
        r"""
        %pip -q install thop
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import copy
        import hashlib
        import json
        import math
        import os
        import platform
        import random
        import subprocess
        import time
        import zlib
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import torch.nn.functional as F
        from scipy import fft, signal
        from scipy.io import wavfile
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from thop import profile
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        SEED = 20260905
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training device:", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    markdown(
        """
        ## Configuration

        Select a GPU runtime in Colab. Outputs are written to Google Drive by
        default; the feature cache stays on Colab's local disk for speed.

        The five-band values below must be replaced by final measured peak and
        -3 dB crossing frequencies before the row is described as *matched* in a
        paper. Until `BANDS_ARE_FROM_FINAL_SWEEP=True`, the result table calls it a
        provisional target-band baseline.
        """
    ),
    code(
        r"""
        IN_COLAB = "google.colab" in str(get_ipython())
        USE_GOOGLE_DRIVE = True

        if IN_COLAB and USE_GOOGLE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")

        DATASET_ROOT = Path("/content/AudioMNIST") if IN_COLAB else Path("datasets/AudioMNIST")
        CACHE_ROOT = Path("/content/audiomnist_digital_cache") if IN_COLAB else Path(".cache/audiomnist_digital")
        if IN_COLAB and USE_GOOGLE_DRIVE:
            OUTPUT_ROOT = Path("/content/drive/MyDrive/audiomnist_digital_baselines")
        else:
            OUTPUT_ROOT = Path("audiomnist_digital_baselines_outputs")

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        # These are the same values used for the physical Conv-GRU unless noted.
        CONTEXT_MS = 50.0
        MAX_SEQUENCE_STEPS = 120
        TRAIN_BATCH_SIZE = 64
        BCRESNET_BATCH_SIZE = 32
        LEGACY_BATCH_SIZE = 64
        ALEXNET_BATCH_SIZE = 32
        MAX_EPOCHS = 30
        LEGACY_MAX_EPOCHS = 25
        EARLY_STOPPING_PATIENCE = 6
        LEARNING_RATE = 1e-3
        ALEXNET_LEARNING_RATE = 3e-4
        BCRESNET_MAX_EPOCHS = 50
        BCRESNET_LEARNING_RATE = 1e-3
        TRANSFORMER_CTC_MAX_EPOCHS = 50
        TRANSFORMER_CTC_LEARNING_RATE = 3e-4
        TRANSFORMER_CTC_D_MODEL = 128
        TRANSFORMER_CTC_HEADS = 4
        TRANSFORMER_CTC_LAYERS = 4
        TRANSFORMER_CTC_FF_DIM = 256
        TRANSFORMER_CTC_DROPOUT = 0.1
        NUM_WORKERS = 2 if IN_COLAB else 0
        LATENCY_WARMUP = 10
        LATENCY_REPEATS = 50

        DIGITAL_SAMPLE_RATE = 48_000
        LOGMEL_SAMPLE_RATE = 16_000
        BCRESNET_N_MELS = 40
        BCRESNET_HIGH_RES_N_MELS = 120
        BCRESNET_WINDOW_MS = 30.0
        BCRESNET_HOP_MS = 10.0
        BCRESNET_FIXED_SECONDS = 1.0
        BCRESNET_SEQUENCE_STEPS = 101
        LEGACY_SAMPLE_RATE = 8_000
        TARGET_PEAK_DBFS = -9.0

        # Measured apparent response from the final five-channel sweep on 2026-09-03.
        # Values are: channel label, dominant peak, lower -3 dB crossing, upper crossing.
        BAND_ROWS = [
            ("R300", 7190.177566, 7114.624859, 7260.040358),
            ("R3500", 3291.411479, 3179.548891, 3461.522389),
            ("R7500", 7015.432600, 6977.949541, 7277.872674),
            ("R1800", 1626.905551, 1587.428088, 1668.553394),
            ("R4500", 4392.087728, 4197.867995, 4427.693701),
        ]
        BANDS_ARE_FROM_FINAL_SWEEP = True
        BAND_CALIBRATION_SOURCE = (
            "2026-09-03 10-28 AM/Analog - 3-9-2026 10-28-36.37038 AM_fullband_fft/"
            "dominant_resonance_summary.csv"
        )
        BAND_CALIBRATION_LIMITATION = (
            "Apparent channel response: no independent reference microphone was used to divide out "
            "the phone-speaker, propagation, and room transfer functions."
        )

        bands = pd.DataFrame(BAND_ROWS, columns=["name", "center_hz", "low_3db_hz", "high_3db_hz"])
        bands["bandwidth_hz"] = bands.high_3db_hz - bands.low_3db_hz
        bands["Q"] = bands.center_hz / bands.bandwidth_hz
        assert (bands.low_3db_hz > 0).all()
        assert (bands.high_3db_hz < DIGITAL_SAMPLE_RATE / 2).all()
        display(bands)

        if not BANDS_ARE_FROM_FINAL_SWEEP:
            print("WARNING: five-band settings are provisional. Replace them with final measured values for a paper.")
        else:
            print("Band source:", BAND_CALIBRATION_SOURCE)
            print("Limitation:", BAND_CALIBRATION_LIMITATION)
        """
    ),
    markdown(
        """
        ## Dataset and exact speaker split

        The repository is downloaded only when it is not already present. The
        assertions stop the experiment if the expected 30,000 WAV files or the
        18,000/6,000/6,000 split counts are not recovered.
        """
    ),
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
            speaker: split for split, speakers in SPLIT_SPEAKERS.items() for speaker in speakers
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
        """
    ),
    markdown(
        """
        ## Preprocessing and cache

        - Five-band: five causal second-order peak filters, 10 ms RMS windows at a
          2.5 ms hop, then four ordered subframes per 10 ms model step. This gives
          the same 20-dimensional timing layout as the physical classifier.
        - Log-Mel: 20 HTK Mel filters, exact 10 ms nonoverlapping frames, 100 model
          steps/s.
        - Legacy waveform: the official 8 kHz, one-second embedding and 95th
          percentile normalization. A stable seeded offset reproduces the
          repository's one-time random placement without changing across epochs.
        """
    ),
    code(
        r"""
        def read_mono(path: str | Path) -> tuple[int, np.ndarray]:
            sample_rate, values = wavfile.read(path)
            if values.ndim == 2:
                values = values.astype(np.float32).mean(axis=1)
            if np.issubdtype(values.dtype, np.integer):
                info = np.iinfo(values.dtype)
                values = values.astype(np.float32) / float(max(abs(info.min), info.max))
            else:
                values = values.astype(np.float32)
            return int(sample_rate), values


        def resample_audio(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
            if source_rate == target_rate:
                return values.astype(np.float32, copy=False)
            divisor = math.gcd(source_rate, target_rate)
            return signal.resample_poly(
                values, target_rate // divisor, source_rate // divisor
            ).astype(np.float32)


        def peak_normalize(values: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
            peak = float(np.max(np.abs(values)))
            if peak < 1e-9:
                return values.astype(np.float32)
            return (values * (10.0 ** (target_dbfs / 20.0) / peak)).astype(np.float32)


        def frame_rms(values: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
            if len(values) < frame_samples:
                values = np.pad(values, (0, frame_samples - len(values)))
            frame_count = 1 + (len(values) - frame_samples) // hop_samples
            windows = np.lib.stride_tricks.sliding_window_view(values, frame_samples)
            frames = windows[::hop_samples][:frame_count]
            return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-16).astype(np.float32)


        def matched_five_band_features(values_48k: np.ndarray) -> np.ndarray:
            context = int(round(CONTEXT_MS * DIGITAL_SAMPLE_RATE / 1000.0))
            values = np.pad(peak_normalize(values_48k), (context, context))
            frame = int(round(0.010 * DIGITAL_SAMPLE_RATE))
            hop = int(round(0.0025 * DIGITAL_SAMPLE_RATE))
            columns = []
            for row in bands.itertuples(index=False):
                quality = float(row.center_hz / row.bandwidth_hz)
                b, a = signal.iirpeak(float(row.center_hz), quality, fs=DIGITAL_SAMPLE_RATE)
                filtered = signal.lfilter(b, a, values).astype(np.float32)
                columns.append(frame_rms(filtered, frame, hop))
            envelope = np.column_stack(columns)
            envelope_db = 20.0 * np.log10(np.maximum(envelope, 1e-8))
            envelope_db = np.clip(envelope_db, -120.0, 20.0)
            usable = len(envelope_db) - len(envelope_db) % 4
            return (
                envelope_db[:usable]
                .reshape(-1, 4, 5)
                .transpose(0, 2, 1)
                .reshape(-1, 20)
                .astype(np.float32)
            )


        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


        def make_mel_filter(
            n_mels: int = 20,
            n_fft: int = 512,
            f_min: float = 20.0,
            f_max: float = 8000.0,
            area_normalize: bool = True,
        ) -> np.ndarray:
            edges = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
            frequencies = np.fft.rfftfreq(n_fft, d=1.0 / LOGMEL_SAMPLE_RATE)
            filters = np.zeros((n_mels, len(frequencies)), dtype=np.float64)
            for index in range(n_mels):
                left, center, right = edges[index:index + 3]
                rising = (frequencies - left) / max(center - left, 1e-12)
                falling = (right - frequencies) / max(right - center, 1e-12)
                filters[index] = np.maximum(0.0, np.minimum(rising, falling))
                if area_normalize:
                    filters[index] *= 2.0 / max(right - left, 1e-12)
            return filters.astype(np.float32)


        MEL_FILTER_20 = make_mel_filter()
        MEL_FILTER_40_BCRESNET = make_mel_filter(
            n_mels=BCRESNET_N_MELS, f_min=0.0, f_max=LOGMEL_SAMPLE_RATE / 2,
            area_normalize=False,
        )
        MEL_FILTER_120_BCRESNET = make_mel_filter(
            n_mels=BCRESNET_HIGH_RES_N_MELS, f_min=0.0, f_max=LOGMEL_SAMPLE_RATE / 2,
            area_normalize=False,
        )


        def logmel_20_features(values_16k: np.ndarray) -> np.ndarray:
            context = int(round(CONTEXT_MS * LOGMEL_SAMPLE_RATE / 1000.0))
            values = np.pad(peak_normalize(values_16k), (context, context))
            frame_samples = int(round(0.010 * LOGMEL_SAMPLE_RATE))
            frame_count = len(values) // frame_samples
            frames = values[:frame_count * frame_samples].reshape(frame_count, frame_samples)
            frames = frames - frames.mean(axis=1, keepdims=True)
            window = np.hanning(frame_samples).astype(np.float32)
            spectrum = fft.rfft(frames * window, n=512, axis=1, workers=1)
            power = np.abs(spectrum) ** 2 / max(float(np.sum(window ** 2)), 1e-12)
            mel_power = power @ MEL_FILTER_20.T
            return np.clip(10.0 * np.log10(np.maximum(mel_power, 1e-12)), -120.0, 40.0).astype(np.float32)


        def logmel_bcresnet_features(
            values_16k: np.ndarray, mel_filter: np.ndarray, n_mels: int
        ) -> np.ndarray:
            # BC-ResNet temporal geometry: 30 ms window, 10 ms hop.
            target_samples = int(round(BCRESNET_FIXED_SECONDS * LOGMEL_SAMPLE_RATE))
            values = peak_normalize(values_16k)
            if len(values) > target_samples:
                raise ValueError(
                    f"BC-ResNet preprocessing expects <= {target_samples} samples, got {len(values)}"
                )
            values = np.pad(values, (0, target_samples - len(values)))

            n_fft = 512
            win_samples = int(round(BCRESNET_WINDOW_MS * LOGMEL_SAMPLE_RATE / 1000.0))
            hop_samples = int(round(BCRESNET_HOP_MS * LOGMEL_SAMPLE_RATE / 1000.0))
            padded = np.pad(values, (n_fft // 2, n_fft // 2), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_samples]
            frames = frames[:BCRESNET_SEQUENCE_STEPS]
            periodic_hann = np.hanning(win_samples + 1)[:-1].astype(np.float32)
            window = np.pad(
                periodic_hann,
                ((n_fft - win_samples) // 2, n_fft - win_samples - (n_fft - win_samples) // 2),
            )
            spectrum = fft.rfft(frames * window, n=n_fft, axis=1, workers=1)
            power = np.abs(spectrum) ** 2
            mel_power = power @ mel_filter.T
            features = np.log(np.maximum(mel_power, 1e-6)).astype(np.float32)
            if features.shape != (BCRESNET_SEQUENCE_STEPS, n_mels):
                raise RuntimeError(f"Unexpected BC-ResNet feature shape: {features.shape}")
            return features


        def logmel_40_bcresnet_features(values_16k: np.ndarray) -> np.ndarray:
            return logmel_bcresnet_features(
                values_16k, MEL_FILTER_40_BCRESNET, BCRESNET_N_MELS
            )


        def logmel_120_bcresnet_features(values_16k: np.ndarray) -> np.ndarray:
            return logmel_bcresnet_features(
                values_16k, MEL_FILTER_120_BCRESNET, BCRESNET_HIGH_RES_N_MELS
            )


        def legacy_waveform(values: np.ndarray, source_rate: int, key: str) -> np.ndarray:
            values = resample_audio(values, source_rate, LEGACY_SAMPLE_RATE)
            if len(values) > LEGACY_SAMPLE_RATE:
                raise ValueError(f"{key}: official AudioMNIST preprocessing requires <= 8000 samples")
            embedded = np.zeros(LEGACY_SAMPLE_RATE, dtype=np.float32)
            room = LEGACY_SAMPLE_RATE - len(values)
            rng = np.random.default_rng(SEED + zlib.crc32(key.encode("utf-8")))
            offset = int(rng.integers(0, room + 1)) if room else 0
            embedded[offset:offset + len(values)] = values
            scale = float(np.percentile(embedded, 95))
            return (embedded / (scale + 1e-6)).astype(np.float32)
        """
    ),
    code(
        r"""
        cache_contract = {
            "bands": BAND_ROWS,
            "band_calibration_source": BAND_CALIBRATION_SOURCE,
            "context_ms": CONTEXT_MS,
            "digital_sample_rate": DIGITAL_SAMPLE_RATE,
            "logmel_sample_rate": LOGMEL_SAMPLE_RATE,
            "bcresnet_logmel": {
                "n_mels": BCRESNET_N_MELS,
                "high_resolution_n_mels": BCRESNET_HIGH_RES_N_MELS,
                "window_ms": BCRESNET_WINDOW_MS,
                "hop_ms": BCRESNET_HOP_MS,
                "fixed_seconds": BCRESNET_FIXED_SECONDS,
            },
            "legacy_sample_rate": LEGACY_SAMPLE_RATE,
            "target_peak_dbfs": TARGET_PEAK_DBFS,
            "seed": SEED,
            "record_count": len(records),
        }
        cache_hash = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()[:12]
        CACHE_DIR = CACHE_ROOT / cache_hash
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        COMPLETE_FLAG = CACHE_DIR / "complete.json"


        def prepare_feature_cache() -> None:
            if COMPLETE_FLAG.is_file():
                print("Using complete cache:", CACHE_DIR)
                return

            count = len(records)
            band_map = np.lib.format.open_memmap(
                CACHE_DIR / "five_band.npy", mode="w+", dtype=np.float16,
                shape=(count, MAX_SEQUENCE_STEPS, 20),
            )
            mel_map = np.lib.format.open_memmap(
                CACHE_DIR / "logmel20.npy", mode="w+", dtype=np.float16,
                shape=(count, MAX_SEQUENCE_STEPS, 20),
            )
            bcresnet_mel_map = np.lib.format.open_memmap(
                CACHE_DIR / "logmel40_bcresnet.npy", mode="w+", dtype=np.float16,
                shape=(count, BCRESNET_SEQUENCE_STEPS, BCRESNET_N_MELS),
            )
            bcresnet_mel120_map = np.lib.format.open_memmap(
                CACHE_DIR / "logmel120_bcresnet.npy", mode="w+", dtype=np.float16,
                shape=(count, BCRESNET_SEQUENCE_STEPS, BCRESNET_HIGH_RES_N_MELS),
            )
            legacy_map = np.lib.format.open_memmap(
                CACHE_DIR / "legacy_waveform.npy", mode="w+", dtype=np.float16,
                shape=(count, LEGACY_SAMPLE_RATE),
            )
            band_lengths = np.zeros(count, dtype=np.int16)
            mel_lengths = np.zeros(count, dtype=np.int16)
            bcresnet_mel_lengths = np.full(count, BCRESNET_SEQUENCE_STEPS, dtype=np.int16)

            for row in tqdm(records.itertuples(index=False), total=count, desc="Preprocessing WAV files"):
                sample_rate, raw = read_mono(row.path)
                values_48k = resample_audio(raw, sample_rate, DIGITAL_SAMPLE_RATE)
                values_16k = resample_audio(raw, sample_rate, LOGMEL_SAMPLE_RATE)
                band_values = matched_five_band_features(values_48k)
                mel_values = logmel_20_features(values_16k)
                bcresnet_mel_values = logmel_40_bcresnet_features(values_16k)
                bcresnet_mel120_values = logmel_120_bcresnet_features(values_16k)
                if len(band_values) > MAX_SEQUENCE_STEPS or len(mel_values) > MAX_SEQUENCE_STEPS:
                    raise RuntimeError(
                        f"Increase MAX_SEQUENCE_STEPS for {row.path}: "
                        f"band={len(band_values)}, mel={len(mel_values)}"
                    )
                index = int(row.record_index)
                band_map[index, :len(band_values)] = band_values.astype(np.float16)
                mel_map[index, :len(mel_values)] = mel_values.astype(np.float16)
                bcresnet_mel_map[index] = bcresnet_mel_values.astype(np.float16)
                bcresnet_mel120_map[index] = bcresnet_mel120_values.astype(np.float16)
                legacy_map[index] = legacy_waveform(raw, sample_rate, Path(row.path).name).astype(np.float16)
                band_lengths[index] = len(band_values)
                mel_lengths[index] = len(mel_values)

            band_map.flush()
            mel_map.flush()
            bcresnet_mel_map.flush()
            bcresnet_mel120_map.flush()
            legacy_map.flush()
            del band_map, mel_map, bcresnet_mel_map, bcresnet_mel120_map, legacy_map
            np.save(CACHE_DIR / "five_band_lengths.npy", band_lengths)
            np.save(CACHE_DIR / "logmel20_lengths.npy", mel_lengths)
            np.save(CACHE_DIR / "logmel40_bcresnet_lengths.npy", bcresnet_mel_lengths)
            records.to_csv(CACHE_DIR / "inventory.csv", index=False)
            COMPLETE_FLAG.write_text(json.dumps(cache_contract, indent=2) + "\n")
            print("Created cache:", CACHE_DIR)


        prepare_feature_cache()

        FIVE_BAND = np.load(CACHE_DIR / "five_band.npy", mmap_mode="r")
        FIVE_BAND_LENGTHS = np.load(CACHE_DIR / "five_band_lengths.npy")
        LOGMEL20 = np.load(CACHE_DIR / "logmel20.npy", mmap_mode="r")
        LOGMEL20_LENGTHS = np.load(CACHE_DIR / "logmel20_lengths.npy")
        LOGMEL40_BCRESNET = np.load(CACHE_DIR / "logmel40_bcresnet.npy", mmap_mode="r")
        LOGMEL40_BCRESNET_LENGTHS = np.load(CACHE_DIR / "logmel40_bcresnet_lengths.npy")
        LOGMEL120_BCRESNET = np.load(CACHE_DIR / "logmel120_bcresnet.npy", mmap_mode="r")
        LEGACY_WAVEFORMS = np.load(CACHE_DIR / "legacy_waveform.npy", mmap_mode="r")
        LABELS = records.digit.to_numpy(np.int64)
        SPEAKERS = records.speaker.to_numpy(np.int64)
        SPLITS = records.split.to_numpy()
        INDICES = {name: np.flatnonzero(SPLITS == name) for name in ("train", "validation", "test")}

        print("Five-band cache:", FIVE_BAND.shape)
        print("Log-Mel cache:", LOGMEL20.shape)
        print("BC-ResNet log-Mel cache:", LOGMEL40_BCRESNET.shape)
        print("High-resolution BC-ResNet log-Mel cache:", LOGMEL120_BCRESNET.shape)
        print("Legacy waveform cache:", LEGACY_WAVEFORMS.shape)
        """
    ),
    markdown("## Shared causal Conv-GRU and training utilities"),
    code(
        r"""
        class SequenceDataset(Dataset):
            def __init__(self, values, lengths, indices, mean, std):
                self.values = values
                self.lengths = lengths
                self.indices = np.asarray(indices)
                self.mean = mean.astype(np.float32)
                self.std = std.astype(np.float32)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                index = int(self.indices[item])
                length = int(self.lengths[index])
                values = np.array(self.values[index, :length], dtype=np.float32, copy=True)
                values = (values - self.mean) / self.std
                return torch.from_numpy(values), int(LABELS[index]), index


        def collate_sequences(batch):
            sequences, labels, indices = zip(*batch)
            lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
            padded = torch.zeros(len(sequences), int(lengths.max()), sequences[0].shape[1])
            for index, sequence in enumerate(sequences):
                padded[index, :len(sequence)] = sequence
            return padded, lengths, torch.tensor(labels), torch.tensor(indices)


        class LegacyDataset(Dataset):
            def __init__(self, indices):
                self.indices = np.asarray(indices)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                index = int(self.indices[item])
                values = np.array(LEGACY_WAVEFORMS[index], dtype=np.float32, copy=True)
                return torch.from_numpy(values), int(LABELS[index]), index


        def fit_sequence_normalizer(values, lengths, indices):
            total = np.zeros(values.shape[-1], dtype=np.float64)
            square = np.zeros(values.shape[-1], dtype=np.float64)
            count = 0
            for index in tqdm(indices, desc="Fitting train-only normalization", leave=False):
                block = np.asarray(values[index, :int(lengths[index])], dtype=np.float64)
                total += block.sum(axis=0)
                square += np.square(block).sum(axis=0)
                count += len(block)
            mean = total / count
            variance = np.maximum(square / count - mean ** 2, 1e-6)
            return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


        def make_sequence_loaders(values, lengths, batch_size=TRAIN_BATCH_SIZE):
            mean, std = fit_sequence_normalizer(values, lengths, INDICES["train"])
            datasets = {
                split: SequenceDataset(values, lengths, INDICES[split], mean, std)
                for split in ("train", "validation", "test")
            }
            loaders = {
                split: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=split == "train",
                    collate_fn=collate_sequences,
                    num_workers=NUM_WORKERS,
                    pin_memory=DEVICE.type == "cuda",
                    persistent_workers=NUM_WORKERS > 0,
                    generator=torch.Generator().manual_seed(SEED),
                )
                for split, dataset in datasets.items()
            }
            return datasets, loaders, mean, std


        def make_legacy_loaders(batch_size):
            datasets = {split: LegacyDataset(INDICES[split]) for split in ("train", "validation", "test")}
            loaders = {
                split: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=split == "train",
                    num_workers=NUM_WORKERS,
                    pin_memory=DEVICE.type == "cuda",
                    persistent_workers=NUM_WORKERS > 0,
                    generator=torch.Generator().manual_seed(SEED),
                )
                for split, dataset in datasets.items()
            }
            return datasets, loaders


        class CausalConv1d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size):
                super().__init__()
                self.left_padding = kernel_size - 1
                self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size)

            def forward(self, values):
                return self.convolution(F.pad(values, (self.left_padding, 0)))


        class CausalDigitClassifier(nn.Module):
            def __init__(self, input_features=20, hidden_size=96, layers=2, dropout=0.15):
                super().__init__()
                self.frontend = nn.Sequential(
                    CausalConv1d(input_features, 64, 5), nn.GELU(), nn.Dropout(dropout),
                    CausalConv1d(64, 96, 5), nn.GELU(), nn.Dropout(dropout),
                )
                self.recurrent = nn.GRU(
                    96, hidden_size, num_layers=layers, batch_first=True,
                    bidirectional=False, dropout=dropout if layers > 1 else 0.0,
                )
                self.classifier = nn.Linear(hidden_size, 10)

            def forward(self, values, lengths):
                features = self.frontend(values.transpose(1, 2)).transpose(1, 2)
                packed = pack_padded_sequence(
                    features, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, hidden = self.recurrent(packed)
                return self.classifier(hidden[-1])


        class SubSpectralNorm(nn.Module):
            def __init__(self, channels, groups=5):
                super().__init__()
                self.groups = groups
                self.norm = nn.BatchNorm2d(channels * groups)

            def forward(self, values):
                batch, channels, frequency, time_steps = values.shape
                if frequency % self.groups:
                    raise RuntimeError(
                        f"Frequency dimension {frequency} is not divisible by {self.groups}"
                    )
                values = values.reshape(
                    batch, channels * self.groups, frequency // self.groups, time_steps
                )
                values = self.norm(values)
                return values.reshape(batch, channels, frequency, time_steps)


        class BCConvBNAct(nn.Module):
            def __init__(
                self, in_channels, out_channels, stage_index, kernel_size=3, stride=1,
                groups=1, use_dilation=False, activation="relu", batch_norm=True,
                subspectral_norm=False,
            ):
                super().__init__()
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                if isinstance(stride, int):
                    stride = (stride, stride)
                dilation = 2 ** stage_index if use_dilation else 1
                dilation_pair = (dilation, dilation)
                padding = tuple(
                    dilation * (size - 1) // 2 for size in kernel_size
                )
                layers = [nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation_pair, groups=groups, bias=False,
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
                    raise ValueError(f"Unknown activation: {activation}")
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
                    stride=(stride, 1), groups=in_channels, activation=None,
                    subspectral_norm=True,
                ))
                self.frequency_path = nn.Sequential(*frequency_layers)
                self.temporal_pool = nn.AdaptiveAvgPool2d((1, None))
                self.temporal_path = nn.Sequential(
                    BCConvBNAct(
                        out_channels, out_channels, stage_index, kernel_size=(1, 3),
                        groups=out_channels, use_dilation=True, activation="silu",
                    ),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                    nn.Dropout2d(0.1),
                )

            def forward(self, values):
                shortcut = values
                frequency_values = self.frequency_path(values)
                temporal_values = self.temporal_path(self.temporal_pool(frequency_values))
                output = temporal_values + frequency_values
                if not self.transition:
                    output = output + shortcut
                return F.relu(output, inplace=True)


        class BCResNet(nn.Module):
            # BC-ResNet architecture adapted from Qualcomm's official implementation.
            def __init__(self, scale, num_classes=10):
                super().__init__()
                base_channels = int(scale * 8)
                repeats = [2, 2, 4, 4]
                channels = [
                    base_channels * 2,
                    base_channels,
                    int(base_channels * 1.5),
                    base_channels * 2,
                    int(base_channels * 2.5),
                    base_channels * 4,
                ]
                self.head = nn.Sequential(
                    nn.Conv2d(1, channels[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
                    nn.BatchNorm2d(channels[0]),
                    nn.ReLU(inplace=True),
                )
                stages = []
                for stage_index, repeat_count in enumerate(repeats):
                    stage = []
                    stage_channels = [channels[stage_index]] + [channels[stage_index + 1]] * repeat_count
                    for block_index in range(repeat_count):
                        frequency_stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                        stage.append(BCResBlock(
                            stage_channels[block_index], stage_channels[block_index + 1],
                            stage_index, frequency_stride,
                        ))
                    stages.append(nn.Sequential(*stage))
                self.stages = nn.Sequential(*stages)
                self.classifier = nn.Sequential(
                    nn.Conv2d(
                        channels[-2], channels[-2], kernel_size=(5, 5),
                        padding=(0, 2), groups=channels[-2], bias=False,
                    ),
                    nn.Conv2d(channels[-2], channels[-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels[-1]),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(channels[-1], num_classes, kernel_size=1),
                )

            def forward(self, values, lengths):
                del lengths  # Every BC-ResNet input is padded to exactly one second before the STFT.
                values = values.transpose(1, 2).unsqueeze(1)
                values = self.head(values)
                values = self.stages(values)
                values = self.classifier(values)
                return values.flatten(1)


        class SinusoidalPositionEncoding(nn.Module):
            def __init__(self, width, maximum_steps=256):
                super().__init__()
                positions = torch.arange(maximum_steps, dtype=torch.float32).unsqueeze(1)
                scales = torch.exp(
                    torch.arange(0, width, 2, dtype=torch.float32)
                    * (-math.log(10_000.0) / width)
                )
                encoding = torch.zeros(maximum_steps, width)
                encoding[:, 0::2] = torch.sin(positions * scales)
                encoding[:, 1::2] = torch.cos(positions * scales)
                self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

            def forward(self, values):
                return values + self.encoding[:, :values.shape[1]].to(values.dtype)


        class LogMelTransformerCTC(nn.Module):
            def __init__(
                self, input_features, d_model=TRANSFORMER_CTC_D_MODEL,
                heads=TRANSFORMER_CTC_HEADS, layers=TRANSFORMER_CTC_LAYERS,
                ff_dim=TRANSFORMER_CTC_FF_DIM, dropout=TRANSFORMER_CTC_DROPOUT,
                vocabulary_size=11,
            ):
                super().__init__()
                self.subsampler = nn.Sequential(
                    nn.Conv1d(input_features, d_model, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                )
                self.position = SinusoidalPositionEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
                    dropout=dropout, activation="gelu", batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.output_norm = nn.LayerNorm(d_model)
                self.ctc_head = nn.Linear(d_model, vocabulary_size)

            @staticmethod
            def subsampled_lengths(lengths):
                lengths = (lengths + 1) // 2
                return (lengths + 1) // 2

            def forward(self, values, lengths):
                values = self.subsampler(values.transpose(1, 2)).transpose(1, 2)
                output_lengths = self.subsampled_lengths(lengths)
                values = self.position(values)
                time_index = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
                padding_mask = time_index >= output_lengths.to(values.device).unsqueeze(1)
                values = self.encoder(values, src_key_padding_mask=padding_mask)
                return self.ctc_head(self.output_norm(values)), output_lengths
        """
    ),
    code(
        r"""
        def unpack_logits(model, batch, kind):
            if kind == "sequence":
                values, lengths, labels, indices = batch
                logits = model(values.to(DEVICE, non_blocking=True), lengths)
            else:
                values, labels, indices = batch
                logits = model(values.to(DEVICE, non_blocking=True))
            return logits, labels.to(DEVICE), indices


        @torch.inference_mode()
        def evaluate_torch(model, loader, kind):
            model.eval()
            loss_sum = 0.0
            correct = 0
            count = 0
            rows = []
            for batch in loader:
                logits, labels, indices = unpack_logits(model, batch, kind)
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
                predictions = logits.argmax(dim=1)
                correct += int((predictions == labels).sum())
                count += len(labels)
                for index, truth, prediction in zip(
                    indices.numpy(), labels.cpu().numpy(), predictions.cpu().numpy()
                ):
                    rows.append({
                        "record_index": int(index), "speaker": int(SPEAKERS[index]),
                        "true_digit": int(truth), "predicted_digit": int(prediction),
                        "correct": int(truth) == int(prediction),
                    })
            return loss_sum / count, correct / count, pd.DataFrame(rows)


        TRAINED = {}


        def train_torch_model(name, model, loaders, kind, epochs, learning_rate, metadata=None):
            model = model.to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            history = []
            checkpoint = OUTPUT_ROOT / f"{name}.pt"

            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                train_count = 0
                for batch in loaders["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=DEVICE.type, dtype=torch.float16,
                        enabled=DEVICE.type == "cuda",
                    ):
                        logits, labels, _ = unpack_logits(model, batch, kind)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += float(loss.detach()) * len(labels)
                    train_count += len(labels)

                validation_loss, validation_accuracy, _ = evaluate_torch(
                    model, loaders["validation"], kind
                )
                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss / train_count,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                })
                print(
                    f"{name} epoch {epoch:02d}: train_loss={train_loss/train_count:.4f} "
                    f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4%}"
                )
                if validation_loss < best_loss - 1e-5:
                    best_loss = validation_loss
                    best_epoch = epoch
                    stale = 0
                    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, checkpoint)
                else:
                    stale += 1
                    if stale >= EARLY_STOPPING_PATIENCE:
                        print("Early stopping.")
                        break

            saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
            model.load_state_dict(saved["state_dict"])
            history_table = pd.DataFrame(history)
            history_table.to_csv(OUTPUT_ROOT / f"{name}_history.csv", index=False)
            TRAINED[name] = {
                "model": model,
                "loaders": loaders,
                "kind": kind,
                "checkpoint": checkpoint,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "metadata": metadata or {},
            }
            return model


        def ctc_greedy_decode(token_ids, blank=0):
            decoded = []
            previous = None
            for token in token_ids:
                token = int(token)
                if token != previous and token != blank:
                    decoded.append(token - 1)
                previous = token
            return decoded


        def sequence_edit_distance(reference, hypothesis):
            previous = list(range(len(hypothesis) + 1))
            for row, reference_token in enumerate(reference, start=1):
                current = [row]
                for column, hypothesis_token in enumerate(hypothesis, start=1):
                    current.append(min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (reference_token != hypothesis_token),
                    ))
                previous = current
            return previous[-1]


        @torch.inference_mode()
        def evaluate_ctc(model, loader):
            model.eval()
            loss_sum = 0.0
            count = 0
            edit_errors = 0
            rows = []
            for values, lengths, labels, indices in loader:
                labels_device = labels.to(DEVICE, non_blocking=True)
                logits, output_lengths = model(
                    values.to(DEVICE, non_blocking=True), lengths
                )
                log_probabilities = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                targets = labels_device + 1
                target_lengths = torch.ones_like(lengths)
                loss = F.ctc_loss(
                    log_probabilities, targets, output_lengths.cpu(), target_lengths,
                    blank=0, reduction="sum", zero_infinity=True,
                )
                loss_sum += float(loss)
                token_predictions = logits.argmax(dim=-1).cpu()
                for batch_index, (index, truth) in enumerate(zip(indices.numpy(), labels.numpy())):
                    decoded = ctc_greedy_decode(
                        token_predictions[batch_index, :int(output_lengths[batch_index])]
                    )
                    reference = [int(truth)]
                    distance = sequence_edit_distance(reference, decoded)
                    exact = decoded == reference
                    edit_errors += distance
                    rows.append({
                        "record_index": int(index), "speaker": int(SPEAKERS[index]),
                        "true_digit": int(truth),
                        "predicted_digit": decoded[0] if decoded else -1,
                        "decoded_sequence": " ".join(map(str, decoded)),
                        "decoded_length": len(decoded), "edit_distance": distance,
                        "correct": exact,
                    })
                    count += 1
            table = pd.DataFrame(rows)
            return loss_sum / count, float(table.correct.mean()), edit_errors / count, table


        def train_ctc_model(name, model, loaders, epochs, learning_rate, metadata=None):
            model = model.to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            history = []
            checkpoint = OUTPUT_ROOT / f"{name}.pt"

            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                train_count = 0
                for values, lengths, labels, _ in loaders["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    labels_device = labels.to(DEVICE, non_blocking=True)
                    with torch.autocast(
                        device_type=DEVICE.type, dtype=torch.float16,
                        enabled=DEVICE.type == "cuda",
                    ):
                        logits, output_lengths = model(
                            values.to(DEVICE, non_blocking=True), lengths
                        )
                    log_probabilities = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                    loss = F.ctc_loss(
                        log_probabilities, labels_device + 1,
                        output_lengths.cpu(), torch.ones_like(lengths),
                        blank=0, reduction="mean", zero_infinity=True,
                    )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += float(loss.detach()) * len(labels)
                    train_count += len(labels)
                scheduler.step()

                validation_loss, validation_accuracy, validation_ter, _ = evaluate_ctc(
                    model, loaders["validation"]
                )
                history.append({
                    "epoch": epoch, "train_loss": train_loss / train_count,
                    "validation_loss": validation_loss,
                    "validation_exact_sequence_accuracy": validation_accuracy,
                    "validation_token_error_rate": validation_ter,
                })
                print(
                    f"{name} epoch {epoch:02d}: train_loss={train_loss/train_count:.4f} "
                    f"val_loss={validation_loss:.4f} exact={validation_accuracy:.4%} "
                    f"TER={validation_ter:.4%}"
                )
                if validation_loss < best_loss - 1e-5:
                    best_loss = validation_loss
                    best_epoch = epoch
                    stale = 0
                    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, checkpoint)
                else:
                    stale += 1
                    if stale >= EARLY_STOPPING_PATIENCE:
                        print("Early stopping.")
                        break

            saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
            model.load_state_dict(saved["state_dict"])
            pd.DataFrame(history).to_csv(OUTPUT_ROOT / f"{name}_history.csv", index=False)
            TRAINED[name] = {
                "model": model, "loaders": loaders, "kind": "ctc",
                "checkpoint": checkpoint, "best_epoch": best_epoch,
                "best_validation_loss": best_loss, "metadata": metadata or {},
            }
            return model
        """
    ),
    markdown("## 1. Digital five-band 20D + causal Conv-GRU"),
    code(
        r"""
        band_datasets, band_loaders, band_mean, band_std = make_sequence_loaders(
            FIVE_BAND, FIVE_BAND_LENGTHS
        )
        band_name = "matched_five_band_convgru" if BANDS_ARE_FROM_FINAL_SWEEP else "provisional_five_band_convgru"
        train_torch_model(
            band_name,
            CausalDigitClassifier(input_features=20),
            band_loaders,
            kind="sequence",
            epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            metadata={
                "frontend": "five causal peak filters; four 2.5 ms envelope subframes per 10 ms step",
                "features_per_step": 20,
                "bands_are_from_final_sweep": BANDS_ARE_FROM_FINAL_SWEEP,
                "bands": bands.to_dict(orient="records"),
                "band_calibration_source": BAND_CALIBRATION_SOURCE,
                "band_calibration_limitation": BAND_CALIBRATION_LIMITATION,
                "mean": band_mean.tolist(), "std": band_std.tolist(),
            },
        )
        """
    ),
    markdown("## 2. Digital 20-bin log-Mel + causal Conv-GRU"),
    code(
        r"""
        mel_datasets, mel_loaders, mel_mean, mel_std = make_sequence_loaders(
            LOGMEL20, LOGMEL20_LENGTHS
        )
        train_torch_model(
            "logmel20_convgru",
            CausalDigitClassifier(input_features=20),
            mel_loaders,
            kind="sequence",
            epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            metadata={
                "frontend": "20 HTK log-Mel bins; 10 ms frame; 10 ms hop; no overlap",
                "features_per_step": 20,
                "mean": mel_mean.tolist(), "std": mel_std.tolist(),
            },
        )
        """
    ),
    markdown(
        """
        ## 3-5. Digital log-Mel + BC-ResNet

        [BC-ResNet](https://github.com/Qualcomm-AI-research/bcresnet) is the
        task-appropriate primary digital encoder. This notebook follows the
        official 40-bin, 30 ms window, 10 ms hop feature geometry and the official
        scale-1 and scale-8 architectures, adapted from 12 to 10 output classes.
        It trains new AudioMNIST checkpoints, so it does not claim to reproduce the
        Speech Commands scores from the paper.

        BC-ResNet performs fixed-utterance classification with symmetric temporal
        convolution and global pooling. It is a strong digital accuracy/efficiency
        baseline, but it is not a causal streaming latency match for the physical
        Conv-GRU. The 120-bin row changes only spectral resolution; it uses the same
        BC-ResNet-8 encoder and training protocol as the 40-bin row.
        """
    ),
    code(
        r"""
        bc_mel_datasets, bc_mel_loaders, bc_mel_mean, bc_mel_std = make_sequence_loaders(
            LOGMEL40_BCRESNET, LOGMEL40_BCRESNET_LENGTHS,
            batch_size=BCRESNET_BATCH_SIZE,
        )

        for scale in (1, 8):
            train_torch_model(
                f"logmel40_bcresnet{scale}",
                BCResNet(scale=scale, num_classes=10),
                bc_mel_loaders,
                kind="sequence",
                epochs=BCRESNET_MAX_EPOCHS,
                learning_rate=BCRESNET_LEARNING_RATE,
                metadata={
                    "frontend": "40 HTK log-Mel bins; 30 ms window; 10 ms hop; one-second right padding",
                    "features_per_frame": BCRESNET_N_MELS,
                    "encoder": f"BC-ResNet-{scale}",
                    "architecture_source": "https://github.com/Qualcomm-AI-research/bcresnet",
                    "training_note": "AudioMNIST adaptation with AdamW and validation early stopping",
                    "causal": False,
                    "mean": bc_mel_mean.tolist(), "std": bc_mel_std.tolist(),
                },
            )

        bc120_mel_datasets, bc120_mel_loaders, bc120_mel_mean, bc120_mel_std = make_sequence_loaders(
            LOGMEL120_BCRESNET, LOGMEL40_BCRESNET_LENGTHS,
            batch_size=BCRESNET_BATCH_SIZE,
        )
        train_torch_model(
            "logmel120_bcresnet8",
            BCResNet(scale=8, num_classes=10),
            bc120_mel_loaders,
            kind="sequence",
            epochs=BCRESNET_MAX_EPOCHS,
            learning_rate=BCRESNET_LEARNING_RATE,
            metadata={
                "frontend": "120 HTK log-Mel bins; 30 ms window; 10 ms hop; one-second right padding",
                "features_per_frame": BCRESNET_HIGH_RES_N_MELS,
                "encoder": "BC-ResNet-8",
                "architecture_source": "https://github.com/Qualcomm-AI-research/bcresnet",
                "training_note": "AudioMNIST spectral-resolution ablation with AdamW and validation early stopping",
                "causal": False,
                "mean": bc120_mel_mean.tolist(), "std": bc120_mel_std.tolist(),
            },
        )
        """
    ),
    markdown(
        """
        ## 6-7. Digital log-Mel + Transformer-CTC

        Both CTC rows use the same two-layer stride-2 temporal subsampler and
        four-layer full-context Transformer encoder. The vocabulary contains one
        CTC blank plus digits 0-9. AudioMNIST has one digit per file, so each target
        sequence has length one. Exact decoded-sequence accuracy is directly
        comparable to digit accuracy, but this is not evidence of continuous
        multi-digit recognition; that requires a sequence dataset.
        """
    ),
    code(
        r"""
        for name, feature_count, loaders, feature_label, mean, std in [
            (
                "logmel40_transformer_ctc", BCRESNET_N_MELS, bc_mel_loaders,
                "40 HTK log-Mel bins; 30 ms window; 10 ms hop", bc_mel_mean, bc_mel_std,
            ),
            (
                "logmel120_transformer_ctc", BCRESNET_HIGH_RES_N_MELS, bc120_mel_loaders,
                "120 HTK log-Mel bins; 30 ms window; 10 ms hop", bc120_mel_mean, bc120_mel_std,
            ),
        ]:
            train_ctc_model(
                name,
                LogMelTransformerCTC(input_features=feature_count),
                loaders,
                epochs=TRANSFORMER_CTC_MAX_EPOCHS,
                learning_rate=TRANSFORMER_CTC_LEARNING_RATE,
                metadata={
                    "frontend": feature_label,
                    "features_per_frame": feature_count,
                    "encoder": (
                        f"two stride-2 temporal convolutions + "
                        f"{TRANSFORMER_CTC_LAYERS}-layer Transformer, "
                        f"d_model={TRANSFORMER_CTC_D_MODEL}, "
                        f"heads={TRANSFORMER_CTC_HEADS}, ff={TRANSFORMER_CTC_FF_DIM}"
                    ),
                    "decoder": "CTC greedy decoding; blank=0; digit tokens=1..10",
                    "target_length": 1,
                    "causal": False,
                    "mean": mean.tolist(), "std": std.tolist(),
                },
            )
        """
    ),
    markdown(
        """
        ## 8. Digital 20-bin log-Mel + RBF SVM

        The SVM receives the arithmetic mean of each Mel bin over the isolated
        digit. It therefore has exactly 20 scalar inputs but does not retain the
        Conv-GRU's temporal ordering. Its hyperparameters are frozen before the
        test split is evaluated.
        """
    ),
    code(
        r"""
        def pooled_logmel(indices):
            output = np.empty((len(indices), 20), dtype=np.float32)
            for row, index in enumerate(tqdm(indices, desc="Pooling log-Mel", leave=False)):
                length = int(LOGMEL20_LENGTHS[index])
                output[row] = np.asarray(LOGMEL20[index, :length], dtype=np.float32).mean(axis=0)
            return output


        svm_train_x = pooled_logmel(INDICES["train"])
        svm_validation_x = pooled_logmel(INDICES["validation"])
        svm = Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(C=2.0, gamma="scale", kernel="rbf", class_weight="balanced", cache_size=4096)),
        ])
        svm.fit(svm_train_x, LABELS[INDICES["train"]])
        svm_validation_prediction = svm.predict(svm_validation_x)
        print("RBF SVM validation accuracy:", accuracy_score(
            LABELS[INDICES["validation"]], svm_validation_prediction
        ))
        svm_path = OUTPUT_ROOT / "logmel20_rbf_svm.joblib"
        joblib.dump(svm, svm_path)
        TRAINED["logmel20_rbf_svm"] = {
            "model": svm,
            "checkpoint": svm_path,
            "metadata": {"features": 20, "temporal_pooling": "arithmetic mean"},
        }
        """
    ),
    markdown(
        """
        ## Official legacy architecture references

        These classes translate the official Caffe prototxt layer dimensions into
        PyTorch. AudioNet uses the one-second 8 kHz waveform. AlexNet computes the
        repository's Hann STFT geometry and feeds a 227 x 227 decibel image to the
        CNN. Training uses AdamW and validation early stopping for practical Colab
        execution, so report these as **PyTorch architecture reimplementations**,
        not reproduced published checkpoints.
        """
    ),
    code(
        r"""
        class OfficialAudioNetPyTorch(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, 100, 3, padding=1), nn.ReLU(), nn.MaxPool1d(3, 2, ceil_mode=True),
                    nn.Conv1d(100, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2, ceil_mode=True),
                    nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2, ceil_mode=True),
                    nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2, ceil_mode=True),
                    nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2, ceil_mode=True),
                    nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2, ceil_mode=True),
                )
                self.fc7 = nn.Linear(128 * 125, 1024)
                self.fc8 = nn.Linear(1024, 512)
                self.fc9 = nn.Linear(512, 10)
                self.dropout = nn.Dropout(0.5)

            def forward(self, values):
                values = self.features(values.unsqueeze(1)).flatten(1)
                values = self.dropout(self.fc7(values))
                values = self.dropout(self.fc8(values))
                return self.fc9(values)


        class OfficialAlexNetCorePyTorch(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 96, 11, stride=4), nn.ReLU(), nn.MaxPool2d(3, 2),
                    nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(), nn.MaxPool2d(3, 2),
                    nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(),
                    nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(), nn.MaxPool2d(3, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(256 * 6 * 6, 1024), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(1024, 10),
                )

            def forward(self, images):
                return self.classifier(self.features(images).flatten(1))


        class OfficialAlexNetPyTorch(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = OfficialAlexNetCorePyTorch()
                self.register_buffer("hann", torch.hann_window(455, periodic=True), persistent=False)

            def spectrogram(self, values):
                # SciPy's default padded STFT has 230 columns for an 8000-sample input.
                # Sixteen trailing zeros make torch.stft reproduce that count.
                with torch.autocast(device_type=values.device.type, enabled=False):
                    spectrum = torch.stft(
                        F.pad(values.float(), (0, 16)), n_fft=455, hop_length=35,
                        win_length=455, window=self.hann.float(), center=True,
                        pad_mode="constant", normalized=False, onesided=True,
                        return_complex=True,
                    )
                    magnitude = spectrum.abs()[:, :227, 2:-1]
                    if magnitude.shape[-2:] != (227, 227):
                        magnitude = F.interpolate(
                            magnitude.unsqueeze(1), size=(227, 227),
                            mode="bilinear", align_corners=False,
                        ).squeeze(1)
                    reference = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
                    return 20.0 * torch.log10(magnitude.clamp_min(1e-12) / reference)

            def forward(self, values):
                return self.core(self.spectrogram(values).unsqueeze(1))


        legacy_datasets, legacy_loaders = make_legacy_loaders(LEGACY_BATCH_SIZE)
        alex_datasets, alex_loaders = make_legacy_loaders(ALEXNET_BATCH_SIZE)
        """
    ),
    markdown("## 9. AudioNet PyTorch architecture reference"),
    code(
        r"""
        train_torch_model(
            "official_audionet_pytorch",
            OfficialAudioNetPyTorch(),
            legacy_loaders,
            kind="fixed",
            epochs=LEGACY_MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            metadata={
                "architecture_source": "AudioMNIST models/AudioNet/audionet_train.prototxt",
                "input": "8000-sample 8 kHz embedded waveform",
                "training_note": "PyTorch reimplementation with AdamW and early stopping",
            },
        )
        """
    ),
    markdown("## 10. AlexNet PyTorch architecture reference"),
    code(
        r"""
        train_torch_model(
            "official_alexnet_pytorch",
            OfficialAlexNetPyTorch(),
            alex_loaders,
            kind="fixed",
            epochs=LEGACY_MAX_EPOCHS,
            learning_rate=ALEXNET_LEARNING_RATE,
            metadata={
                "architecture_source": "AudioMNIST models/AlexNet/alexnet_train.prototxt",
                "input": "227 x 227 dB STFT image from one-second 8 kHz waveform",
                "training_note": "PyTorch reimplementation with AdamW and early stopping",
            },
        )
        """
    ),
    markdown(
        """
        ## Held-out test evaluation

        Run this cell only after every model above has finished and its validation
        checkpoint has been frozen. Nothing in this cell changes a model or a
        normalizer.
        """
    ),
    code(
        r"""
        TEST_RESULTS = {}
        PREDICTIONS = {}

        for name, item in TRAINED.items():
            if name == "logmel20_rbf_svm":
                test_x = pooled_logmel(INDICES["test"])
                prediction = item["model"].predict(test_x)
                truth = LABELS[INDICES["test"]]
                table = pd.DataFrame({
                    "record_index": INDICES["test"],
                    "speaker": SPEAKERS[INDICES["test"]],
                    "true_digit": truth,
                    "predicted_digit": prediction,
                    "correct": truth == prediction,
                })
                accuracy = float(accuracy_score(truth, prediction))
                token_error_rate = None
            elif item["kind"] == "ctc":
                _, accuracy, token_error_rate, table = evaluate_ctc(
                    item["model"], item["loaders"]["test"]
                )
            else:
                _, accuracy, table = evaluate_torch(
                    item["model"], item["loaders"]["test"], item["kind"]
                )
                token_error_rate = None

            speaker_accuracy = table.groupby("speaker").correct.mean()
            TEST_RESULTS[name] = {
                "accuracy": accuracy,
                "macro_speaker_accuracy": float(speaker_accuracy.mean()),
                "worst_speaker_accuracy": float(speaker_accuracy.min()),
                "token_error_rate": token_error_rate,
            }
            PREDICTIONS[name] = table
            table.to_csv(OUTPUT_ROOT / f"{name}_heldout_predictions.csv", index=False)
            print(
                f"{name}: accuracy={accuracy:.4%}, "
                f"macro-speaker={speaker_accuracy.mean():.4%}, "
                f"worst-speaker={speaker_accuracy.min():.4%}"
                + (f", TER={token_error_rate:.4%}" if token_error_rate is not None else "")
            )
        """
    ),
    markdown(
        """
        ## Complexity and latency table

        Latency is the median of batch-one CPU runs on one held-out utterance near
        the median duration. File I/O is excluded; feature extraction is included
        in the end-to-end column. PyTorch MACs count multiply-accumulates in learned
        layers. Transformer MACs are calculated analytically for its convolutions,
        attention projections/products, feed-forward layers, and CTC head. AlexNet's
        FFT is timed but not included in THOP MACs. RBF SVM MACs
        are a kernel dot-product proxy (`support vectors x 20`) and its reported
        learned-scalar count is not directly equivalent to neural-network
        parameters.
        """
    ),
    code(
        r"""
        def convgru_macs(time_steps, input_features=20, hidden=96):
            convolution = time_steps * (input_features * 64 * 5 + 64 * 96 * 5)
            recurrent = time_steps * (
                3 * hidden * (96 + hidden) + 3 * hidden * (hidden + hidden)
            )
            classifier = hidden * 10
            return int(convolution + recurrent + classifier)


        def transformer_ctc_macs(
            time_steps, input_features, d_model=TRANSFORMER_CTC_D_MODEL,
            layers=TRANSFORMER_CTC_LAYERS, ff_dim=TRANSFORMER_CTC_FF_DIM,
            vocabulary_size=11,
        ):
            first_steps = (time_steps + 1) // 2
            encoded_steps = (first_steps + 1) // 2
            subsampling = (
                first_steps * input_features * d_model * 5
                + encoded_steps * d_model * d_model * 5
            )
            per_layer = (
                4 * encoded_steps * d_model * d_model
                + 2 * encoded_steps * encoded_steps * d_model
                + 2 * encoded_steps * d_model * ff_dim
            )
            output = encoded_steps * d_model * vocabulary_size
            return int(subsampling + layers * per_layer + output)


        def latency_ms(function, warmup=LATENCY_WARMUP, repeats=LATENCY_REPEATS):
            for _ in range(warmup):
                function()
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                function()
                timings.append((time.perf_counter() - started) * 1000.0)
            return float(np.median(timings)), float(np.percentile(timings, 95))


        # Choose a held-out sample near the median source duration.
        duration_rows = []
        for index in INDICES["test"]:
            sample_rate, values = read_mono(records.loc[index, "path"])
            duration_rows.append((index, len(values) / sample_rate))
        median_duration = float(np.median([item[1] for item in duration_rows]))
        benchmark_index = min(duration_rows, key=lambda item: abs(item[1] - median_duration))[0]
        benchmark_rate, benchmark_raw = read_mono(records.loc[benchmark_index, "path"])
        benchmark_48k = resample_audio(benchmark_raw, benchmark_rate, DIGITAL_SAMPLE_RATE)
        benchmark_16k = resample_audio(benchmark_raw, benchmark_rate, LOGMEL_SAMPLE_RATE)
        benchmark_legacy = legacy_waveform(
            benchmark_raw, benchmark_rate, Path(records.loc[benchmark_index, "path"]).name
        )

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        benchmark_rows = []

        def benchmark_sequence(name, frontend, mean, std, input_label):
            model = copy.deepcopy(TRAINED[name]["model"]).cpu().eval()
            prepared = frontend()
            normalized = (prepared - mean) / std
            tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
            lengths = torch.tensor([len(prepared)])
            frontend_median, _ = latency_ms(frontend)
            with torch.inference_mode():
                model_median, _ = latency_ms(lambda: model(tensor, lengths))
                e2e_median, e2e_p95 = latency_ms(lambda: model(
                    torch.from_numpy(((frontend() - mean) / std).astype(np.float32)).unsqueeze(0),
                    torch.tensor([len(prepared)]),
                ))
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            benchmark_rows.append({
                "model": name, "input": input_label, "test_accuracy": TEST_RESULTS[name]["accuracy"],
                "macro_speaker_accuracy": TEST_RESULTS[name]["macro_speaker_accuracy"],
                "parameters_or_stored_scalars": parameters, "parameter_definition": "trainable NN parameters",
                "MACs_per_benchmark_utterance": convgru_macs(len(prepared)),
                "checkpoint_size_MiB": TRAINED[name]["checkpoint"].stat().st_size / 2**20,
                "frontend_latency_ms_median": frontend_median,
                "model_latency_ms_median": model_median,
                "end_to_end_latency_ms_median": e2e_median,
                "end_to_end_latency_ms_p95": e2e_p95,
            })


        benchmark_sequence(
            band_name, lambda: matched_five_band_features(benchmark_48k),
            band_mean, band_std, "20 values/10 ms (5 bands x 4 subframes)",
        )
        benchmark_sequence(
            "logmel20_convgru", lambda: logmel_20_features(benchmark_16k),
            mel_mean, mel_std, "20 log-Mel values/10 ms",
        )

        # Published BC-ResNet architecture at two scales.
        def benchmark_bcresnet(name, input_label, frontend, mean, std):
            model = copy.deepcopy(TRAINED[name]["model"]).cpu().eval()
            prepared = frontend()
            normalized = (prepared - mean) / std
            tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
            lengths = torch.tensor([len(prepared)])
            frontend_median, _ = latency_ms(frontend)
            with torch.inference_mode():
                model_median, _ = latency_ms(lambda: model(tensor, lengths))
                e2e_median, e2e_p95 = latency_ms(lambda: model(
                    torch.from_numpy(
                        ((frontend() - mean) / std).astype(np.float32)
                    ).unsqueeze(0),
                    lengths,
                ))
                macs, _ = profile(model, inputs=(tensor, lengths), verbose=False)
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            benchmark_rows.append({
                "model": name, "input": input_label,
                "test_accuracy": TEST_RESULTS[name]["accuracy"],
                "macro_speaker_accuracy": TEST_RESULTS[name]["macro_speaker_accuracy"],
                "parameters_or_stored_scalars": parameters,
                "parameter_definition": "trainable NN parameters",
                "MACs_per_benchmark_utterance": int(macs),
                "checkpoint_size_MiB": TRAINED[name]["checkpoint"].stat().st_size / 2**20,
                "frontend_latency_ms_median": frontend_median,
                "model_latency_ms_median": model_median,
                "end_to_end_latency_ms_median": e2e_median,
                "end_to_end_latency_ms_p95": e2e_p95,
            })


        benchmark_bcresnet(
            "logmel40_bcresnet1", "40 log-Mel values/10 ms; BC-ResNet-1",
            lambda: logmel_40_bcresnet_features(benchmark_16k), bc_mel_mean, bc_mel_std,
        )
        benchmark_bcresnet(
            "logmel40_bcresnet8", "40 log-Mel values/10 ms; BC-ResNet-8",
            lambda: logmel_40_bcresnet_features(benchmark_16k), bc_mel_mean, bc_mel_std,
        )
        benchmark_bcresnet(
            "logmel120_bcresnet8", "120 log-Mel values/10 ms; BC-ResNet-8",
            lambda: logmel_120_bcresnet_features(benchmark_16k),
            bc120_mel_mean, bc120_mel_std,
        )

        def benchmark_transformer_ctc(
            name, input_label, frontend, mean, std, input_features
        ):
            model = copy.deepcopy(TRAINED[name]["model"]).cpu().eval()
            prepared = frontend()
            normalized = (prepared - mean) / std
            tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
            lengths = torch.tensor([len(prepared)])
            frontend_median, _ = latency_ms(frontend)
            with torch.inference_mode():
                model_median, _ = latency_ms(lambda: model(tensor, lengths))
                e2e_median, e2e_p95 = latency_ms(lambda: model(
                    torch.from_numpy(
                        ((frontend() - mean) / std).astype(np.float32)
                    ).unsqueeze(0),
                    lengths,
                ))
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            benchmark_rows.append({
                "model": name, "input": input_label,
                "test_accuracy": TEST_RESULTS[name]["accuracy"],
                "macro_speaker_accuracy": TEST_RESULTS[name]["macro_speaker_accuracy"],
                "token_error_rate": TEST_RESULTS[name]["token_error_rate"],
                "parameters_or_stored_scalars": parameters,
                "parameter_definition": "trainable NN parameters",
                "MACs_per_benchmark_utterance": transformer_ctc_macs(
                    len(prepared), input_features
                ),
                "checkpoint_size_MiB": TRAINED[name]["checkpoint"].stat().st_size / 2**20,
                "frontend_latency_ms_median": frontend_median,
                "model_latency_ms_median": model_median,
                "end_to_end_latency_ms_median": e2e_median,
                "end_to_end_latency_ms_p95": e2e_p95,
            })


        benchmark_transformer_ctc(
            "logmel40_transformer_ctc", "40 log-Mel values/10 ms; Transformer-CTC",
            lambda: logmel_40_bcresnet_features(benchmark_16k),
            bc_mel_mean, bc_mel_std, BCRESNET_N_MELS,
        )
        benchmark_transformer_ctc(
            "logmel120_transformer_ctc", "120 log-Mel values/10 ms; Transformer-CTC",
            lambda: logmel_120_bcresnet_features(benchmark_16k),
            bc120_mel_mean, bc120_mel_std, BCRESNET_HIGH_RES_N_MELS,
        )

        # RBF SVM.
        svm_model = TRAINED["logmel20_rbf_svm"]["model"]
        svm_frontend = lambda: logmel_20_features(benchmark_16k).mean(axis=0, keepdims=True)
        svm_input = svm_frontend()
        svm_frontend_latency, _ = latency_ms(svm_frontend)
        svm_model_latency, _ = latency_ms(lambda: svm_model.predict(svm_input))
        svm_e2e_latency, svm_e2e_p95 = latency_ms(lambda: svm_model.predict(svm_frontend()))
        svc = svm_model.named_steps["svc"]
        scaler = svm_model.named_steps["scale"]
        svm_scalars = (
            svc.support_vectors_.size + svc.dual_coef_.size + svc.intercept_.size
            + scaler.mean_.size + scaler.scale_.size
        )
        benchmark_rows.append({
            "model": "logmel20_rbf_svm", "input": "20 utterance-pooled log-Mel values",
            "test_accuracy": TEST_RESULTS["logmel20_rbf_svm"]["accuracy"],
            "macro_speaker_accuracy": TEST_RESULTS["logmel20_rbf_svm"]["macro_speaker_accuracy"],
            "parameters_or_stored_scalars": int(svm_scalars),
            "parameter_definition": "stored SVM/scaler learned scalars",
            "MACs_per_benchmark_utterance": int(len(svc.support_) * 20),
            "checkpoint_size_MiB": TRAINED["logmel20_rbf_svm"]["checkpoint"].stat().st_size / 2**20,
            "frontend_latency_ms_median": svm_frontend_latency,
            "model_latency_ms_median": svm_model_latency,
            "end_to_end_latency_ms_median": svm_e2e_latency,
            "end_to_end_latency_ms_p95": svm_e2e_p95,
        })

        # AudioNet and AlexNet. Both include 8 kHz waveform preparation; AlexNet's model timing includes STFT.
        for name, input_label in [
            ("official_audionet_pytorch", "8000 waveform samples"),
            ("official_alexnet_pytorch", "227 x 227 STFT image"),
        ]:
            model = copy.deepcopy(TRAINED[name]["model"]).cpu().eval()
            tensor = torch.from_numpy(benchmark_legacy).unsqueeze(0)
            raw_frontend = lambda: legacy_waveform(
                benchmark_raw, benchmark_rate, Path(records.loc[benchmark_index, "path"]).name
            )
            frontend_latency, _ = latency_ms(raw_frontend)
            with torch.inference_mode():
                model_latency, _ = latency_ms(lambda: model(tensor))
                e2e_latency, e2e_p95 = latency_ms(
                    lambda: model(torch.from_numpy(raw_frontend()).unsqueeze(0))
                )
            if name == "official_audionet_pytorch":
                macs, _ = profile(model, inputs=(tensor,), verbose=False)
            else:
                image = torch.zeros(1, 1, 227, 227)
                macs, _ = profile(model.core, inputs=(image,), verbose=False)
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            benchmark_rows.append({
                "model": name, "input": input_label,
                "test_accuracy": TEST_RESULTS[name]["accuracy"],
                "macro_speaker_accuracy": TEST_RESULTS[name]["macro_speaker_accuracy"],
                "parameters_or_stored_scalars": parameters,
                "parameter_definition": "trainable NN parameters",
                "MACs_per_benchmark_utterance": int(macs),
                "checkpoint_size_MiB": TRAINED[name]["checkpoint"].stat().st_size / 2**20,
                "frontend_latency_ms_median": frontend_latency,
                "model_latency_ms_median": model_latency,
                "end_to_end_latency_ms_median": e2e_latency,
                "end_to_end_latency_ms_p95": e2e_p95,
            })

        torch.set_num_threads(original_threads)
        results = pd.DataFrame(benchmark_rows)
        results["test_accuracy_percent"] = 100.0 * results.test_accuracy
        results["macro_speaker_accuracy_percent"] = 100.0 * results.macro_speaker_accuracy
        results["MACs_million"] = results.MACs_per_benchmark_utterance / 1e6
        results["benchmark_duration_s"] = median_duration
        results["latency_device"] = "CPU, one PyTorch thread"
        results.to_csv(OUTPUT_ROOT / "digital_baseline_comparison.csv", index=False)

        # Add the already measured fabricated-device result only where its metrics are known.
        # Unknown physical front-end latency and energy are deliberately not invented here.
        accuracy_parameter_frontier = results.loc[
            results.parameter_definition == "trainable NN parameters",
            ["model", "test_accuracy", "parameters_or_stored_scalars"],
        ].copy()
        accuracy_parameter_frontier["result_origin"] = "clean digital AudioMNIST rerun"
        accuracy_parameter_frontier = pd.concat([
            accuracy_parameter_frontier,
            pd.DataFrame([{
                "model": "fabricated_5resonator_convgru_reported",
                "test_accuracy": 0.9451666666666667,
                "parameters_or_stored_scalars": 149_994,
                "result_origin": "held-out fabricated-device experiment",
            }]),
        ], ignore_index=True)
        accuracy_parameter_frontier["test_accuracy_percent"] = (
            100.0 * accuracy_parameter_frontier.test_accuracy
        )
        accuracy_parameter_frontier.to_csv(
            OUTPUT_ROOT / "accuracy_parameter_frontier.csv", index=False
        )

        environment = {
            "platform": platform.platform(), "processor": platform.processor(),
            "torch_version": torch.__version__,
            "training_device": str(DEVICE),
            "training_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "latency_protocol": {
                "device": "CPU", "torch_threads": 1, "batch_size": 1,
                "warmup_runs": LATENCY_WARMUP, "timed_runs": LATENCY_REPEATS,
                "file_io_included": False, "feature_extraction_in_end_to_end": True,
                "benchmark_record_index": int(benchmark_index),
                "benchmark_duration_s": median_duration,
            },
        }
        (OUTPUT_ROOT / "benchmark_environment.json").write_text(json.dumps(environment, indent=2) + "\n")

        display(results[[
            "model", "input", "test_accuracy_percent", "macro_speaker_accuracy_percent",
            "token_error_rate", "parameters_or_stored_scalars", "MACs_million", "checkpoint_size_MiB",
            "frontend_latency_ms_median", "model_latency_ms_median",
            "end_to_end_latency_ms_median", "end_to_end_latency_ms_p95",
        ]].round(4))
        display(accuracy_parameter_frontier.sort_values("parameters_or_stored_scalars"))
        """
    ),
    markdown("## Figures and paper-ready exports"),
    code(
        r"""
        order = results.sort_values("test_accuracy", ascending=False).model.tolist()
        figure, axes = plt.subplots(1, 3, figsize=(21, 5))
        sns.barplot(data=results, x="test_accuracy_percent", y="model", order=order, ax=axes[0], color="#21867a")
        axes[0].axvline(10.0, color="black", linestyle="--", linewidth=1, label="chance")
        axes[0].set(xlabel="Held-out digit accuracy (%)", ylabel="", xlim=(0, 100))
        axes[0].legend()
        sns.scatterplot(
            data=accuracy_parameter_frontier,
            x="parameters_or_stored_scalars", y="test_accuracy_percent",
            hue="result_origin", style="result_origin", s=110, ax=axes[1],
        )
        axes[1].set(xscale="log", xlabel="Trainable parameters (log scale)",
                    ylabel="Held-out digit accuracy (%)")
        sns.scatterplot(
            data=results.loc[results.parameter_definition == "trainable NN parameters"],
            x="MACs_million", y="test_accuracy_percent",
            hue="model", s=110, ax=axes[2], legend=False,
        )
        axes[2].set(xscale="log", xlabel="MACs per benchmark utterance (millions, log scale)",
                    ylabel="Held-out digit accuracy (%)")
        figure.suptitle("AudioMNIST clean-digital baselines on published speaker split 1")
        figure.tight_layout()
        figure.savefig(OUTPUT_ROOT / "accuracy_complexity_comparison.png", dpi=220, bbox_inches="tight")
        plt.show()

        confusion_columns = 4
        confusion_rows = math.ceil(len(PREDICTIONS) / confusion_columns)
        figure, axes = plt.subplots(
            confusion_rows, confusion_columns,
            figsize=(5 * confusion_columns, 5 * confusion_rows), constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(-1)
        for axis, (name, table) in zip(axes.flat, PREDICTIONS.items()):
            matrix = confusion_matrix(table.true_digit, table.predicted_digit, labels=np.arange(10), normalize="true")
            sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", square=True, cbar=False, ax=axis)
            axis.set_title(f"{name}\naccuracy={table.correct.mean():.2%}")
            axis.set(xlabel="Predicted digit", ylabel="True digit")
        for axis in axes.flat[len(PREDICTIONS):]:
            axis.axis("off")
        figure.savefig(OUTPUT_ROOT / "heldout_confusion_matrices.png", dpi=220, bbox_inches="tight")
        plt.show()

        summary = {
            "dataset": "AudioMNIST",
            "split": "published digit split 1",
            "split_counts": {key: int(len(value)) for key, value in INDICES.items()},
            "bands_are_from_final_sweep": BANDS_ARE_FROM_FINAL_SWEEP,
            "band_calibration_source": BAND_CALIBRATION_SOURCE,
            "band_calibration_limitation": BAND_CALIBRATION_LIMITATION,
            "physical_reference": {
                "features_per_10ms_step": 20,
                "model": "same causal Conv-GRU",
                "parameters": 149_994,
                "heldout_test_accuracy": 0.9451666666666667,
                "correct": 5671,
                "test_examples": 6000,
            },
            "digital_results": TEST_RESULTS,
            "primary_digital_encoder": {
                "name": "BC-ResNet-8",
                "input": "40-bin log-Mel, 30 ms window, 10 ms hop",
                "causal": False,
                "architecture_source": "https://github.com/Qualcomm-AI-research/bcresnet",
            },
            "spectral_resolution_ablation": {
                "name": "BC-ResNet-8",
                "input": "120-bin log-Mel, 30 ms window, 10 ms hop",
                "parameter_count_expected_to_match_40_bin_model": True,
            },
            "transformer_ctc_baselines": {
                "mel_bins": [40, 120],
                "encoder": (
                    "two stride-2 temporal convolutions + four-layer full-context Transformer"
                ),
                "d_model": TRANSFORMER_CTC_D_MODEL,
                "attention_heads": TRANSFORMER_CTC_HEADS,
                "feedforward_dimension": TRANSFORMER_CTC_FF_DIM,
                "ctc_vocabulary_size": 11,
                "target_tokens_per_audiomnist_file": 1,
                "limitation": (
                    "Single-digit AudioMNIST evaluates CTC mechanics and exact token decoding, "
                    "not continuous multi-digit ASR."
                ),
            },
            "small_digital_control": "BC-ResNet-1",
            "comparison_csv": str(OUTPUT_ROOT / "digital_baseline_comparison.csv"),
            "accuracy_parameter_frontier_csv": str(OUTPUT_ROOT / "accuracy_parameter_frontier.csv"),
        }
        (OUTPUT_ROOT / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("All outputs:", OUTPUT_ROOT)
        """
    ),
    markdown(
        """
        ## Reporting checklist

        Report the 36/12/12 speaker-disjoint split, 18,000/6,000/6,000 utterance
        counts, per-utterance peak normalization for the two 20D front ends,
        train-only normalization, validation-selected checkpoints, and one-shot
        held-out results. Include macro and worst-speaker accuracy alongside pooled
        accuracy. State that AudioNet/AlexNet are architecture reimplementations and
        that AlexNet MACs exclude FFT preprocessing. Describe BC-ResNet-8 as the
        primary strong digital encoder and BC-ResNet-1 as the small-digital control.
        Report the 120-bin BC-ResNet-8 row as a spectral-resolution ablation, not as
        an independently tuned architecture. For Transformer-CTC, report exact
        decoded-sequence accuracy and token error rate. Explicitly disclose that
        every AudioMNIST target contains one digit, so these two rows are not a
        continuous-sequence recognition experiment.
        State explicitly that both BC-ResNet rows are fixed-utterance, noncausal
        classifiers and therefore are not streaming-latency matches.

        The physical result to compare against is 5,671/6,000 = 94.5167% using
        149,994 parameters and 20 resonator-envelope features per 10 ms step. Claim
        fewer classifier parameters only if the measured accuracy-parameter frontier
        supports it. Otherwise, limit the claim to the measured front-end, MAC,
        latency, or sensing advantage.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
