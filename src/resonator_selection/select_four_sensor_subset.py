"""Select a four-channel subset from the measured COMSOL S1-S8 bank.

The script reconstructs each alphabet segment from the saved measured transfer
functions.  Crucially, feature normalization is repeated after selecting four
channels; discarded sensors therefore cannot leak information through an
eight-channel normalization denominator.
"""

from __future__ import annotations

import csv
import itertools
import re
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
)


ROOT = Path(__file__).resolve().parents[1]
SENSOR_CSV = ROOT / "V4R_frequency_response_S1_S8.csv"
INLET_CSV = ROOT / "V4R_frequency_response_inlet.csv"
SEGMENT_CSV = (
    ROOT
    / "outputs"
    / "alphabet_svm_pattern_recognition"
    / "features"
    / "alphabet_segments_auto_A_to_Z.csv"
)
OUT_DIR = ROOT / "outputs" / "four_sensor_subset_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE_HZ = 16_000
WINDOW_MS = 40.0
HOP_MS = 10.0
N_FFT = 2048
EPS = 1e-30
RANDOM_SEED = 7
SENSOR_NAMES = tuple(f"S{i}" for i in range(1, 9))
TARGET_HZ = np.asarray([110, 180, 300, 1800, 3500, 4500, 6000, 7500], dtype=float)
LOW_SENSORS = frozenset(("S1", "S2", "S3", "S4"))
HIGH_SENSORS = frozenset(("S5", "S6", "S7", "S8"))


def comment_header_and_rows(path: Path, required_header_text: str):
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header_index = next(
        i for i, line in enumerate(lines)
        if line.startswith("%") and required_header_text in line
    )
    columns = [item.strip() for item in next(csv.reader([lines[header_index].lstrip("% ")]))]
    rows = []
    for line in lines[header_index + 1 :]:
        if not line.strip() or line.startswith("%"):
            continue
        values = next(csv.reader([line]))
        if len(values) == len(columns):
            rows.append([float(value) for value in values])
    return columns, np.asarray(rows, dtype=float)


def parse_inlet(path: Path):
    _, values = comment_header_and_rows(path, "freq (Hz)")
    order = np.argsort(values[:, 0])
    return values[order, 0], values[order, 1] + 1j * values[order, 2]


def parse_sensors(path: Path):
    _, values = comment_header_and_rows(path, "freq (Hz)")
    response = np.column_stack(
        [
            values[:, 1 + 2 * index] + 1j * values[:, 2 + 2 * index]
            for index in range(8)
        ]
    )
    order = np.argsort(values[:, 0])
    return values[order, 0], response[order]


def load_audio_mono(path: Path, target_sr: int):
    if path.suffix.lower() == ".wav":
        sample_rate, data = wavfile.read(str(path))
        audio = data.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            audio /= max(1.0, float(np.iinfo(data.dtype).max))
    else:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
        sample_rate = target_sr
    audio -= float(np.mean(audio))
    if int(sample_rate) != int(target_sr):
        divisor = np.gcd(int(sample_rate), int(target_sr))
        audio = resample_poly(
            audio,
            int(target_sr) // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
        sample_rate = target_sr
    return int(sample_rate), audio


def segment_stft(audio: np.ndarray, sample_rate: int):
    window_samples = int(round(WINDOW_MS * sample_rate / 1000.0))
    hop_samples = int(round(HOP_MS * sample_rate / 1000.0))
    if len(audio) < window_samples:
        audio = np.pad(audio, (0, window_samples - len(audio)))
    frequency, _, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=window_samples,
        noverlap=window_samples - hop_samples,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    return frequency, np.abs(spectrum) ** 2


def interpolate_bank(source_frequency, bank, target_frequency):
    return np.column_stack(
        [np.interp(target_frequency, source_frequency, bank[:, index]) for index in range(bank.shape[1])]
    )


def unit_sum(values):
    values = np.asarray(values, dtype=float)
    return values / (values.sum() + EPS)


def feature_families(trajectory: np.ndarray):
    probability = trajectory / (trajectory.sum(axis=0, keepdims=True) + EPS)
    profile = unit_sum(trajectory.sum(axis=1))
    delta = np.diff(probability, axis=1)
    abs_delta = np.mean(np.abs(delta), axis=1) if delta.size else np.zeros(len(profile))
    chunks = np.array_split(np.arange(trajectory.shape[1]), 3)
    bins = [unit_sum(trajectory[:, chunk].sum(axis=1)) for chunk in chunks]
    integrated = profile
    integrated_delta = np.concatenate([profile, abs_delta])
    three_bins = np.concatenate(bins)
    full = np.concatenate([profile, abs_delta, *bins])
    return {
        "integrated": integrated,
        "integrated_delta": integrated_delta,
        "three_bins": three_bins,
        "full": full,
    }


def select_anova(X_train, y_train, X_test, requested):
    variance_mask = np.var(X_train, axis=0) > 1e-12
    if not np.any(variance_mask):
        variance_mask[:] = True
    X_train = X_train[:, variance_mask]
    X_test = X_test[:, variance_mask]
    if requested == "all" or int(requested) >= X_train.shape[1]:
        return X_train, X_test
    scores, _ = f_classif(X_train, y_train)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    selected = np.argsort(scores)[::-1][: int(requested)]
    return X_train[:, selected], X_test[:, selected]


def evaluate_subset(records, subset, family, requested, classifier_template):
    X = np.vstack([record["features"][subset][family] for record in records])
    y = np.asarray([record["letter"] for record in records])
    groups = np.asarray([record["file_index"] for record in records])
    logo = LeaveOneGroupOut()
    true_all, predicted_all, fold_scores = [], [], []
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = select_anova(X[train_index], y[train_index], X[test_index], requested)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        classifier = clone(classifier_template)
        classifier.fit(X_train, y[train_index])
        predicted = classifier.predict(X_test)
        true_all.extend(y[test_index])
        predicted_all.extend(predicted)
        # Every held-out recording contains one example of each A-Z class, so
        # fold accuracy is exactly fold balanced accuracy without triggering
        # sklearn's small-sample target-type warning.
        fold_scores.append(accuracy_score(y[test_index], predicted))
    return {
        "accuracy": accuracy_score(true_all, predicted_all),
        "balanced_accuracy": balanced_accuracy_score(true_all, predicted_all),
        "macro_f1": f1_score(true_all, predicted_all, average="macro"),
        "fold_balanced_accuracy_mean": float(np.mean(fold_scores)),
        "fold_balanced_accuracy_std": float(np.std(fold_scores)),
        "fold_balanced_accuracy_min": float(np.min(fold_scores)),
        "n": len(true_all),
        "folds": len(fold_scores),
        "raw_feature_dimension": X.shape[1],
    }


def main():
    inlet_frequency, inlet_pressure = parse_inlet(INLET_CSV)
    sensor_frequency, sensor_velocity = parse_sensors(SENSOR_CSV)
    if not np.array_equal(inlet_frequency, sensor_frequency):
        raise ValueError("Inlet and sensor frequency grids differ.")
    inlet_safe = np.where(np.abs(inlet_pressure) > EPS, inlet_pressure, 1.0 + 0.0j)
    transfer_energy = np.abs(sensor_velocity / inlet_safe[:, None]) ** 2
    transfer_calibrated = transfer_energy / (transfer_energy.max(axis=0, keepdims=True) + EPS)

    raw_segments = pd.read_csv(SEGMENT_CSV)
    segments = pd.DataFrame(
        {
            "file_index": pd.to_numeric(raw_segments["take_index"], errors="raise").astype(int),
            "letter": raw_segments["letter"].astype(str).str.upper(),
            "start_s": pd.to_numeric(raw_segments["start_s"], errors="raise"),
            "end_s": pd.to_numeric(raw_segments["end_s"], errors="raise"),
            "audio_path": raw_segments["audio_path"].astype(str),
        }
    )
    segments = segments[segments["letter"].isin(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))]
    segments = segments.sort_values(["file_index", "letter"]).reset_index(drop=True)

    subsets = list(itertools.combinations(range(8), 4))
    audio_cache = {}
    records = []
    for row_index, row in segments.iterrows():
        path = Path(row["audio_path"])
        if str(path) not in audio_cache:
            audio_cache[str(path)] = load_audio_mono(path, SAMPLE_RATE_HZ)
        sample_rate, audio = audio_cache[str(path)]
        start = max(0, int(round(float(row["start_s"]) * sample_rate)))
        end = min(len(audio), int(round(float(row["end_s"]) * sample_rate)))
        segment = audio[start:end].copy()
        segment *= 0.1 / np.sqrt(np.mean(segment**2) + EPS)
        frequency, power = segment_stft(segment, sample_rate)
        mask = (frequency >= 80.0) & (frequency <= 8000.0)
        bank = interpolate_bank(inlet_frequency, transfer_calibrated, frequency[mask])
        trajectory8 = bank.T @ power[mask]
        per_subset = {
            subset: feature_families(trajectory8[np.asarray(subset)])
            for subset in subsets
        }
        records.append(
            {
                "file_index": int(row["file_index"]),
                "letter": str(row["letter"]),
                "features": per_subset,
            }
        )
        if row_index % 25 == 0:
            print(f"rebuilt {row_index + 1}/{len(segments)} letter trajectories")

    classifiers = {
        "linear_svm": LinearSVC(
            C=0.3,
            class_weight="balanced",
            max_iter=30_000,
            random_state=RANDOM_SEED,
        ),
    }
    configurations = (
        ("integrated_delta", "all"),
        ("three_bins", "all"),
        ("full", 16),
        ("full", "all"),
    )
    rows = []
    for subset_index, subset in enumerate(subsets):
        names = tuple(SENSOR_NAMES[index] for index in subset)
        low_count = sum(name in LOW_SENSORS for name in names)
        high_count = sum(name in HIGH_SENSORS for name in names)
        for family, requested in configurations:
            dimension = len(records[0]["features"][subset][family])
            if requested != "all" and int(requested) > dimension:
                continue
            for classifier_name, classifier in classifiers.items():
                result = evaluate_subset(records, subset, family, requested, classifier)
                rows.append(
                    {
                        "subset": ";".join(names),
                        "low_count": low_count,
                        "high_count": high_count,
                        "balanced_2low_2high": low_count == 2 and high_count == 2,
                        "target_frequencies_hz": ";".join(str(int(TARGET_HZ[i])) for i in subset),
                        "feature_family": family,
                        "requested_features": str(requested),
                        "classifier": classifier_name,
                        **result,
                    }
                )
        print(f"evaluated subset {subset_index + 1}/{len(subsets)}: {';'.join(names)}")

    results = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "fold_balanced_accuracy_std"],
        ascending=[False, False, True],
    )
    results.to_csv(OUT_DIR / "four_sensor_subset_sweep.csv", index=False)
    best_by_subset = (
        results.sort_values(
            ["balanced_accuracy", "macro_f1", "fold_balanced_accuracy_std"],
            ascending=[False, False, True],
        )
        .groupby("subset", as_index=False)
        .first()
        .sort_values(
            ["balanced_accuracy", "macro_f1", "fold_balanced_accuracy_std"],
            ascending=[False, False, True],
        )
    )
    best_by_subset.to_csv(OUT_DIR / "best_result_by_four_sensor_subset.csv", index=False)
    balanced = best_by_subset[best_by_subset["balanced_2low_2high"]].copy()
    balanced.to_csv(OUT_DIR / "ranked_balanced_2low_2high_subsets.csv", index=False)

    top = balanced.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["subset"], top["balanced_accuracy"], color="#167d72")
    ax.set(
        xlabel="leave-one-recording-out balanced accuracy",
        ylabel="four-sensor subset",
        title="Best 2-low / 2-high subsets on 182 A-Z segments",
        xlim=(0, 1),
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "balanced_four_sensor_subset_ranking.png", dpi=180)
    plt.close(fig)

    best_balanced = balanced.iloc[0]
    best_unconstrained = best_by_subset.iloc[0]
    summary = pd.DataFrame([best_balanced, best_unconstrained])
    summary.insert(0, "selection_type", ["best_balanced", "best_unconstrained"])
    summary.to_csv(OUT_DIR / "selection_summary.csv", index=False)
    print("\nBEST BALANCED 2-LOW / 2-HIGH")
    print(best_balanced.to_string())
    print("\nBEST UNCONSTRAINED")
    print(best_unconstrained.to_string())


if __name__ == "__main__":
    main()
