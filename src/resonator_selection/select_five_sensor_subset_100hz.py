"""Select five COMSOL-derived resonator channels at a 100 Hz readout rate.

This is the five-channel counterpart of ``select_four_sensor_subset.py``.  It
uses the measured complex COMSOL transfer functions, a 10 ms trajectory hop,
and leave-one-recording-out evaluation over the seven A-Z recordings.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from select_four_sensor_subset import (
    EPS,
    INLET_CSV,
    LOW_SENSORS,
    RANDOM_SEED,
    SAMPLE_RATE_HZ,
    SEGMENT_CSV,
    SENSOR_CSV,
    SENSOR_NAMES,
    TARGET_HZ,
    evaluate_subset,
    feature_families,
    interpolate_bank,
    load_audio_mono,
    parse_inlet,
    parse_sensors,
    segment_stft,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "five_sensor_subset_selection_100hz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
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

    subsets = list(itertools.combinations(range(8), 5))
    audio_cache: dict[str, tuple[int, np.ndarray]] = {}
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
        records.append(
            {
                "file_index": int(row["file_index"]),
                "letter": str(row["letter"]),
                "trajectory8": trajectory8,
                "features": {
                    subset: feature_families(trajectory8[np.asarray(subset)])
                    for subset in subsets
                },
            }
        )
        if row_index % 25 == 0:
            print(f"rebuilt {row_index + 1}/{len(segments)} letter trajectories")

    classifier = LinearSVC(
        C=0.3,
        class_weight="balanced",
        max_iter=30_000,
        random_state=RANDOM_SEED,
    )
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
        high_count = len(names) - low_count
        for family, requested in configurations:
            result = evaluate_subset(records, subset, family, requested, classifier)
            rows.append(
                {
                    "subset": ";".join(names),
                    "low_count": low_count,
                    "high_count": high_count,
                    "target_frequencies_hz": ";".join(str(int(TARGET_HZ[i])) for i in subset),
                    "feature_family": family,
                    "requested_features": str(requested),
                    "classifier": "linear_svm",
                    **result,
                }
            )
        print(f"evaluated subset {subset_index + 1}/{len(subsets)}: {';'.join(names)}")

    results = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "fold_balanced_accuracy_std"],
        ascending=[False, False, True],
    )
    results.to_csv(OUT_DIR / "five_sensor_subset_sweep.csv", index=False)
    best_by_subset = (
        results.groupby("subset", as_index=False)
        .first()
        .sort_values(
            ["balanced_accuracy", "macro_f1", "fold_balanced_accuracy_std"],
            ascending=[False, False, True],
        )
    )
    best_by_subset.to_csv(OUT_DIR / "best_result_by_five_sensor_subset.csv", index=False)

    top = best_by_subset.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["subset"], top["balanced_accuracy"], color="#167d72")
    ax.set(
        xlabel="leave-one-recording-out balanced accuracy",
        ylabel="five-sensor subset",
        title="Five COMSOL Resonances at a 100 Hz Readout Rate",
        xlim=(0, 1),
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "five_sensor_subset_ranking.png", dpi=180)
    plt.close(fig)

    best = best_by_subset.iloc[[0]].copy()
    best.insert(0, "selection_type", "best_five_sensor_100hz")
    best.to_csv(OUT_DIR / "selection_summary.csv", index=False)

    best_names = str(best.iloc[0]["subset"]).split(";")
    best_indices = tuple(SENSOR_NAMES.index(name) for name in best_names)
    candidate_subsets = [
        subset
        for size in range(1, len(best_indices) + 1)
        for subset in itertools.combinations(best_indices, size)
    ]
    for record in records:
        for subset in candidate_subsets:
            if subset not in record["features"]:
                record["features"][subset] = feature_families(
                    record["trajectory8"][np.asarray(subset)]
                )

    singleton_rows = []
    for sensor_index in best_indices:
        subset = (sensor_index,)
        best_single = None
        for family, requested in configurations:
            dimension = len(records[0]["features"][subset][family])
            if requested != "all" and int(requested) > dimension:
                continue
            result = evaluate_subset(records, subset, family, requested, classifier)
            row = {
                "sensor": SENSOR_NAMES[sensor_index],
                "frequency_hz": int(TARGET_HZ[sensor_index]),
                "feature_family": family,
                "requested_features": str(requested),
                **result,
            }
            if best_single is None or (
                row["balanced_accuracy"], row["macro_f1"], -row["fold_balanced_accuracy_std"]
            ) > (
                best_single["balanced_accuracy"],
                best_single["macro_f1"],
                -best_single["fold_balanced_accuracy_std"],
            ):
                best_single = row
        singleton_rows.append(best_single)
    pd.DataFrame(singleton_rows).sort_values(
        ["balanced_accuracy", "macro_f1"], ascending=False
    ).to_csv(OUT_DIR / "single_resonator_results.csv", index=False)

    current: tuple[int, ...] = ()
    build_rows = []
    for step in range(1, len(best_indices) + 1):
        candidates = []
        for added in best_indices:
            if added in current:
                continue
            subset = tuple(sorted((*current, added)))
            for family, requested in configurations:
                dimension = len(records[0]["features"][subset][family])
                if requested != "all" and int(requested) > dimension:
                    continue
                result = evaluate_subset(records, subset, family, requested, classifier)
                candidates.append(
                    {
                        "build_step": step,
                        "added_sensor": SENSOR_NAMES[added],
                        "added_frequency_hz": int(TARGET_HZ[added]),
                        "subset": ";".join(SENSOR_NAMES[index] for index in subset),
                        "frequencies_hz": ";".join(str(int(TARGET_HZ[index])) for index in subset),
                        "feature_family": family,
                        "requested_features": str(requested),
                        **result,
                        "added_index": added,
                    }
                )
        winner = sorted(
            candidates,
            key=lambda row: (
                row["balanced_accuracy"],
                row["macro_f1"],
                -row["fold_balanced_accuracy_std"],
            ),
            reverse=True,
        )[0]
        current = tuple(sorted((*current, int(winner.pop("added_index")))))
        build_rows.append(winner)
    pd.DataFrame(build_rows).to_csv(OUT_DIR / "one_at_a_time_build_order.csv", index=False)

    print("\nBEST FIVE-SENSOR SUBSET AT 100 HZ")
    print(best.iloc[0].to_string())
    print("\nONE-AT-A-TIME BUILD ORDER WITHIN THE WINNING SET")
    print(pd.DataFrame(build_rows).to_string(index=False))


if __name__ == "__main__":
    main()
