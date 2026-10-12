"""Convert the raw BIDSleep files into one row per 30-second sleep epoch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


DEFAULT_DATASET_PATH = Path(
    "SleepData/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
)


def process_night(subject_path: Path, night: str) -> pd.DataFrame:
    night_path = subject_path / str(night)
    hr_data = pd.read_csv(night_path / "hr.csv", header=None, names=["Timestamp", "HR"])
    motion_data = pd.read_csv(
        night_path / "motion.csv", usecols=["Timestamp", "x", "y", "z"]
    )
    labels = loadmat(night_path / "labels.mat")

    rec_start = pd.Timestamp(str(labels["recStart"][0])).tz_localize("US/Eastern").timestamp()
    expert_labels = labels["expert_label"].reshape(-1)
    num_epochs = len(expert_labels)

    hr_data["epoch"] = ((hr_data["Timestamp"] - rec_start) // 30).astype(int)
    hr_data = hr_data[hr_data["epoch"].between(0, num_epochs - 1)]
    hr_by_epoch = hr_data.groupby("epoch").agg(
        HR=("HR", "mean"),
        hr_std=("HR", "std"),
        hr_min=("HR", "min"),
        hr_max=("HR", "max"),
        hr_count=("HR", "count"),
    )
    hr_by_epoch["hr_range"] = hr_by_epoch["hr_max"] - hr_by_epoch["hr_min"]

    # Combine the x, y, and z acceleration axes into one magnitude. A stationary
    # watch has a magnitude close to 1 because gravity is measured as 1 g.
    motion_data["magnitude"] = np.sqrt(
        motion_data["x"] ** 2 + motion_data["y"] ** 2 + motion_data["z"] ** 2
    )

    # ENMO (Euclidean Norm Minus One) removes the expected 1 g gravity component.
    # Negative values are clipped to zero so ENMO represents movement intensity.
    motion_data["enmo"] = (motion_data["magnitude"] - 1.0).clip(lower=0.0)
    motion_data["epoch"] = ((motion_data["Timestamp"] - rec_start) // 30).astype(int)
    motion_data = motion_data[motion_data["epoch"].between(0, num_epochs - 1)]

    motion_by_epoch = motion_data.groupby("epoch").agg(
        # Keep the original features so old and new experiments remain comparable.
        movement_mean=("magnitude", "mean"),
        movement_std=("magnitude", "std"),
        # Add gravity-adjusted movement summaries for every 30-second epoch.
        enmo_mean=("enmo", "mean"),
        enmo_std=("enmo", "std"),
        enmo_min=("enmo", "min"),
        enmo_max=("enmo", "max"),
    )
    motion_by_epoch["enmo_range"] = (
        motion_by_epoch["enmo_max"] - motion_by_epoch["enmo_min"]
    )

    night_data = pd.DataFrame(
        {"epoch": np.arange(num_epochs), "sleep_stage": expert_labels}
    )
    night_data = night_data.merge(
        hr_by_epoch, left_on="epoch", right_index=True, how="left"
    )
    # The measurement count also tells the model when HR coverage was sparse.
    night_data["hr_count"] = night_data["hr_count"].fillna(0).astype(int)
    # Use only the previous epoch so this feature is suitable for real-time use.
    night_data["hr_change_prev"] = night_data["HR"].diff()
    night_data = night_data.merge(
        motion_by_epoch, left_on="epoch", right_index=True, how="left"
    )
    night_data["subject"] = subject_path.name
    night_data["night"] = night
    return night_data


def night_sort_key(path: Path) -> tuple[int, float | str]:
    try:
        return (0, float(path.name))
    except ValueError:
        return (1, path.name)


def build_dataset(dataset_path: Path) -> pd.DataFrame:
    all_data = []
    subject_paths = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    for subject_path in subject_paths:
        night_paths = sorted(
            (path for path in subject_path.iterdir() if path.is_dir()),
            key=night_sort_key,
        )
        for night_path in night_paths:
            night_data = process_night(subject_path, night_path.name)
            all_data.append(night_data)
            print(
                f"Processed {subject_path.name} night {night_path.name}: "
                f"{len(night_data)} epochs"
            )

    if not all_data:
        raise ValueError(f"No subject/night data found under {dataset_path}")
    return pd.concat(all_data, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=Path("processed_sleep_data.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_data = build_dataset(args.dataset_path)
    all_data.to_csv(args.output, index=False)

    print("\nFinal dataset shape:", all_data.shape)
    print("Number of subjects:", all_data["subject"].nunique())
    print("Number of nights:", all_data.groupby(["subject", "night"]).ngroups)
    print("\nMissing values:")
    print(all_data.isna().sum())
    print("\nSleep stage counts:")
    print(all_data["sleep_stage"].value_counts().sort_index())
    print(f"\nSaved enhanced dataset to {args.output.resolve()}")


if __name__ == "__main__":
    main()
