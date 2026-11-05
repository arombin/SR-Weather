from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SMOOTHING_RADIUS = 5


def _moving_nonzero_mean(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered moving mean that ignores zero-valued pixels."""
    value_sum = np.zeros_like(dataset, dtype=np.result_type(dataset, np.float32))
    valid_count = np.zeros_like(dataset, dtype=np.int16)
    valid = dataset != 0

    for offset in range(-SMOOTHING_RADIUS, SMOOTHING_RADIUS + 1):
        source_start = max(0, -offset)
        source_stop = min(dataset.shape[0], dataset.shape[0] - offset)
        target_start = source_start + offset
        target_stop = source_stop + offset

        value_sum[target_start:target_stop] += dataset[source_start:source_stop]
        valid_count[target_start:target_stop] += valid[source_start:source_stop]

    safe_count = np.where(valid_count == 0, 1, valid_count)
    return value_sum / safe_count, safe_count


def smooth_year_dataset(year_dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the initial 11-day moving mean used to construct the SCM."""
    return _moving_nonzero_mean(year_dataset)


def extended_smooth_year_dataset(
    smoothed_dataset: np.ndarray,
    year_dataset: np.ndarray,
    binary_dataset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend neighboring values into gaps while retaining existing estimates."""
    combined = np.nanmean(
        np.stack(
            [
                np.where(smoothed_dataset == 0, np.nan, smoothed_dataset),
                np.where(year_dataset == 0, np.nan, year_dataset),
            ],
            axis=0,
        ),
        axis=0,
    )
    combined = np.nan_to_num(combined, nan=0.0)
    new_smoothed_dataset, new_binary_dataset = _moving_nonzero_mean(combined)
    result = np.where(binary_dataset == 1, new_smoothed_dataset, smoothed_dataset)
    return np.nan_to_num(result, nan=0.0), new_binary_dataset


def create_scm(
    input_dir: str | Path,
    output_path: str | Path,
    iterations: int = 10,
) -> np.ndarray:
    """Average yearly arrays, fill temporal gaps, and save the resulting SCM."""
    input_dir = Path(input_dir)
    file_paths = sorted(input_dir.glob("*.npy"))
    if not file_paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    first = np.load(file_paths[0])
    value_sum = np.zeros_like(first, dtype=np.result_type(first, np.float32))
    valid_count = np.zeros_like(first, dtype=np.int32)

    for file_path in file_paths:
        yearly_array = np.load(file_path)
        if yearly_array.shape != first.shape:
            raise ValueError(
                f"Shape mismatch: {file_path} has {yearly_array.shape}, "
                f"expected {first.shape}"
            )
        value_sum += yearly_array
        valid_count += yearly_array != 0

    year_dataset = value_sum / np.where(valid_count == 0, 1, valid_count)
    scm, binary_dataset = smooth_year_dataset(year_dataset)

    for _ in range(iterations):
        scm, binary_dataset = extended_smooth_year_dataset(
            scm, year_dataset, binary_dataset
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scm)
    return scm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing yearly .npy files")
    parser.add_argument("output_path", type=Path, help="Output SCM .npy path")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of gap-filling passes (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = create_scm(args.input_dir, args.output_path, args.iterations)
    print(f"Saved SCM with shape {result.shape} to {args.output_path}")
