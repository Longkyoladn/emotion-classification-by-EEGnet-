"""Training-partition-only channel normalization for EEG windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class RawWindowDataset(Protocol):
    normalizer: ChannelStandardizer | None

    def __len__(self) -> int: ...

    def load_raw_window(self, index: int) -> np.ndarray: ...


@dataclass(frozen=True)
class ChannelStandardizer:
    """Per-channel z-score parameters fitted on one training partition."""

    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    fit_partition: str = "train"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("mean and std must be same-length one-dimensional arrays")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Normalization statistics must be finite")
        if (std <= 0).any() or self.sample_count <= 0:
            raise ValueError("std and sample_count must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def channels(self) -> int:
        return len(self.mean)

    def transform(self, window: np.ndarray) -> np.ndarray:
        data = np.asarray(window, dtype=np.float32)
        if data.ndim != 2 or data.shape[0] != self.channels:
            raise ValueError(
                f"Expected ({self.channels}, time) window, got {data.shape}"
            )
        return ((data - self.mean[:, None]) / self.std[:, None]).astype(
            np.float32, copy=False
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "sample_count": self.sample_count,
            "fit_partition": self.fit_partition,
            "epsilon": self.epsilon,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChannelStandardizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            sample_count=int(payload["sample_count"]),
            fit_partition=str(payload["fit_partition"]),
            epsilon=float(payload["epsilon"]),
        )


def fit_channel_standardizer(
    dataset: RawWindowDataset,
    fit_partition: str = "train",
    epsilon: float = 1e-8,
) -> ChannelStandardizer:
    """Stream mean/std from raw windows belonging only to the training dataset."""
    if dataset.normalizer is not None:
        raise ValueError("Fit normalization from an unnormalized training dataset")
    if len(dataset) == 0:
        raise ValueError("Cannot fit normalization on an empty training dataset")

    channel_sum: np.ndarray | None = None
    channel_sum_squares: np.ndarray | None = None
    sample_count = 0
    for index in range(len(dataset)):
        window = np.asarray(dataset.load_raw_window(index), dtype=np.float64)
        if window.ndim != 2 or not np.isfinite(window).all():
            raise ValueError("Training windows must be finite channel-by-time arrays")
        if channel_sum is None:
            channel_sum = np.zeros(window.shape[0], dtype=np.float64)
            channel_sum_squares = np.zeros(window.shape[0], dtype=np.float64)
        if window.shape[0] != len(channel_sum):
            raise ValueError("All training windows must use the same channel count")
        channel_sum += window.sum(axis=1)
        channel_sum_squares += np.square(window).sum(axis=1)
        sample_count += window.shape[1]

    if channel_sum is None or channel_sum_squares is None:
        raise RuntimeError("No training statistics were accumulated")
    mean = channel_sum / sample_count
    variance = np.maximum(channel_sum_squares / sample_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), epsilon)
    return ChannelStandardizer(
        mean=mean,
        std=std,
        sample_count=sample_count,
        fit_partition=fit_partition,
        epsilon=epsilon,
    )

