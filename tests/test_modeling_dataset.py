from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_emotion.data.dataset import EEGWindowDataset, make_partition_dataloaders


def _write_test_fif(path: Path) -> None:
    data = np.vstack(
        [
            np.concatenate([np.arange(10), np.arange(100, 110)]),
            np.concatenate([np.arange(10, 20), np.arange(200, 210)]),
        ]
    ).astype(float)
    info = mne.create_info(["C1", "C2"], sfreq=10.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.save(path, overwrite=True, verbose="ERROR")


def _index(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s1-t1-w1",
                "subject": "s1",
                "trial": 1,
                "window_index": 1,
                "start_sample": 0,
                "stop_sample": 10,
                "fif_path": path.as_posix(),
                "split_group": "s1-trial-1",
                "sample_weight": 1.0,
            },
            {
                "sample_id": "s1-t2-w1",
                "subject": "s1",
                "trial": 2,
                "window_index": 1,
                "start_sample": 10,
                "stop_sample": 20,
                "fif_path": path.as_posix(),
                "split_group": "s1-trial-2",
                "sample_weight": 1.0,
            },
        ]
    )


def test_dataset_reads_cleaned_fif_window() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = Path(folder) / "trial_eeg.fif"
        _write_test_fif(path)
        table = _index(path)
        table["label"] = ["low", "high"]
        dataset = EEGWindowDataset(table, project_root=Path(folder))
        sample = dataset[0]

        assert tuple(sample["eeg"].shape) == (2, 10)
        assert sample["label"].item() == 0
        assert sample["sample_weight"].item() == 1.0


def test_dataloader_fits_normalization_on_training_partition_only() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = Path(folder) / "trial_eeg.fif"
        _write_test_fif(path)
        index = _index(path)
        assignments = pd.DataFrame(
            {
                "sample_id": ["s1-t1-w1", "s1-t2-w1"],
                "partition": ["train", "validation"],
                "label": ["low", "high"],
            }
        )
        bundle = make_partition_dataloaders(
            index,
            assignments,
            project_root=Path(folder),
            train_partition="train",
            batch_size=1,
        )

        assert np.allclose(bundle.normalizer.mean, [4.5, 14.5])
        train_batch = next(iter(bundle.loaders["train"]))["eeg"].numpy()
        validation_batch = next(iter(bundle.loaders["validation"]))["eeg"].numpy()
        assert np.allclose(train_batch.mean(axis=2), 0.0, atol=1e-6)
        assert not np.allclose(validation_batch.mean(axis=2), 0.0, atol=1e-3)
        assert (
            bundle.datasets["train"].normalizer
            is bundle.datasets["validation"].normalizer
        )
