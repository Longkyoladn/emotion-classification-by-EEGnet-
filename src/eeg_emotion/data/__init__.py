"""Data indexing and leakage-safe split utilities."""

from .dataset import DataLoaderBundle, EEGWindowDataset, make_partition_dataloaders
from .modeling_index import build_modeling_index
from .normalization import ChannelStandardizer, fit_channel_standardizer
from .splits import (
    build_nested_loso_splits,
    build_personalization_splits,
    build_sensitivity_split,
    build_subject_dependent_splits,
)

__all__ = [
    "ChannelStandardizer",
    "DataLoaderBundle",
    "EEGWindowDataset",
    "build_modeling_index",
    "build_nested_loso_splits",
    "build_personalization_splits",
    "build_sensitivity_split",
    "build_subject_dependent_splits",
    "fit_channel_standardizer",
    "make_partition_dataloaders",
]
