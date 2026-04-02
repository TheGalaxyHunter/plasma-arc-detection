"""PyTorch Dataset for plasma arc sensor data.

Loads multi-channel time-series recordings (voltage, current, spectral bands)
and provides windowed sequences for arc detection training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import SignalProcessor

logger = logging.getLogger(__name__)


class PlasmaArcDataset(Dataset):
    """Dataset for plasma arc detection from multi-channel sensor recordings.

    Each sample is a fixed-length window of sensor readings across multiple
    channels (voltage, current, and spectral intensity bands). Labels indicate
    whether an arc event is present in the window, along with an optional
    severity score (0.0 to 1.0).

    Expected data format per file (.npy):
        signals: shape (num_timesteps, num_channels)
        labels:  shape (num_timesteps,) with 0 = no arc, 1 = arc
        severity: shape (num_timesteps,) with float values in [0, 1]

    Args:
        root_dir: Path to directory containing processed .npy files.
        sequence_length: Number of timesteps per sample window.
        num_channels: Number of sensor channels expected.
        normalize: Whether to apply per-channel z-score normalization.
        processor: Optional SignalProcessor for on-the-fly feature extraction.
        transform: Optional callable for data augmentation.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 256,
        num_channels: int = 6,
        normalize: bool = True,
        processor: Optional[SignalProcessor] = None,
        transform: Optional[callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.normalize = normalize
        self.processor = processor or SignalProcessor(num_channels=num_channels)
        self.transform = transform

        self.signals: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.severity: list[np.ndarray] = []
        self._windows: list[tuple[int, int]] = []  # (file_idx, start_idx)

        self._load_data()

    def _load_data(self) -> None:
        """Load all .npy files and compute valid window positions."""
        data_files = sorted(self.root_dir.glob("*.npy"))
        if not data_files:
            logger.warning(f"No .npy files found in {self.root_dir}")
            return

        for file_idx, fpath in enumerate(data_files):
            raw = np.load(fpath, allow_pickle=True).item()
            signals = raw["signals"].astype(np.float32)
            labels = raw["labels"].astype(np.int64)
            severity = raw.get("severity", np.zeros(len(labels), dtype=np.float32))

            if signals.shape[1] != self.num_channels:
                raise ValueError(
                    f"Expected {self.num_channels} channels, "
                    f"got {signals.shape[1]} in {fpath.name}"
                )

            if self.normalize:
                signals = self.processor.normalize(signals)

            self.signals.append(signals)
            self.labels.append(labels)
            self.severity.append(severity.astype(np.float32))

            num_windows = max(0, len(signals) - self.sequence_length + 1)
            for start in range(0, num_windows, self.sequence_length // 2):
                self._windows.append((file_idx, start))

        logger.info(
            f"Loaded {len(data_files)} files, {len(self._windows)} windows "
            f"(seq_len={self.sequence_length})"
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, start = self._windows[idx]
        end = start + self.sequence_length

        x = self.signals[file_idx][start:end]  # (seq_len, channels)
        y_label = self.labels[file_idx][start:end]
        y_severity = self.severity[file_idx][start:end]

        # Window-level label: arc present if any timestep contains an arc
        arc_present = int(y_label.max() > 0)
        # Window-level severity: max severity in the window
        max_severity = float(y_severity.max())

        if self.transform is not None:
            x = self.transform(x)

        # Transpose to (channels, seq_len) for 1D conv layers
        x_tensor = torch.from_numpy(x).float().permute(1, 0)

        return {
            "input": x_tensor,
            "label": torch.tensor(arc_present, dtype=torch.long),
            "severity": torch.tensor(max_severity, dtype=torch.float32),
            "timestep_labels": torch.from_numpy(y_label),
        }


def build_dataloaders(
    root_dir: str | Path,
    sequence_length: int = 256,
    num_channels: int = 6,
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, ...]:
    """Build train/val/test DataLoaders with stratified splitting.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    full_dataset = PlasmaArcDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        num_channels=num_channels,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader
