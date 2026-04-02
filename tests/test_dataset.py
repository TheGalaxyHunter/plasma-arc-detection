"""Tests for PlasmaArcDataset and data pipeline components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import PlasmaArcDataset
from src.data.preprocessing import SignalProcessor


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic plasma arc data."""
    num_timesteps = 2048
    num_channels = 6

    rng = np.random.default_rng(42)
    signals = rng.standard_normal((num_timesteps, num_channels)).astype(np.float32)
    labels = np.zeros(num_timesteps, dtype=np.int64)

    # Simulate an arc event in the middle of the recording
    arc_start, arc_end = 800, 1000
    labels[arc_start:arc_end] = 1
    signals[arc_start:arc_end, 0] *= 3.0  # Voltage spike during arc

    severity = np.zeros(num_timesteps, dtype=np.float32)
    severity[arc_start:arc_end] = np.linspace(0.3, 0.9, arc_end - arc_start)

    data = {"signals": signals, "labels": labels, "severity": severity}
    np.save(tmp_path / "recording_001.npy", data)

    return tmp_path


class TestPlasmaArcDataset:
    """Tests for the PlasmaArcDataset class."""

    def test_dataset_loads(self, sample_data_dir: Path) -> None:
        ds = PlasmaArcDataset(root_dir=sample_data_dir, sequence_length=256)
        assert len(ds) > 0

    def test_sample_shape(self, sample_data_dir: Path) -> None:
        ds = PlasmaArcDataset(
            root_dir=sample_data_dir,
            sequence_length=256,
            num_channels=6,
        )
        sample = ds[0]
        assert sample["input"].shape == (6, 256)
        assert sample["label"].dtype == torch.long
        assert sample["severity"].dtype == torch.float32

    def test_label_values(self, sample_data_dir: Path) -> None:
        ds = PlasmaArcDataset(root_dir=sample_data_dir, sequence_length=256)
        labels = set()
        for i in range(len(ds)):
            labels.add(ds[i]["label"].item())
        assert labels.issubset({0, 1})

    def test_channel_mismatch_raises(self, sample_data_dir: Path) -> None:
        with pytest.raises(ValueError, match="Expected 3 channels"):
            PlasmaArcDataset(
                root_dir=sample_data_dir,
                sequence_length=256,
                num_channels=3,
            )

    def test_empty_directory(self, tmp_path: Path) -> None:
        ds = PlasmaArcDataset(root_dir=tmp_path, sequence_length=256)
        assert len(ds) == 0


class TestSignalProcessor:
    """Tests for the SignalProcessor class."""

    def test_normalize_zero_mean_unit_var(self) -> None:
        proc = SignalProcessor(num_channels=4)
        signals = np.random.randn(1000, 4).astype(np.float32) * 10 + 5
        normalized = proc.normalize(signals)

        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-5)

    def test_bandpass_filter_shape(self) -> None:
        proc = SignalProcessor(num_channels=6)
        signals = np.random.randn(2048, 6).astype(np.float32)
        filtered = proc.bandpass_filter(signals)

        assert filtered.shape == signals.shape

    def test_spectral_features_shape(self) -> None:
        proc = SignalProcessor(num_channels=6)
        signals = np.random.randn(1024, 6).astype(np.float32)
        features = proc.extract_spectral_features(signals, window_size=256)

        expected_windows = 1024 // 256
        expected_features = 6 * 4  # 4 features per channel
        assert features.shape == (expected_windows, expected_features)

    def test_rms_envelope_shape(self) -> None:
        proc = SignalProcessor(num_channels=6)
        signals = np.random.randn(512, 6).astype(np.float32)
        envelope = proc.compute_rms_envelope(signals, window_size=64)

        assert envelope.shape == signals.shape

    def test_detect_transients(self) -> None:
        proc = SignalProcessor(num_channels=2)
        signals = np.zeros((100, 2), dtype=np.float32)
        signals[50, 0] = 100.0  # Large spike

        transients = proc.detect_transients(signals, threshold_sigma=3.0)
        assert transients[50, 0], "Spike at t=50 should be detected as transient"

    def test_fit_transform(self) -> None:
        proc = SignalProcessor(num_channels=4)
        train_data = np.random.randn(5000, 4).astype(np.float32) * 3 + 2

        proc.fit_normalization(train_data)
        transformed = proc.transform(train_data)

        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=0.05)

    def test_transform_without_fit_raises(self) -> None:
        proc = SignalProcessor(num_channels=4)
        signals = np.random.randn(100, 4).astype(np.float32)

        with pytest.raises(RuntimeError, match="Call fit_normalization"):
            proc.transform(signals)
