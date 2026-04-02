"""Signal processing and feature extraction for plasma arc sensor data.

Provides spectral analysis, filtering, and normalization utilities
tailored to high-frequency electrical discharge signals.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq


class SignalProcessor:
    """Multi-channel signal processor for plasma arc sensor data.

    Handles voltage/current waveforms and spectral intensity channels.
    Supports bandpass filtering, spectral feature extraction, and
    per-channel normalization.

    Args:
        num_channels: Number of input sensor channels.
        sample_rate: Sampling rate in Hz.
        bandpass_low: Lower cutoff frequency for bandpass filter (Hz).
        bandpass_high: Upper cutoff frequency for bandpass filter (Hz).
        filter_order: Order of the Butterworth bandpass filter.
    """

    def __init__(
        self,
        num_channels: int = 6,
        sample_rate: float = 10_000.0,
        bandpass_low: float = 50.0,
        bandpass_high: float = 4_500.0,
        filter_order: int = 4,
    ) -> None:
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.filter_order = filter_order

        self._channel_means: np.ndarray | None = None
        self._channel_stds: np.ndarray | None = None

        nyquist = sample_rate / 2.0
        low = bandpass_low / nyquist
        high = bandpass_high / nyquist
        self._sos = scipy_signal.butter(
            filter_order, [low, high], btype="band", output="sos"
        )

    def normalize(self, signals: np.ndarray) -> np.ndarray:
        """Apply per-channel z-score normalization.

        Args:
            signals: Array of shape (num_timesteps, num_channels).

        Returns:
            Normalized signals with zero mean and unit variance per channel.
        """
        mean = signals.mean(axis=0, keepdims=True)
        std = signals.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (signals - mean) / std

    def fit_normalization(self, signals: np.ndarray) -> None:
        """Compute and store global normalization statistics.

        Args:
            signals: Array of shape (num_timesteps, num_channels).
        """
        self._channel_means = signals.mean(axis=0)
        self._channel_stds = signals.std(axis=0)
        self._channel_stds = np.where(
            self._channel_stds < 1e-8, 1.0, self._channel_stds
        )

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Apply stored normalization statistics.

        Args:
            signals: Array of shape (num_timesteps, num_channels).

        Returns:
            Normalized array using previously fitted statistics.

        Raises:
            RuntimeError: If fit_normalization has not been called.
        """
        if self._channel_means is None or self._channel_stds is None:
            raise RuntimeError(
                "Call fit_normalization() before transform(). "
                "Use normalize() for per-sample normalization instead."
            )
        return (signals - self._channel_means) / self._channel_stds

    def bandpass_filter(self, signals: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter to each channel.

        Args:
            signals: Array of shape (num_timesteps, num_channels).

        Returns:
            Bandpass-filtered signals with same shape.
        """
        filtered = np.zeros_like(signals)
        for ch in range(signals.shape[1]):
            filtered[:, ch] = scipy_signal.sosfiltfilt(self._sos, signals[:, ch])
        return filtered

    def extract_spectral_features(
        self, signals: np.ndarray, window_size: int = 256
    ) -> np.ndarray:
        """Extract frequency-domain features from sliding windows.

        Computes the magnitude spectrum for each channel over non-overlapping
        windows, returning dominant frequency, spectral centroid, bandwidth,
        and spectral energy.

        Args:
            signals: Array of shape (num_timesteps, num_channels).
            window_size: Number of samples per FFT window.

        Returns:
            Feature array of shape (num_windows, num_channels * 4).
        """
        num_timesteps, num_channels = signals.shape
        num_windows = num_timesteps // window_size
        features = np.zeros((num_windows, num_channels * 4), dtype=np.float32)
        freqs = rfftfreq(window_size, d=1.0 / self.sample_rate)

        for w in range(num_windows):
            segment = signals[w * window_size : (w + 1) * window_size]
            for ch in range(num_channels):
                magnitude = np.abs(rfft(segment[:, ch]))
                col_offset = ch * 4

                # Dominant frequency
                features[w, col_offset] = freqs[np.argmax(magnitude)]

                # Spectral centroid
                total_mag = magnitude.sum()
                if total_mag > 1e-10:
                    features[w, col_offset + 1] = (freqs * magnitude).sum() / total_mag
                else:
                    features[w, col_offset + 1] = 0.0

                # Spectral bandwidth (weighted std of frequencies)
                centroid = features[w, col_offset + 1]
                if total_mag > 1e-10:
                    variance = (magnitude * (freqs - centroid) ** 2).sum() / total_mag
                    features[w, col_offset + 2] = np.sqrt(max(variance, 0.0))
                else:
                    features[w, col_offset + 2] = 0.0

                # Spectral energy
                features[w, col_offset + 3] = (magnitude**2).sum()

        return features

    def compute_rms_envelope(
        self, signals: np.ndarray, window_size: int = 64
    ) -> np.ndarray:
        """Compute RMS envelope for each channel using a sliding window.

        Args:
            signals: Array of shape (num_timesteps, num_channels).
            window_size: Size of the RMS computation window.

        Returns:
            RMS envelope array of shape (num_timesteps, num_channels).
        """
        envelope = np.zeros_like(signals)
        half_w = window_size // 2

        for ch in range(signals.shape[1]):
            padded = np.pad(signals[:, ch], (half_w, half_w), mode="reflect")
            cumsum_sq = np.cumsum(padded**2)
            rms = np.sqrt(
                (cumsum_sq[window_size:] - cumsum_sq[:-window_size]) / window_size
            )
            envelope[:, ch] = rms[: signals.shape[0]]

        return envelope

    def detect_transients(
        self, signals: np.ndarray, threshold_sigma: float = 3.0
    ) -> np.ndarray:
        """Detect sudden transients that may indicate arc initiation.

        Uses first-order differentiation and threshold-based detection
        on each channel independently.

        Args:
            signals: Array of shape (num_timesteps, num_channels).
            threshold_sigma: Number of standard deviations above the mean
                derivative magnitude to flag as a transient.

        Returns:
            Boolean array of shape (num_timesteps, num_channels) marking
            detected transient locations.
        """
        diff = np.diff(signals, axis=0, prepend=signals[:1])
        abs_diff = np.abs(diff)

        mean_diff = abs_diff.mean(axis=0, keepdims=True)
        std_diff = abs_diff.std(axis=0, keepdims=True)
        std_diff = np.where(std_diff < 1e-8, 1.0, std_diff)

        threshold = mean_diff + threshold_sigma * std_diff
        return abs_diff > threshold
