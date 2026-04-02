"""Inference pipeline for real-time plasma arc detection.

Provides a high-level API for loading a trained model and running
predictions on new sensor data, either in batch or streaming mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.preprocessing import SignalProcessor
from src.models.cnn_lstm import ArcDetectionModel

logger = logging.getLogger(__name__)


class ArcDetector:
    """High-level inference wrapper for arc detection.

    Loads a trained model checkpoint and provides methods for single-window
    and streaming predictions on raw sensor data.

    Args:
        checkpoint_path: Path to a saved model checkpoint.
        config_path: Path to the model config YAML. If None, config is
            loaded from the checkpoint metadata.
        device: Torch device for inference. Defaults to CUDA if available.
        confidence_threshold: Minimum probability to classify as arc event.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.confidence_threshold = confidence_threshold

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if config_path is not None:
            cfg = OmegaConf.load(config_path)
        else:
            cfg = OmegaConf.create(checkpoint.get("config", {}))

        self.model = ArcDetectionModel(cfg.model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.processor = SignalProcessor(
            num_channels=cfg.get("data", {}).get("num_channels", 6),
            sample_rate=cfg.get("data", {}).get("sample_rate_hz", 10_000),
        )

        self._sequence_length = cfg.get("data", {}).get("sequence_length", 256)
        logger.info(
            f"ArcDetector initialized on {self.device} "
            f"(threshold={confidence_threshold})"
        )

    @torch.no_grad()
    def predict(self, signals: np.ndarray) -> dict[str, float | bool | np.ndarray]:
        """Run arc detection on a single window of sensor data.

        Args:
            signals: Array of shape (seq_len, num_channels) containing
                raw sensor readings.

        Returns:
            Dictionary with keys:
                - "arc_detected": bool indicating if arc was found.
                - "confidence": float probability of arc event.
                - "severity": float severity score (0 to 1).
                - "attention_weights": array of attention weights over time.
        """
        normalized = self.processor.normalize(signals)

        # (seq_len, channels) -> (1, channels, seq_len)
        x = torch.from_numpy(normalized).float().permute(1, 0).unsqueeze(0)
        x = x.to(self.device)

        outputs = self.model(x)
        probs = torch.softmax(outputs["logits"], dim=-1)
        arc_prob = probs[0, 1].item()

        result = {
            "arc_detected": arc_prob >= self.confidence_threshold,
            "confidence": arc_prob,
            "attention_weights": outputs["attention_weights"][0].cpu().numpy(),
        }

        if "severity" in outputs:
            result["severity"] = outputs["severity"][0].item()

        return result

    @torch.no_grad()
    def predict_stream(
        self,
        signals: np.ndarray,
        stride: Optional[int] = None,
    ) -> list[dict]:
        """Run arc detection over a long recording using sliding windows.

        Args:
            signals: Array of shape (total_timesteps, num_channels).
            stride: Step size between windows. Defaults to half the
                sequence length for 50% overlap.

        Returns:
            List of prediction dictionaries, one per window. Each includes
            a "window_start" key with the starting timestep index.
        """
        if stride is None:
            stride = self._sequence_length // 2

        total_len = signals.shape[0]
        results: list[dict] = []

        for start in range(0, total_len - self._sequence_length + 1, stride):
            window = signals[start : start + self._sequence_length]
            pred = self.predict(window)
            pred["window_start"] = start
            pred["window_end"] = start + self._sequence_length
            results.append(pred)

        logger.info(
            f"Processed {len(results)} windows "
            f"({total_len} timesteps, stride={stride})"
        )
        return results

    def get_arc_events(
        self, stream_results: list[dict], merge_gap: int = 2
    ) -> list[dict]:
        """Post-process streaming results to extract contiguous arc events.

        Merges adjacent arc-positive windows into discrete events with
        start/end times and peak severity.

        Args:
            stream_results: Output from predict_stream().
            merge_gap: Maximum number of non-arc windows between two arc
                windows to still merge them into one event.

        Returns:
            List of event dictionaries with keys: start, end, peak_confidence,
            peak_severity, num_windows.
        """
        events: list[dict] = []
        current_event: Optional[dict] = None
        gap_count = 0

        for result in stream_results:
            if result["arc_detected"]:
                if current_event is None:
                    current_event = {
                        "start": result["window_start"],
                        "end": result["window_end"],
                        "peak_confidence": result["confidence"],
                        "peak_severity": result.get("severity", 0.0),
                        "num_windows": 1,
                    }
                else:
                    current_event["end"] = result["window_end"]
                    current_event["peak_confidence"] = max(
                        current_event["peak_confidence"], result["confidence"]
                    )
                    current_event["peak_severity"] = max(
                        current_event["peak_severity"],
                        result.get("severity", 0.0),
                    )
                    current_event["num_windows"] += 1
                gap_count = 0
            else:
                if current_event is not None:
                    gap_count += 1
                    if gap_count > merge_gap:
                        events.append(current_event)
                        current_event = None
                        gap_count = 0

        if current_event is not None:
            events.append(current_event)

        return events
