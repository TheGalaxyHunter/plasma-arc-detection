"""CNN-LSTM hybrid model for temporal plasma arc detection.

Architecture overview:
  1. 1D CNN layers extract local features from multi-channel sensor input
  2. LSTM layers capture temporal dependencies across the sequence
  3. Self-attention selects the most relevant temporal features
  4. Classification head outputs arc/no-arc prediction and severity score
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.attention import TemporalSelfAttention


class CNNFeatureExtractor(nn.Module):
    """1D CNN stack for extracting local features from sensor channels.

    Applies a series of Conv1d -> BatchNorm -> Activation -> MaxPool blocks
    to progressively downsample the temporal dimension while increasing
    feature depth.

    Args:
        in_channels: Number of input sensor channels.
        channel_sizes: List of output channel sizes for each conv layer.
        kernel_sizes: List of kernel sizes for each conv layer.
        pool_size: Max pooling kernel and stride size.
        dropout: Dropout rate after each conv block.
        activation: Activation function name ("relu", "gelu").
    """

    def __init__(
        self,
        in_channels: int = 6,
        channel_sizes: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_size: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        channel_sizes = channel_sizes or [32, 64, 128]
        kernel_sizes = kernel_sizes or [7, 5, 3]
        assert len(channel_sizes) == len(kernel_sizes)

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for out_ch, k_size in zip(channel_sizes, kernel_sizes):
            padding = k_size // 2
            layers.extend([
                nn.Conv1d(prev_channels, out_ch, kernel_size=k_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                act_fn,
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
                nn.Dropout(dropout),
            ])
            prev_channels = out_ch

        self.net = nn.Sequential(*layers)
        self.out_channels = channel_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Feature tensor of shape (batch, out_channels, reduced_seq_len).
        """
        return self.net(x)


class TemporalEncoder(nn.Module):
    """Bidirectional LSTM for capturing temporal patterns in CNN features.

    Args:
        input_size: Feature dimension from CNN output.
        hidden_size: LSTM hidden state size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (applied when num_layers > 1).
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, features).

        Returns:
            LSTM outputs of shape (batch, seq_len, output_size).
        """
        outputs, _ = self.lstm(x)
        return outputs


class ClassificationHead(nn.Module):
    """Dual-output head for arc classification and severity regression.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout rate in the MLP.
        num_classes: Number of classification categories.
        predict_severity: Whether to include a severity regression output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.4,
        num_classes: int = 2,
        predict_severity: bool = True,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [128, 64]
        self.predict_severity = predict_severity

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev_dim, num_classes)

        if predict_severity:
            self.severity_head = nn.Sequential(
                nn.Linear(prev_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Feature tensor of shape (batch, input_dim).

        Returns:
            Tuple of (class_logits, severity_score).
            severity_score is None if predict_severity is False.
        """
        features = self.shared(x)
        logits = self.cls_head(features)

        severity = None
        if self.predict_severity:
            severity = self.severity_head(features).squeeze(-1)

        return logits, severity


class ArcDetectionModel(nn.Module):
    """CNN-LSTM model for plasma arc detection from sensor time series.

    Combines convolutional feature extraction, recurrent temporal encoding,
    self-attention pooling, and a dual-output classification/regression head.

    Args:
        cfg: Model configuration (OmegaConf DictConfig).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cnn = CNNFeatureExtractor(
            in_channels=cfg.cnn.in_channels,
            channel_sizes=list(cfg.cnn.channel_sizes),
            kernel_sizes=list(cfg.cnn.kernel_sizes),
            pool_size=cfg.cnn.pool_size,
            dropout=cfg.cnn.dropout,
            activation=cfg.cnn.activation,
        )

        self.temporal = TemporalEncoder(
            input_size=self.cnn.out_channels,
            hidden_size=cfg.lstm.hidden_size,
            num_layers=cfg.lstm.num_layers,
            dropout=cfg.lstm.dropout,
            bidirectional=cfg.lstm.bidirectional,
        )

        self.attention = TemporalSelfAttention(
            embed_dim=cfg.attention.embed_dim,
            num_heads=cfg.attention.num_heads,
            dropout=cfg.attention.dropout,
        )

        self.classifier = ClassificationHead(
            input_dim=cfg.attention.embed_dim,
            hidden_dims=list(cfg.classifier.hidden_dims),
            dropout=cfg.classifier.dropout,
            num_classes=cfg.classifier.num_classes,
            predict_severity=cfg.classifier.predict_severity,
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the full pipeline.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Dictionary with keys:
                - "logits": classification logits (batch, num_classes)
                - "severity": severity scores (batch,), if enabled
                - "attention_weights": attention weights (batch, seq_len')
        """
        # CNN feature extraction: (batch, channels, seq_len) -> (batch, cnn_out, reduced_len)
        cnn_features = self.cnn(x)

        # Reshape for LSTM: (batch, reduced_len, cnn_out)
        temporal_input = cnn_features.permute(0, 2, 1)

        # Temporal encoding: (batch, reduced_len, lstm_out)
        temporal_features = self.temporal(temporal_input)

        # Self-attention pooling: (batch, embed_dim), (batch, reduced_len)
        context, attn_weights = self.attention(temporal_features)

        # Classification
        logits, severity = self.classifier(context)

        output = {
            "logits": logits,
            "attention_weights": attn_weights,
        }
        if severity is not None:
            output["severity"] = severity

        return output
