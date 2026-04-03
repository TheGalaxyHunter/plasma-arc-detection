[![CI](https://github.com/TheGalaxyHunter/plasma-arc-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/TheGalaxyHunter/plasma-arc-detection/actions/workflows/ci.yml)

# plasma-arc-detection

**Deep learning pipeline for plasma arc detection and analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end ML/DL system for detecting and classifying plasma arc events in industrial settings.
Built on research conducted at the **Indian Institute of Science (IISc), CST Department**, this pipeline
processes multi-channel sensor data (voltage, current, spectral intensity) to identify arc discharges
in real time using a CNN-LSTM hybrid architecture with self-attention.

---

## Architecture

```
                        Input: Multi-channel sensor data
                        (voltage, current, spectral bands)
                                    |
                                    v
                    +-------------------------------+
                    |   Signal Preprocessing        |
                    |   - Bandpass filtering         |
                    |   - Z-score normalization      |
                    |   - Transient detection        |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |   1D CNN Feature Extractor     |
                    |   Conv1d -> BN -> GELU -> Pool |
                    |   [32, 64, 128] channels       |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |   Bidirectional LSTM           |
                    |   2 layers, hidden_size=128    |
                    |   Temporal dependency modeling  |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |   Self-Attention Pooling       |
                    |   4-head, learned query        |
                    |   Sequence -> fixed vector     |
                    +-------------------------------+
                                    |
                          +---------+---------+
                          |                   |
                          v                   v
                  +---------------+   +---------------+
                  | Arc/No-Arc    |   |   Severity    |
                  | Classifier    |   |   Regressor   |
                  | (2-class)     |   |   (0.0-1.0)   |
                  +---------------+   +---------------+
```

## Quick Start

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible Python environment and dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/TheGalaxyHunter/plasma-arc-detection.git
cd plasma-arc-detection
uv sync
```

### Training

```bash
# Using Hydra config
uv run python -m src.training.trainer --config-path=../configs --config-name=train

# Or use the convenience script
bash scripts/train.sh
```

### Inference

```python
from src.inference import ArcDetector

detector = ArcDetector(
    checkpoint_path="checkpoints/best_model.pt",
    config_path="configs/train.yaml",
    confidence_threshold=0.5,
)

# Single window prediction
result = detector.predict(sensor_window)
print(f"Arc detected: {result['arc_detected']} (conf: {result['confidence']:.3f})")

# Streaming over a long recording
events = detector.predict_stream(full_recording, stride=128)
arc_events = detector.get_arc_events(events)
```

## Project Structure

```
plasma-arc-detection/
├── configs/
│   ├── train.yaml              # Training + data + tracking config
│   └── model/
│       └── cnn_lstm.yaml       # CNN-LSTM architecture config
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset with windowed sampling
│   │   └── preprocessing.py    # Bandpass filtering, spectral features, RMS
│   ├── models/
│   │   ├── cnn_lstm.py         # CNN-LSTM hybrid model
│   │   └── attention.py        # Multi-head self-attention pooling
│   ├── training/
│   │   ├── trainer.py          # Training loop, early stopping, checkpoints
│   │   └── metrics.py          # Precision, recall, F1, AUC
│   └── inference/
│       └── predict.py          # Real-time and batch inference API
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── tests/
│   └── test_dataset.py
├── scripts/
│   └── train.sh
├── pyproject.toml
└── README.md
```

## Results

Performance on held-out test set (IISc plasma arc benchmark):

| Metric    | Value  |
|-----------|--------|
| Precision | 0.943  |
| Recall    | 0.927  |
| F1 Score  | 0.935  |
| ROC-AUC   | 0.981  |
| Accuracy  | 0.952  |

Model processes 256-sample windows at ~2.1ms per window on an NVIDIA A100,
making it suitable for real-time monitoring at 10 kHz sampling rates.

## Key Features

- **Multi-channel input**: Processes voltage, current, and spectral intensity data simultaneously
- **Hybrid CNN-LSTM**: Local feature extraction via 1D convolutions combined with temporal modeling via bidirectional LSTMs
- **Attention-based pooling**: Learned query attention highlights the most informative timesteps
- **Dual output**: Binary arc classification plus continuous severity estimation
- **Streaming inference**: Sliding-window prediction with automatic event merging
- **Experiment tracking**: Native support for W&B and MLflow
- **Hydra configs**: Composable YAML configuration for reproducible experiments

## Tech Stack

- **PyTorch** - model architecture and training
- **NumPy / SciPy** - signal processing and feature extraction
- **scikit-learn** - evaluation metrics
- **Hydra** - configuration management
- **W&B / MLflow** - experiment tracking
- **Pandas** - data manipulation

## References

This work builds on research conducted at the **Centre for Sustainable Technologies (CST),
Indian Institute of Science (IISc), Bengaluru**, focusing on electrical discharge
characterization and automated fault detection in plasma systems.

Key references:

1. Arc discharge detection in industrial plasma systems using spectral analysis (IISc CST, 2023)
2. Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997.
3. Vaswani, A. et al. "Attention Is All You Need." NeurIPS, 2017.
4. Bahdanau, D. et al. "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR, 2015.

## License

MIT License. See [LICENSE](LICENSE) for details.
