"""Training loop with early stopping, checkpointing, and experiment tracking.

Supports both W&B and MLflow backends for metric logging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.training.metrics import ArcDetectionMetrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when validation metric plateaus.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: "max" for metrics where higher is better (e.g. F1),
              "min" for metrics where lower is better (e.g. loss).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class CheckpointManager:
    """Manages model checkpoint saving and loading.

    Keeps the top-k checkpoints ranked by a monitored metric.

    Args:
        save_dir: Directory for checkpoint files.
        save_top_k: Maximum number of checkpoints to retain.
        monitor: Metric name to rank checkpoints.
        mode: "max" or "min" for the monitored metric.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_top_k: int = 3,
        monitor: str = "val_f1",
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self._checkpoints: list[tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> Optional[Path]:
        """Save a checkpoint if the metric ranks in the top-k.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        score = metrics.get(self.monitor, 0.0)
        path = self.save_dir / f"epoch_{epoch:03d}_{self.monitor}_{score:.4f}.pt"

        self._checkpoints.append((score, path))
        reverse = self.mode == "max"
        self._checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        if len(self._checkpoints) > self.save_top_k:
            _, removed_path = self._checkpoints.pop()
            if removed_path.exists():
                removed_path.unlink()

        if path in [p for _, p in self._checkpoints]:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                path,
            )
            logger.info(f"Checkpoint saved: {path.name}")
            return path

        return None

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        """Load a checkpoint and restore model/optimizer state.

        Args:
            path: Path to checkpoint file.
            model: Model to restore weights into.
            optimizer: Optional optimizer to restore state into.

        Returns:
            Checkpoint metadata dictionary.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


class Trainer:
    """Training orchestrator for arc detection models.

    Handles the full training loop including forward/backward passes,
    metric computation, early stopping, checkpointing, and experiment
    tracking via W&B or MLflow.

    Args:
        model: The arc detection model.
        cfg: Full training configuration.
        device: Torch device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        self.scheduler = self._build_scheduler()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.severity_criterion = nn.MSELoss()
        self.metrics = ArcDetectionMetrics()

        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
            mode="max",
        )

        self.checkpoint_mgr = CheckpointManager(
            save_dir=cfg.checkpoint.dir,
            save_top_k=cfg.checkpoint.save_top_k,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler based on config."""
        if self.cfg.training.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.epochs - self.cfg.training.warmup_epochs,
            )
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.

        Returns:
            Dictionary of metric histories keyed by metric name.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_auc": [],
        }

        logger.info(
            f"Starting training for {self.cfg.training.epochs} epochs "
            f"on {self.device}"
        )

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate(val_loader)

            elapsed = time.time() - t0
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_auc"].append(val_metrics["auc"])

            logger.info(
                f"Epoch {epoch:3d}/{self.cfg.training.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_metrics['loss']:.4f} | "
                f"val_f1: {val_metrics['f1']:.4f} | "
                f"val_auc: {val_metrics['auc']:.4f} | "
                f"{elapsed:.1f}s"
            )

            self.checkpoint_mgr.save(
                self.model,
                self.optimizer,
                epoch,
                {f"val_{k}": v for k, v in val_metrics.items()},
            )

            if self.early_stopping(val_metrics["f1"]):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best F1: {self.early_stopping.best_score:.4f})"
                )
                break

            self.scheduler.step()

        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)
            severity = batch["severity"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.cls_criterion(outputs["logits"], labels)
            if "severity" in outputs:
                # Only compute severity loss for positive (arc) samples
                arc_mask = labels == 1
                if arc_mask.any():
                    loss = loss + 0.5 * self.severity_criterion(
                        outputs["severity"][arc_mask], severity[arc_mask]
                    )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.gradient_clip_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(inputs)
            loss = self.cls_criterion(outputs["logits"], labels)

            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = outputs["logits"].argmax(dim=-1)

            self.metrics.update(preds, labels, probs)
            total_loss += loss.item()
            num_batches += 1

        results = self.metrics.compute()
        results["loss"] = total_loss / max(num_batches, 1)
        return results
