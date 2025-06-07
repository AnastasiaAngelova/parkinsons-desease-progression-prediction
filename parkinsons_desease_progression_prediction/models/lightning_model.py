from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from parkinsons_desease_progression_prediction.utils.metrics import smape_metric


class ParkinsonsLightningModel(pl.LightningModule):
    """PyTorch Lightning model for Parkinsons disease prediction."""

    def __init__(
        self,
        n_features,
        n_hidden=64,
        n_layers=2,
        lr=2e-3,
        dropout=0.1,
        weight_decay=0.01,
        warmup_steps=1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = [nn.Linear(n_features, n_hidden), nn.LeakyReLU(), nn.Dropout(dropout)]

        for _ in range(n_layers):
            layers.extend(
                [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(), nn.Dropout(dropout)]
            )

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(n_hidden, 1)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.train_losses = []
        self.val_losses = []
        self.val_smapes = []

    def forward(self, x):
        """Forward pass."""
        x = self.network(x)
        return self.head(x).squeeze(-1)

    def smape_loss(self, predictions, targets):
        """Custom SMAPE loss function."""
        return torch.mean(
            torch.abs(targets - predictions)
            / (torch.abs(0.01 + targets) + torch.abs(0.01 + predictions))
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        features = batch["features"]
        targets = batch["target"]

        predictions = self(features)
        loss = self.smape_loss(predictions, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_losses.append(loss.item())

        mlflow.log_metric("train_loss_step", loss.item(), step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features = batch["features"]
        targets = batch["target"]

        predictions = self(features)
        loss = self.smape_loss(predictions, targets)

        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "predictions": predictions.detach().cpu(),
                "targets": targets.detach().cpu(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """Calculate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        preds_denorm = all_preds * 100
        targets_denorm = all_targets * 100

        smape_score = smape_metric(
            targets_denorm.float().numpy(), preds_denorm.float().numpy()
        )

        mae = torch.mean(torch.abs(targets_denorm - preds_denorm)).item()
        rmse = torch.sqrt(torch.mean((targets_denorm - preds_denorm) ** 2)).item()

        self.log("val_smape", smape_score, prog_bar=True)
        self.log("val_mae", mae)
        self.log("val_rmse", rmse)

        avg_val_loss = torch.mean(
            torch.stack([x["val_loss"] for x in self.validation_step_outputs])
        ).item()
        self.val_losses.append(avg_val_loss)
        self.val_smapes.append(smape_score)

        mlflow.log_metric("val_loss_epoch", avg_val_loss, step=self.current_epoch)
        mlflow.log_metric("val_smape_epoch", smape_score, step=self.current_epoch)
        mlflow.log_metric("val_mae_epoch", mae, step=self.current_epoch)
        mlflow.log_metric("val_rmse_epoch", rmse, step=self.current_epoch)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        features = batch["features"]
        targets = batch["target"]

        predictions = self(features)
        loss = self.smape_loss(predictions, targets)

        self.test_step_outputs.append(
            {
                "test_loss": loss,
                "predictions": predictions.detach().cpu(),
                "targets": targets.detach().cpu(),
            }
        )

        return loss

    def on_test_epoch_end(self):
        """Calculate test metrics at epoch end."""
        if not self.test_step_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

        preds_denorm = all_preds * 100
        targets_denorm = all_targets * 100

        smape_score = smape_metric(
            targets_denorm.float().numpy(), preds_denorm.float().numpy()
        )

        mae = torch.mean(torch.abs(targets_denorm - preds_denorm)).item()
        rmse = torch.sqrt(torch.mean((targets_denorm - preds_denorm) ** 2)).item()

        self.log("test_smape", smape_score)
        self.log("test_mae", mae)
        self.log("test_rmse", rmse)

        self.test_results = {
            "smape": smape_score,
            "mae": mae,
            "rmse": rmse,
            "predictions": preds_denorm.float().numpy(),
            "targets": targets_denorm.float().numpy(),
        }

        mlflow.log_metric("test_smape_final", smape_score)
        mlflow.log_metric("test_mae_final", mae)
        mlflow.log_metric("test_rmse_final", rmse)

        self.create_and_log_plots()

        self.test_step_outputs.clear()

    def create_and_log_plots(self):
        """Create and log training plots."""
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.val_losses) + 1)

        train_losses_epoch = []
        steps_per_epoch = (
            len(self.train_losses) // len(self.val_losses)
            if len(self.val_losses) > 0
            else 1
        )
        for i in range(len(self.val_losses)):
            start_idx = i * steps_per_epoch
            end_idx = min((i + 1) * steps_per_epoch, len(self.train_losses))
            if start_idx < len(self.train_losses):
                train_losses_epoch.append(np.mean(self.train_losses[start_idx:end_idx]))

        plt.plot(
            epochs[: len(train_losses_epoch)],
            train_losses_epoch,
            "b-",
            label="Training Loss",
            alpha=0.7,
        )
        plt.plot(epochs, self.val_losses, "r-", label="Validation Loss", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("SMAPE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        loss_plot_path = plots_dir / "training_validation_loss.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(loss_plot_path))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.val_smapes, "g-", label="Validation SMAPE", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("SMAPE Score")
        plt.title("Validation SMAPE Score Over Training")
        plt.legend()
        plt.grid(True, alpha=0.3)

        smape_plot_path = plots_dir / "validation_smape.png"
        plt.savefig(smape_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(smape_plot_path))

        if hasattr(self, "test_results"):
            plt.figure(figsize=(10, 8))
            predictions = self.test_results["predictions"]
            targets = self.test_results["targets"]

            plt.subplot(2, 1, 1)
            plt.scatter(targets, predictions, alpha=0.6, s=20)

            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            plt.plot(
                [min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2
            )

            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predictions vs Actual Values (Test Set)")
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            residuals = predictions - targets
            plt.scatter(targets, residuals, alpha=0.6, s=20)
            plt.axhline(y=0, color="r", linestyle="--", alpha=0.8)
            plt.xlabel("Actual Values")
            plt.ylabel("Residuals (Predicted - Actual)")
            plt.title("Residuals Plot")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            predictions_plot_path = plots_dir / "predictions_vs_actual.png"
            plt.savefig(predictions_plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            mlflow.log_artifact(str(predictions_plot_path))

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
