import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch.onnx
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from parkinsons_desease_progression_prediction.data.preprocessing import preprocess_data
from parkinsons_desease_progression_prediction.models.lightning_model import ParkinsonsLightningModel
from parkinsons_desease_progression_prediction.training.dataset import ParkinsonsDataModule
from parkinsons_desease_progression_prediction.utils.splitting import create_patient_splits
from parkinsons_desease_progression_prediction.utils.utils import get_git_commit_id, setup_logging


def train_lightning_model(config):
    setup_logging()
    logger = logging.getLogger(__name__)

    mlflow_uri = getattr(config.logging, "mlflow_uri", "http://127.0.0.1:8080")
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = getattr(
        config.logging, "experiment_name", "parkinsons_prediction"
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"lightning_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        git_commit = get_git_commit_id()
        mlflow.log_param("git_commit_id", git_commit)
        logger.info(f"Starting experiment with git commit: {git_commit}")

        hyperparams = {
            "seed": config.task.seed,
            "batch_size": config.task.batch_size,
            "epochs": config.task.epochs,
            "lr": config.task.model.lr,
            "n_hidden": config.task.model.n_hidden,
            "n_layers": config.task.model.n_layers,
            "dropout": config.task.model.dropout,
            "weight_decay": config.task.model.weight_decay,
            "warmup_steps": config.task.model.warmup_steps,
            "test_size": config.task.test_size,
            "val_size": config.task.val_size,
            "use_preprocessing": config.task.get("use_preprocessing", False),
            "target_column": config.task.get("target_column", "target"),
            "accelerator": config.task.get("accelerator", "auto"),
            "devices": config.task.get("devices", "auto"),
            "precision": config.task.get("precision", 32),
            "num_workers": config.task.get("num_workers", 4),
        }

        mlflow.log_params(hyperparams)

        logger.info(f"Hyperparameters: {hyperparams}")

        mlflow.log_param("n_features", len(config.data.features))

        mlflow.log_param("feature_names", str(config.data.features))

        pl.seed_everything(config.task.seed)

        logger.info("Loading and preprocessing data...")
        if config.task.get("use_preprocessing", False):
            data = preprocess_data(config)
            logger.info("Data preprocessing completed")
        else:
            data = pd.read_csv(config.data["preprocessed_train_data_path"])
            logger.info(
                f"Loaded preprocessed data from {config.data['preprocessed_train_data_path']}"
            )

        train_data, val_data, test_data = create_patient_splits(
            data, config.task.test_size, config.task.val_size, config.task.seed
        )

        logger.info("Data splitting...")

        mlflow.log_params(
            {
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
                "total_samples": len(data),
            }
        )

        features = config.data.features

        data_module = ParkinsonsDataModule(
            train_df=train_data,
            val_df=val_data,
            test_df=test_data,
            features=features,
            batch_size=config.task.batch_size,
            num_workers=config.task.get("num_workers", 4),
            target_column=config.task.get("target_column", "target"),
        )

        model = ParkinsonsLightningModel(
            n_features=len(features),
            n_hidden=config.task.model.n_hidden,
            n_layers=config.task.model.n_layers,
            lr=config.task.model.lr,
            dropout=config.task.model.dropout,
            weight_decay=config.task.model.weight_decay,
            warmup_steps=config.task.model.warmup_steps,
        )

        callbacks = [
            ModelCheckpoint(
                monitor="val_smape",
                mode="min",
                save_top_k=1,
                filename="best_model_{epoch:02d}_{val_smape:.4f}",
            ),
            EarlyStopping(monitor="val_smape", patience=10, mode="min", verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_uri,
            run_id=mlflow.active_run().info.run_id,
        )

        run_dir = (
            Path("checkpoints")
            / f"lightning_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        trainer = pl.Trainer(
            default_root_dir=run_dir,
            max_epochs=config.task.epochs,
            logger=mlflow_logger,
            callbacks=callbacks,
            accelerator=config.task.get("accelerator", "auto"),
            devices=config.task.get("devices", "auto"),
            precision=config.task.get("precision", 32),
            deterministic=True,
            enable_checkpointing=True,
            log_every_n_steps=50,
        )

        logger.info("Starting model training...")

        trainer.fit(model, data_module)

        logger.info("Training finished")

        data_module.setup("fit")
        val_dataloader = data_module.val_dataloader()
        sample_batch = next(iter(val_dataloader))
        sample_input = sample_batch["features"][:1]

        input_example = sample_input.cpu().numpy()

        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example,
            registered_model_name=f"parkinsons_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        if trainer.checkpoint_callback.best_model_path:
            mlflow.log_artifact(
                trainer.checkpoint_callback.best_model_path, "checkpoints"
            )
            mlflow.log_param(
                "best_model_path", trainer.checkpoint_callback.best_model_path
            )
            mlflow.log_metric(
                "best_val_smape", trainer.checkpoint_callback.best_model_score.item()
            )

        if hasattr(model, "test_results"):
            mlflow.log_metrics(
                {
                    "final_test_smape": model.test_results["smape"],
                    "final_test_mae": model.test_results["mae"],
                    "final_test_rmse": model.test_results["rmse"],
                }
            )

        plots_dir = Path("plots")
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "plots")

        logger.info("Artifacts logged")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run completed successfully! Run ID: {run_id}")
        logger.info("View results with: mlflow ui --backend-store-uri ./mlruns")

        logger.info("Converting model to ONNX format...")

        Path(config.task["onnx_path"]).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            sample_input,
            config.task["onnx_path"],
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        mlflow.log_artifact(config.task["onnx_path"], "onnx_model")
        mlflow.log_param("onnx_model_path", config.task["onnx_path"])

        logger.info(f"ONNX model saved to: {config.task['onnx_path']}")

    return model
