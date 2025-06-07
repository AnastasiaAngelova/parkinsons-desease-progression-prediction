import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import onnxmltools
import pandas as pd
from onnxmltools.convert import convert_lightgbm
from onnxmltools.utils import save_model

from parkinsons_desease_progression_prediction.data.preprocessing import preprocess_data
from parkinsons_desease_progression_prediction.models.lightgbm_model import LightGBMClassifier
from parkinsons_desease_progression_prediction.utils.metrics import smape_metric
from parkinsons_desease_progression_prediction.utils.splitting import create_patient_splits
from parkinsons_desease_progression_prediction.utils.utils import get_git_commit_id, setup_logging


def train_lgbm_model(config):
    setup_logging()
    logger = logging.getLogger(__name__)

    mlflow_uri = getattr(config.logging, "mlflow_uri", "http://127.0.0.1:8080")
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = getattr(
        config.logging, "experiment_name", "parkinsons_prediction"
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"lgbm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        git_commit = get_git_commit_id()
        mlflow.log_param("git_commit_id", git_commit)
        logger.info(f"Starting experiment with git commit: {git_commit}")

        hyperparams = {
            "seed": config.task.seed,
            "test_size": config.task.test_size,
            "val_size": config.task.val_size,
            "use_preprocessing": config.task.get("use_preprocessing", False),
            "target_column": config.task.get("target_column", "target"),
            "model_type": "LightGBM",
        }

        if hasattr(config.task, "model"):
            lgbm_params = dict(config.task.model)
            hyperparams.update({f"lgbm_{k}": v for k, v in lgbm_params.items()})

        mlflow.log_params(hyperparams)

        logger.info(f"Hyperparameters: {hyperparams}")

        mlflow.log_param("n_features", len(config.data.features))
        mlflow.log_param("feature_names", str(config.data.features))

        logger.info("Loading and preprocessing data...")
        if config.task.get("use_preprocessing", False):
            data = preprocess_data(config)
            logger.info("Data preprocessing completed")
        else:
            data = pd.read_csv(config.data["preprocessed_data_path"])
            logger.info(
                f"Loaded preprocessed data from {config.data['preprocessed_data_path']}"
            )

        train_data, val_data, test_data = create_patient_splits(
            data, config.task.test_size, config.task.val_size, config.task.seed
        )

        logger.info("Data splitting completed")

        mlflow.log_params(
            {
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
                "total_samples": len(data),
            }
        )

        features = config.data.features

        logger.info("Starting model training...")

        model = LightGBMClassifier(config.task.model, features)
        model.fit(train_data)

        logger.info("Training completed")

        val_preds = model.predict(val_data)
        val_true = val_data[config.task.get("target_column", "target")].values
        val_smape = smape_metric(val_true, val_preds)

        logger.info(f"Validation SMAPE: {val_smape:.4f}")

        test_preds = model.predict(test_data)
        test_true = test_data[config.task.get("target_column", "target")].values
        test_smape = smape_metric(test_true, test_preds)

        logger.info(f"Test SMAPE: {test_smape:.4f}")

        mlflow.log_metrics({"val_smape": val_smape, "test_smape": test_smape})

        sample_input = val_data[features].iloc[:1].values.astype(np.float32)

        mlflow.sklearn.log_model(
            model.model,
            "model",
            input_example=sample_input,
            registered_model_name=f"parkinsons_lgbm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        logger.info("Converting model to ONNX format...")

        try:
            onnx_model = convert_lightgbm(
                model.model,
                initial_types=[
                    (
                        "input",
                        onnxmltools.convert.common.data_types.FloatTensorType(
                            [1, len(features)]
                        ),
                    )
                ],
                target_opset=11,
            )

            onnx_path = config.task["onnx_path"]
            Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

            save_model(onnx_model, onnx_path)

            mlflow.log_artifact(onnx_path, "onnx_model")
            mlflow.log_param("onnx_model_path", onnx_path)

            logger.info(f"ONNX model saved to: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run completed successfully! Run ID: {run_id}")
        logger.info("View results with: mlflow ui --backend-store-uri ./mlruns")

    return model
