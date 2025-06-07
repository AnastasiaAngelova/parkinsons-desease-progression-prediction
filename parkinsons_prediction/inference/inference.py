import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from omegaconf import DictConfig

from parkinsons_prediction.data.preprocessing import preprocess_data
from parkinsons_prediction.utils.utils import setup_logging


def optimize_smape_predictions(probabilities):
    """Optimize predictions for SMAPE metric (for LightGBM predictions)."""

    def single_smape(preds, target):
        x = np.tile(np.arange(preds.shape[1]), (preds.shape[0], 1))
        x = np.abs(x - target) / (2 + x + target)
        return (x * preds).sum(axis=1)

    smape_scores = np.hstack(
        [
            single_smape(probabilities, i).reshape(-1, 1)
            for i in range(probabilities.shape[1])
        ]
    )
    return smape_scores.argmin(axis=1)


def load_onnx_model(model_path):
    """Load ONNX model and create inference session."""
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    return session


def predict_with_lgbm_onnx(session, features):
    """Make predictions with LightGBM ONNX model."""
    logger = logging.getLogger(__name__)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_shape = session.get_inputs()[0].shape

    if input_shape[0] == 1:
        predictions = []
        for i in range(features.shape[0]):
            single_sample = features[i : i + 1].astype(np.float32)

            try:
                output = session.run([output_name], {input_name: single_sample})[0]
                predictions.append(output)
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                logger.error(f"Sample shape: {single_sample.shape}")
                raise

        if len(predictions[0].shape) == 2:
            outputs = np.vstack(predictions)
        else:
            outputs = np.array(predictions)
    else:
        outputs = session.run([output_name], {input_name: features.astype(np.float32)})[
            0
        ]

    logger.info(f"Raw ONNX outputs shape: {outputs.shape}")

    if len(outputs.shape) == 2 and outputs.shape[1] > 1:
        probabilities = outputs
        predictions = optimize_smape_predictions(probabilities)
    elif len(outputs.shape) == 1 or (len(outputs.shape) == 2 and outputs.shape[1] == 1):
        if len(outputs.shape) == 2:
            outputs = outputs.flatten()
        predictions = np.round(outputs).astype(int)
        predictions = np.clip(predictions, 0, 4)
    else:
        raise ValueError(
            f"Unexpected output shape from LightGBM model: {outputs.shape}"
        )

    return predictions


def predict_with_nn_onnx(session, features):
    """Make predictions with Neural Network ONNX model."""
    logger = logging.getLogger(__name__)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: features.astype(np.float32)})[0]

    predictions = np.round(outputs).astype(int)
    predictions = np.clip(predictions, 0, 4)

    logger.info(
        f"NN final predictions range: {predictions.min()} - {predictions.max()}"
    )
    logger.info(f"NN predictions sample: {predictions[:10]}")

    return predictions


def create_submission_dataframe(test_data, predictions):
    test_data = test_data.copy()
    test_data["rating"] = predictions

    test_data["updrs"] = test_data["target_i"].apply(lambda x: f"updrs_{x}")
    test_data["delta_months"] = test_data["horizon"]

    test_data["prediction_id"] = (
        test_data["patient_id"].astype(str)
        + "_"
        + test_data["visit_month"].astype(str)
        + "_"
        + test_data["updrs"]
        + "_plus_"
        + test_data["delta_months"].astype(str)
        + "_months"
    )

    test_data["group_key"] = test_data["visit_month"]

    return test_data[["prediction_id", "rating", "group_key"]].drop_duplicates()


def run_inference(cfg: DictConfig):
    """Main inference function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting ensemble inference...")

    logger.info("Loading test data...")
    if cfg.task.get("use_preprocessing", False):
        test_data = preprocess_data(cfg)
        logger.info("Test data preprocessing completed")
    else:
        test_data = pd.read_csv(
            cfg.data.get("test_data_path", "data/test_processed.csv")
        )
        logger.info("Loaded preprocessed test data")

    features = cfg.data.features
    logger.info(f"Using {len(features)} features for prediction")
    X_test = test_data[features].values

    lgbm_model_path = Path(cfg.task.get("lgbm_onnx_path", "parkinsons_lgbm_model.onnx"))
    nn_model_path = Path(cfg.task.get("nn_onnx_path", "parkinsons_model.onnx"))

    if not lgbm_model_path.exists():
        logger.error(f"LightGBM ONNX model not found at {lgbm_model_path}")
        raise FileNotFoundError(f"LightGBM ONNX model not found at {lgbm_model_path}")

    if not nn_model_path.exists():
        logger.error(f"Neural Network ONNX model not found at {nn_model_path}")
        raise FileNotFoundError(
            f"Neural Network ONNX model not found at {nn_model_path}"
        )

    logger.info(f"Loading LightGBM ONNX model from {lgbm_model_path}")
    lgbm_session = load_onnx_model(lgbm_model_path)

    logger.info(f"Loading Neural Network ONNX model from {nn_model_path}")
    nn_session = load_onnx_model(nn_model_path)

    logger.info("Making predictions with LightGBM model...")
    lgbm_predictions = predict_with_lgbm_onnx(lgbm_session, X_test)
    logger.info(
        f"LightGBM predictions range: {lgbm_predictions.min()} - {lgbm_predictions.max()}"
    )

    logger.info("Making predictions with Neural Network model...")
    nn_predictions = predict_with_nn_onnx(nn_session, X_test)
    logger.info(
        f"Neural Network predictions range: {nn_predictions.min()} - {nn_predictions.max()}"
    )

    ensemble_method = cfg.task.get("ensemble_method", "average")

    if ensemble_method == "average":
        logger.info("Averaging predictions from both models...")
        ensemble_predictions = np.round((lgbm_predictions + nn_predictions) / 2).astype(
            int
        )
    elif ensemble_method == "weighted":
        lgbm_weight = cfg.task.get("lgbm_weight", 0.5)
        nn_weight = cfg.task.get("nn_weight", 0.5)
        logger.info(f"Weighted averaging: LightGBM={lgbm_weight}, NN={nn_weight}")
        ensemble_predictions = np.round(
            lgbm_weight * lgbm_predictions + nn_weight * nn_predictions
        ).astype(int)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    logger.info("Creating submission file...")
    submission_df = create_submission_dataframe(test_data, ensemble_predictions)

    output_path = Path(cfg.task.get("output_path", "predictions"))
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = output_path / f"ensemble_predictions_{timestamp}.csv"

    submission_df.to_csv(submission_file, index=False)
    logger.info(f"Predictions saved to {submission_file}")

    logger.info(f"Total predictions: {len(submission_df)}")

    if cfg.task.get("save_individual_predictions", False):
        lgbm_submission = create_submission_dataframe(test_data, lgbm_predictions)
        nn_submission = create_submission_dataframe(test_data, nn_predictions)

        lgbm_file = output_path / f"lgbm_predictions_{timestamp}.csv"
        nn_file = output_path / f"nn_predictions_{timestamp}.csv"

        lgbm_submission.to_csv(lgbm_file, index=False)
        nn_submission.to_csv(nn_file, index=False)

        logger.info("Individual predictions saved:")
        logger.info(f"  - LightGBM: {lgbm_file}")
        logger.info(f"  - Neural Network: {nn_file}")

    logger.info("Inference completed successfully!")
