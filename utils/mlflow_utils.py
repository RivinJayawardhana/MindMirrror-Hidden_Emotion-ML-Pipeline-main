"""
MLflow utilities for experiment tracking and model logging.
"""
import os
import pickle
import mlflow
import mlflow.pytorch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def init_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "hidden_emotion_detection"):
    """
    Initialize MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI (default: file:./mlruns)
        experiment_name: Experiment name
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Default to local file storage
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment: {experiment_name}")
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow initialized: tracking_uri={mlflow.get_tracking_uri()}")


def log_pytorch_model(model, artifact_path: str = "model"):
    """
    Log PyTorch model to MLflow.
    
    Args:
        model: PyTorch model
        artifact_path: Artifact path in MLflow
    """
    try:
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)
        logger.info(f"Logged PyTorch model to MLflow: {artifact_path}")
    except Exception as e:
        logger.warning(f"Failed to log PyTorch model: {e}")


def log_label_encoder(label_encoder, artifact_path: str = "label_encoder.pkl"):
    """
    Log label encoder to MLflow.
    
    Args:
        label_encoder: sklearn LabelEncoder
        artifact_path: Artifact path in MLflow
    """
    try:
        with open(artifact_path, "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)  # Clean up temp file
        logger.info(f"Logged label encoder to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log label encoder: {e}")


def log_training_config(config: Dict[str, Any], artifact_path: str = "training_config.json"):
    """
    Log training configuration to MLflow.
    
    Args:
        config: Configuration dictionary
        artifact_path: Artifact path in MLflow
    """
    try:
        import json
        with open(artifact_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)  # Clean up temp file
        logger.info(f"Logged training config to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log training config: {e}")


def log_dataset_artifact(file_path: str, artifact_path: str = "data"):
    """
    Log dataset file to MLflow.
    
    Args:
        file_path: Path to dataset file
        artifact_path: Artifact path in MLflow
    """
    try:
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
        logger.info(f"Logged dataset artifact: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to log dataset artifact: {e}")
