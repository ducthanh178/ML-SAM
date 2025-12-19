import json
import numpy as np
import os
from pathlib import Path


def get_data_path(dataset: str, optimizer: str, filename: str) -> Path:
    """Get path to data file based on dataset and optimizer."""
    base_path = Path(__file__).parent.parent.parent
    dataset_lower = dataset.lower().replace("-", "")
    optimizer_lower = optimizer.lower()
    return base_path / "experiments" / dataset_lower / optimizer_lower / filename


def load_metrics(dataset: str, optimizer: str) -> dict:
    """Load metrics from JSON file."""
    metrics_path = get_data_path(dataset, optimizer, "metrics.json")
    
    if not metrics_path.exists():
        return {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_accuracy": 0.0,
            "epochs": 0
        }
    
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_predictions(dataset: str, optimizer: str) -> dict:
    """Load predictions from JSON file."""
    predictions_path = get_data_path(dataset, optimizer, "predictions.json")
    
    if not predictions_path.exists():
        return {
            "predictions": [],
            "true_labels": [],
            "confidences": []
        }
    
    with open(predictions_path, "r") as f:
        return json.load(f)


def load_loss_surface(dataset: str, optimizer: str) -> np.ndarray:
    """Load loss surface from NPY file."""
    loss_surface_path = get_data_path(dataset, optimizer, "loss_surface.npy")
    
    if not loss_surface_path.exists():
        # Return empty array if file doesn't exist
        return np.array([])
    
    return np.load(loss_surface_path)


def load_all_metrics(dataset: str) -> dict:
    """Load metrics for both SGD and SAM."""
    return {
        "SGD": load_metrics(dataset, "SGD"),
        "SAM": load_metrics(dataset, "SAM")
    }


def load_all_predictions(dataset: str) -> dict:
    """Load predictions for both SGD and SAM."""
    return {
        "SGD": load_predictions(dataset, "SGD"),
        "SAM": load_predictions(dataset, "SAM")
    }





