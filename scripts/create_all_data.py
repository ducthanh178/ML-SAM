#!/usr/bin/env python3
"""Quick script to generate all sample data files."""
import json
import math
import os
from pathlib import Path

def generate_curves(num_epochs, final_train_val, final_test_val, loss_scale=1.0):
    """Generate training curves."""
    epochs = list(range(1, num_epochs + 1))
    train_loss = [round(loss_scale * (2.0 * math.exp(-i/30) + 0.25), 3) for i in epochs]
    train_acc = [round(final_train_val * (1 - math.exp(-i/25)), 3) for i in epochs]
    val_loss = [round(loss_scale * (2.2 * math.exp(-i/30) + 0.35), 3) for i in epochs]
    val_acc = [round(final_test_val * (1 - math.exp(-i/25)), 3) for i in epochs]
    return train_loss, train_acc, val_loss, val_acc

def create_predictions_simple(num_samples=100, num_classes=10, is_sam=True):
    """Create simple predictions."""
    import random
    random.seed(42 if is_sam else 43)
    preds, labels, confs = [], [], []
    for _ in range(num_samples):
        label = random.randint(0, num_classes-1)
        labels.append(label)
        conf = [random.random() * 0.05 for _ in range(num_classes)]
        conf[label] = (0.65 if is_sam else 0.85) + random.random() * 0.25
        total = sum(conf)
        conf = [round(c/total, 4) for c in conf]
        confs.append(conf)
        preds.append(conf.index(max(conf)))
    return {"predictions": preds, "true_labels": labels, "confidences": confs}

base = Path(__file__).parent.parent
experiments = base / "experiments"

# CIFAR-10 SAM: train=0.978, test=0.968 (small gap)
os.makedirs(experiments / "cifar10" / "sam", exist_ok=True)
tl, ta, vl, va = generate_curves(100, 0.978, 0.968, 1.0)
with open(experiments / "cifar10" / "sam" / "metrics.json", "w") as f:
    json.dump({"train_loss": tl, "train_accuracy": ta, "val_loss": vl, 
               "val_accuracy": va, "test_accuracy": 0.968, "epochs": 100}, f, indent=2)
with open(experiments / "cifar10" / "sam" / "predictions.json", "w") as f:
    json.dump(create_predictions_simple(100, 10, True), f, indent=2)

# CIFAR-10 SGD: train=0.995, test=0.955 (large gap - overfitting)
os.makedirs(experiments / "cifar10" / "sgd", exist_ok=True)
tl, ta, vl, va = generate_curves(100, 0.995, 0.955, 1.0)
with open(experiments / "cifar10" / "sgd" / "metrics.json", "w") as f:
    json.dump({"train_loss": tl, "train_accuracy": ta, "val_loss": vl,
               "val_accuracy": va, "test_accuracy": 0.955, "epochs": 100}, f, indent=2)
with open(experiments / "cifar10" / "sgd" / "predictions.json", "w") as f:
    json.dump(create_predictions_simple(100, 10, False), f, indent=2)

# CIFAR-100 SAM: train=0.835, test=0.812 (small gap)
os.makedirs(experiments / "cifar100" / "sam", exist_ok=True)
tl, ta, vl, va = generate_curves(100, 0.835, 0.812, 1.2)
with open(experiments / "cifar100" / "sam" / "metrics.json", "w") as f:
    json.dump({"train_loss": tl, "train_accuracy": ta, "val_loss": vl,
               "val_accuracy": va, "test_accuracy": 0.812, "epochs": 100}, f, indent=2)
with open(experiments / "cifar100" / "sam" / "predictions.json", "w") as f:
    json.dump(create_predictions_simple(100, 100, True), f, indent=2)

# CIFAR-100 SGD: train=0.865, test=0.785 (large gap - overfitting)
os.makedirs(experiments / "cifar100" / "sgd", exist_ok=True)
tl, ta, vl, va = generate_curves(100, 0.865, 0.785, 1.2)
with open(experiments / "cifar100" / "sgd" / "metrics.json", "w") as f:
    json.dump({"train_loss": tl, "train_accuracy": ta, "val_loss": vl,
               "val_accuracy": va, "test_accuracy": 0.785, "epochs": 100}, f, indent=2)
with open(experiments / "cifar100" / "sgd" / "predictions.json", "w") as f:
    json.dump(create_predictions_simple(100, 100, False), f, indent=2)

print("Created all JSON files successfully!")



