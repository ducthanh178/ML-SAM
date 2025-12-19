# Simple script to create sample data
# Run with: python scripts/create_data.py
import json
import math

def create_metrics_sam_cifar10():
    epochs = 100
    train_loss = [2.0 * math.exp(-i/30) + 0.25 for i in range(1, epochs+1)]
    train_acc = [0.978 * (1 - math.exp(-i/25)) for i in range(1, epochs+1)]
    val_loss = [2.2 * math.exp(-i/30) + 0.35 for i in range(1, epochs+1)]
    val_acc = [0.968 * (1 - math.exp(-i/25)) for i in range(1, epochs+1)]
    return {
        "train_loss": [round(x, 3) for x in train_loss],
        "train_accuracy": [round(x, 3) for x in train_acc],
        "val_loss": [round(x, 3) for x in val_loss],
        "val_accuracy": [round(x, 3) for x in val_acc],
        "test_accuracy": 0.968,
        "epochs": epochs
    }

def create_metrics_sgd_cifar10():
    epochs = 100
    train_loss = [2.0 * math.exp(-i/25) + 0.15 for i in range(1, epochs+1)]
    train_acc = [0.995 * (1 - math.exp(-i/20)) for i in range(1, epochs+1)]
    val_loss = [2.2 * math.exp(-i/25) + 0.50 for i in range(1, epochs+1)]
    val_acc = [0.955 * (1 - math.exp(-i/20)) for i in range(1, epochs+1)]
    return {
        "train_loss": [round(x, 3) for x in train_loss],
        "train_accuracy": [round(x, 3) for x in train_acc],
        "val_loss": [round(x, 3) for x in val_loss],
        "val_accuracy": [round(x, 3) for x in val_acc],
        "test_accuracy": 0.955,
        "epochs": epochs
    }

def create_metrics_sam_cifar100():
    epochs = 100
    train_loss = [2.5 * math.exp(-i/30) + 0.85 for i in range(1, epochs+1)]
    train_acc = [0.835 * (1 - math.exp(-i/25)) for i in range(1, epochs+1)]
    val_loss = [2.7 * math.exp(-i/30) + 1.15 for i in range(1, epochs+1)]
    val_acc = [0.812 * (1 - math.exp(-i/25)) for i in range(1, epochs+1)]
    return {
        "train_loss": [round(x, 3) for x in train_loss],
        "train_accuracy": [round(x, 3) for x in train_acc],
        "val_loss": [round(x, 3) for x in val_loss],
        "val_accuracy": [round(x, 3) for x in val_acc],
        "test_accuracy": 0.812,
        "epochs": epochs
    }

def create_metrics_sgd_cifar100():
    epochs = 100
    train_loss = [2.5 * math.exp(-i/25) + 0.70 for i in range(1, epochs+1)]
    train_acc = [0.865 * (1 - math.exp(-i/20)) for i in range(1, epochs+1)]
    val_loss = [2.7 * math.exp(-i/25) + 1.40 for i in range(1, epochs+1)]
    val_acc = [0.785 * (1 - math.exp(-i/20)) for i in range(1, epochs+1)]
    return {
        "train_loss": [round(x, 3) for x in train_loss],
        "train_accuracy": [round(x, 3) for x in train_acc],
        "val_loss": [round(x, 3) for x in val_loss],
        "val_accuracy": [round(x, 3) for x in val_acc],
        "test_accuracy": 0.785,
        "epochs": epochs
    }

def create_predictions(num_samples=100, num_classes=10, is_sam=True):
    import random
    random.seed(42 if is_sam else 43)
    predictions = []
    true_labels = []
    confidences = []
    
    for i in range(num_samples):
        true_label = random.randint(0, num_classes-1)
        true_labels.append(true_label)
        
        # Create confidence distribution
        conf = [random.random() * 0.1 for _ in range(num_classes)]
        if is_sam:
            # SAM: more calibrated
            conf[true_label] = 0.6 + random.random() * 0.3
        else:
            # SGD: more overconfident
            conf[true_label] = 0.8 + random.random() * 0.15
        
        # Normalize
        total = sum(conf)
        conf = [c/total for c in conf]
        confidences.append([round(c, 4) for c in conf])
        predictions.append(conf.index(max(conf)))
    
    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "confidences": confidences
    }

# Create all files
import os
from pathlib import Path

base = Path(__file__).parent.parent

# CIFAR-10 SAM
os.makedirs(base / "experiments" / "cifar10" / "sam", exist_ok=True)
with open(base / "experiments" / "cifar10" / "sam" / "metrics.json", "w") as f:
    json.dump(create_metrics_sam_cifar10(), f, indent=2)
with open(base / "experiments" / "cifar10" / "sam" / "predictions.json", "w") as f:
    json.dump(create_predictions(100, 10, True), f, indent=2)

# CIFAR-10 SGD
os.makedirs(base / "experiments" / "cifar10" / "sgd", exist_ok=True)
with open(base / "experiments" / "cifar10" / "sgd" / "metrics.json", "w") as f:
    json.dump(create_metrics_sgd_cifar10(), f, indent=2)
with open(base / "experiments" / "cifar10" / "sgd" / "predictions.json", "w") as f:
    json.dump(create_predictions(100, 10, False), f, indent=2)

# CIFAR-100 SAM
os.makedirs(base / "experiments" / "cifar100" / "sam", exist_ok=True)
with open(base / "experiments" / "cifar100" / "sam" / "metrics.json", "w") as f:
    json.dump(create_metrics_sam_cifar100(), f, indent=2)
with open(base / "experiments" / "cifar100" / "sam" / "predictions.json", "w") as f:
    json.dump(create_predictions(100, 100, True), f, indent=2)

# CIFAR-100 SGD
os.makedirs(base / "experiments" / "cifar100" / "sgd", exist_ok=True)
with open(base / "experiments" / "cifar100" / "sgd" / "metrics.json", "w") as f:
    json.dump(create_metrics_sgd_cifar100(), f, indent=2)
with open(base / "experiments" / "cifar100" / "sgd" / "predictions.json", "w") as f:
    json.dump(create_predictions(100, 100, False), f, indent=2)

print("✅ Created all JSON files!")
print("Note: loss_surface.npy files need numpy - run generate_sample_data.py for those")





