"""
Script to generate sample data for the Streamlit demo.
Creates realistic training curves and predictions for SAM vs SGD comparison.
"""
import json
import numpy as np
from pathlib import Path


def generate_metrics(num_epochs=100, optimizer="SAM", dataset="CIFAR-10"):
    """Generate realistic training metrics."""
    if dataset == "CIFAR-10":
        if optimizer == "SAM":
            # SAM: better generalization, smaller gap
            final_train_acc = 0.978
            final_test_acc = 0.968
            final_train_loss = 0.25
            final_val_loss = 0.35
        else:  # SGD
            # SGD: higher train acc but lower test acc (overfitting)
            final_train_acc = 0.995
            final_test_acc = 0.955
            final_train_loss = 0.15
            final_val_loss = 0.50
    else:  # CIFAR-100
        if optimizer == "SAM":
            final_train_acc = 0.835
            final_test_acc = 0.812
            final_train_loss = 0.85
            final_val_loss = 1.15
        else:  # SGD
            final_train_acc = 0.865
            final_test_acc = 0.785
            final_train_loss = 0.70
            final_val_loss = 1.40
    
    # Generate training curves
    epochs = np.arange(1, num_epochs + 1)
    
    # Train loss: exponential decay
    train_loss = final_train_loss + (2.0 - final_train_loss) * np.exp(-epochs / 30)
    
    # Train accuracy: sigmoid growth
    train_accuracy = final_train_acc * (1 - np.exp(-epochs / 25))
    
    # Val loss: similar but slightly higher
    val_loss = final_val_loss + (2.2 - final_val_loss) * np.exp(-epochs / 30)
    
    # Val accuracy: similar but slightly lower
    val_accuracy = final_test_acc * (1 - np.exp(-epochs / 25))
    
    return {
        "train_loss": train_loss.tolist(),
        "train_accuracy": train_accuracy.tolist(),
        "val_loss": val_loss.tolist(),
        "val_accuracy": val_accuracy.tolist(),
        "test_accuracy": float(final_test_acc),
        "epochs": num_epochs
    }


def generate_predictions(num_samples=100, optimizer="SAM", dataset="CIFAR-10"):
    """Generate sample predictions with confidences."""
    num_classes = 10 if dataset == "CIFAR-10" else 100
    
    predictions = []
    true_labels = []
    confidences = []
    
    np.random.seed(42 if optimizer == "SAM" else 43)
    
    for i in range(num_samples):
        true_label = np.random.randint(0, num_classes)
        true_labels.append(int(true_label))
        
        # Generate confidence distribution
        if optimizer == "SAM":
            # SAM: more calibrated, less overconfident
            conf = np.random.dirichlet(np.ones(num_classes) * 2.0)
            # Make prediction more confident but not too much
            conf[true_label] += np.random.beta(3, 2) * 0.3
            conf = conf / conf.sum()
        else:  # SGD
            # SGD: more overconfident
            conf = np.random.dirichlet(np.ones(num_classes) * 1.5)
            # Make prediction very confident
            conf[true_label] += np.random.beta(5, 1) * 0.5
            conf = conf / conf.sum()
        
        pred = int(np.argmax(conf))
        predictions.append(pred)
        confidences.append(conf.tolist())
    
    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "confidences": confidences
    }


def generate_loss_surface(optimizer="SAM"):
    """Generate a 2D loss surface."""
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    if optimizer == "SAM":
        # SAM: flatter minima (wider valley)
        Z = 0.5 * (X**2 + Y**2) + 0.1 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        Z = Z + 0.05 * (X**2 + Y**2)**2
    else:  # SGD
        # SGD: sharper minima (steeper valley)
        Z = 0.8 * (X**2 + Y**2) + 0.3 * np.sin(3 * np.pi * X) * np.sin(3 * np.pi * Y)
        Z = Z + 0.2 * (X**2 + Y**2)**2
    
    # Normalize
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * 2.0 + 0.3
    
    return Z


def main():
    """Generate all sample data files."""
    base_path = Path(__file__).parent.parent
    
    datasets = ["CIFAR-10", "CIFAR-100"]
    optimizers = ["SAM", "SGD"]
    
    for dataset in datasets:
        for optimizer in optimizers:
            dataset_lower = dataset.lower().replace("-", "")
            optimizer_lower = optimizer.lower()
            
            exp_dir = base_path / "experiments" / dataset_lower / optimizer_lower
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate metrics
            metrics = generate_metrics(optimizer=optimizer, dataset=dataset)
            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Generated {metrics_path}")
            
            # Generate predictions
            predictions = generate_predictions(optimizer=optimizer, dataset=dataset)
            predictions_path = exp_dir / "predictions.json"
            with open(predictions_path, "w") as f:
                json.dump(predictions, f, indent=2)
            print(f"Generated {predictions_path}")
            
            # Generate loss surface
            loss_surface = generate_loss_surface(optimizer=optimizer)
            loss_surface_path = exp_dir / "loss_surface.npy"
            np.save(loss_surface_path, loss_surface)
            print(f"Generated {loss_surface_path}")
    
    print("\n✅ All sample data generated successfully!")


if __name__ == "__main__":
    main()





