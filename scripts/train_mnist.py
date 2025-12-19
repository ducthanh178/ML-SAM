"""
Training script cho MNIST digit recognition với SAM và SGD.
Script này train 2 models: một với SAM optimizer và một với SGD.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import MNISTNet
from core.sam import SAM


def train_epoch(model, train_loader, optimizer, criterion, device, use_sam=False):
    """Train một epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_sam:
            # SAM training: 2-step process
            # Cần forward + backward trước để có gradients cho first_step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Closure để tính loss tại w + epsilon
            closure_loss = None
            closure_outputs = None
            def closure():
                nonlocal closure_loss, closure_outputs
                optimizer.zero_grad()
                closure_outputs = model(inputs)
                closure_loss = criterion(closure_outputs, targets)
                closure_loss.backward()
                return closure_loss
            
            optimizer.step(closure)
            # Sử dụng loss và outputs từ closure (tại w + epsilon)
            if closure_loss is not None:
                loss = closure_loss
                outputs = closure_outputs
        else:
            # Standard SGD training
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model trên test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def generate_predictions(model, test_loader, device, num_samples=100):
    """Generate predictions trên test set."""
    model.eval()
    predictions = []
    true_labels = []
    confidences = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            for i in range(inputs.size(0)):
                if len(predictions) >= num_samples:
                    break
                
                pred_prob = probs[i].cpu().numpy().tolist()
                pred_class = np.argmax(pred_prob)
                
                predictions.append(int(pred_class))
                true_labels.append(int(targets[i].item()))
                confidences.append(pred_prob)
            
            if len(predictions) >= num_samples:
                break
    
    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "confidences": confidences
    }


def train_mnist(optimizer_name="SGD", epochs=10, batch_size=128, rho=0.05, lr=0.1):
    """
    Train MNIST model với optimizer được chỉ định.
    
    Args:
        optimizer_name: "SAM" hoặc "SGD"
        epochs: Số epochs
        batch_size: Batch size
        rho: SAM rho parameter (chỉ dùng cho SAM)
        lr: Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with optimizer: {optimizer_name}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Model
    model = MNISTNet().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "SAM":
        base_optimizer = optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=lr,
            momentum=0.9,
            rho=rho
        )
        use_sam = True
    else:  # SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        use_sam = False
    
    # Training
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_test_acc = 0.0
    
    print(f"\nStarting training...")
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, use_sam=use_sam
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc / 100.0)  # Convert to 0-1 range
        test_losses.append(test_loss)
        test_accs.append(test_acc / 100.0)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    # Save model
    base_path = Path(__file__).parent.parent
    save_dir = base_path / "experiments" / "mnist" / optimizer_name.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    metrics = {
        "train_loss": [round(l, 4) for l in train_losses],
        "train_accuracy": [round(a, 4) for a in train_accs],
        "val_loss": [round(l, 4) for l in test_losses],
        "val_accuracy": [round(a, 4) for a in test_accs],
        "test_accuracy": round(test_accs[-1], 4),
        "epochs": epochs
    }
    
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader, device, num_samples=100)
    
    predictions_path = save_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {predictions_path}")
    
    print(f"\n✅ Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    
    return model, metrics, predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MNIST với SAM hoặc SGD')
    parser.add_argument('--optimizer', type=str, choices=['SAM', 'SGD'], 
                       default='SGD', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--rho', type=float, default=0.05, 
                       help='SAM rho parameter (only for SAM)')
    parser.add_argument('--train-both', action='store_true', 
                       help='Train both SAM and SGD')
    
    args = parser.parse_args()
    
    if args.train_both:
        print("=" * 60)
        print("Training SGD model...")
        print("=" * 60)
        train_mnist("SGD", args.epochs, args.batch_size, args.rho, args.lr)
        
        print("\n" + "=" * 60)
        print("Training SAM model...")
        print("=" * 60)
        train_mnist("SAM", args.epochs, args.batch_size, args.rho, args.lr)
    else:
        train_mnist(args.optimizer, args.epochs, args.batch_size, args.rho, args.lr)

