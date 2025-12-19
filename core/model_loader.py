"""
Functions để load và sử dụng trained models cho inference.
"""
import torch
from pathlib import Path
from core.models import MNISTNet
import streamlit as st


@st.cache_resource
def load_mnist_model(optimizer: str, device='cpu'):
    """
    Load trained MNIST model với caching.
    
    Args:
        optimizer: 'SAM' hoặc 'SGD'
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        Loaded model ở eval mode
    """
    base_path = Path(__file__).parent.parent
    model_path = base_path / "experiments" / "mnist" / optimizer.lower() / "model.pth"
    
    model = MNISTNet()
    
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle both dict format and direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi load model {optimizer}: {e}. Sử dụng model chưa train.")
    else:
        st.warning(f"⚠️ Model file không tồn tại: {model_path}. Sử dụng model chưa train.")
    
    return model.to(device)


def predict_digit(model, image_tensor, device='cpu'):
    """
    Predict chữ số từ ảnh đã preprocess.
    
    Args:
        model: Trained MNISTNet model
        image_tensor: Tensor shape (1, 1, 28, 28)
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        dict với:
            - prediction: int (0-9)
            - confidence: float
            - all_probs: list[float] (10 probabilities)
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'prediction': predicted.item(),
            'confidence': confidence.item(),
            'all_probs': probabilities[0].cpu().numpy().tolist()
        }


def compare_predictions_sam_vs_sgd(image_tensor, device='cpu'):
    """
    So sánh predictions từ cả 2 models (SAM và SGD).
    
    Args:
        image_tensor: Tensor shape (1, 1, 28, 28)
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        dict với predictions từ cả 2 models
    """
    model_sam = load_mnist_model('SAM', device)
    model_sgd = load_mnist_model('SGD', device)
    
    pred_sam = predict_digit(model_sam, image_tensor, device)
    pred_sgd = predict_digit(model_sgd, image_tensor, device)
    
    return {
        'SAM': pred_sam,
        'SGD': pred_sgd
    }

