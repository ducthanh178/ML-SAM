"""
Utilities để preprocess ảnh cho MNIST digit recognition.
"""
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import streamlit as st


def preprocess_mnist_image(image, target_size=(28, 28)):
    """
    Preprocess ảnh để predict với MNIST model.
    
    Args:
        image: PIL Image hoặc numpy array
        target_size: Kích thước ảnh đầu vào (28x28 cho MNIST)
    
    Returns:
        Tensor đã preprocess, shape (1, 1, 28, 28)
    """
    # Convert numpy array to PIL Image nếu cần
    if isinstance(image, np.ndarray):
        # Nếu là ảnh RGB (3 channels), convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = Image.fromarray(image[:, :, :3]).convert('L')
            elif image.shape[2] == 3:  # RGB
                image = Image.fromarray(image).convert('L')
            else:
                image = Image.fromarray(image, mode='L')
        else:
            image = Image.fromarray(image, mode='L')
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Ensure grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Transform: resize, normalize với MNIST stats
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension: (1, 1, 28, 28)


def preprocess_canvas_image(canvas_image_data):
    """
    Preprocess ảnh từ streamlit-drawable-canvas.
    
    Args:
        canvas_image_data: Image data từ canvas (PIL Image)
    
    Returns:
        Tensor đã preprocess
    """
    if canvas_image_data is None:
        return None
    
    # Canvas thường trả về RGBA, cần invert và convert to grayscale
    if hasattr(canvas_image_data, 'image_data'):
        img = Image.fromarray(canvas_image_data.image_data)
    else:
        img = canvas_image_data
    
    # Invert colors (canvas có nền trắng, chữ đen; MNIST có nền đen, chữ trắng)
    if img.mode == 'RGBA':
        # Convert RGBA to grayscale, invert
        img_array = np.array(img.convert('L'))
        img_array = 255 - img_array  # Invert
        img = Image.fromarray(img_array, mode='L')
    elif img.mode == 'L':
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert
        img = Image.fromarray(img_array, mode='L')
    
    return preprocess_mnist_image(img)


def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess ảnh từ Streamlit file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Tensor đã preprocess
    """
    if uploaded_file is None:
        return None
    
    image = Image.open(uploaded_file)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Invert nếu ảnh có nền trắng (để match với MNIST format)
    # Có thể detect tự động dựa vào mean pixel value
    img_array = np.array(image)
    if np.mean(img_array) > 128:  # Nền sáng -> invert
        img_array = 255 - img_array
        image = Image.fromarray(img_array, mode='L')
    
    return preprocess_mnist_image(image)

