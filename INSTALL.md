# Hướng Dẫn Cài Đặt

## Bước 1: Cài đặt Dependencies

Cài đặt các thư viện cần thiết để chạy ứng dụng Streamlit:

```bash
pip install -r requirements.txt
```

Hoặc cài từng package:

```bash
pip install streamlit plotly numpy torch torchvision Pillow
```

## Bước 2: Tạo Dữ Liệu Mẫu

### Tạo file JSON (metrics và predictions):

```bash
python scripts/create_all_data.py
```

Script này chỉ cần Python standard library (json, math, os, pathlib) - không cần cài thêm gì.

### Tạo file loss_surface.npy (tùy chọn):

Để có visualization loss landscape, cần numpy (đã có trong requirements.txt):

```bash
python scripts/generate_sample_data.py
```

## Bước 3: Chạy Ứng Dụng

```bash
streamlit run app/app.py
```

Ứng dụng sẽ mở trong trình duyệt tại `http://localhost:8501`

## Bước 4: Train Models cho MNIST (Tùy chọn - cho tính năng Digit Recognition)

Để sử dụng tính năng nhận diện chữ số viết tay, cần train models trước:

```bash
python scripts/train_mnist.py --train-both --epochs 10
```

Xem chi tiết trong `scripts/TRAIN_MNIST.md`.

## Tóm Tắt Dependencies

### Bắt buộc (để chạy app):
- ✅ `streamlit` - Web framework
- ✅ `plotly` - Interactive charts
- ✅ `numpy` - Numerical operations (cần cho loss surface)

### Cần thiết cho tính năng Digit Recognition:
- ✅ `torch` - PyTorch framework
- ✅ `torchvision` - Datasets và transforms
- ✅ `Pillow` - Image processing

**⚠️ Lưu ý về PyTorch trên Windows:**
Nếu gặp lỗi `OSError: [WinError 126]` khi import torch, xem hướng dẫn chi tiết trong `INSTALL_PYTORCH.md`.
Khuyến nghị: Cài CPU-only version (nhẹ hơn, ít dependencies):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Tùy chọn:
- `numpy` - Chỉ cần nếu muốn tạo loss_surface.npy files

## Kiểm Tra Cài Đặt

Chạy lệnh sau để kiểm tra:

```bash
python -c "import streamlit, plotly, numpy, torch, torchvision, PIL; print('✅ All packages installed!')"
```

Nếu có lỗi, cài lại:
```bash
pip install --upgrade streamlit plotly numpy torch torchvision Pillow
```





