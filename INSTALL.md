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

## Bước 2: Export Dữ Liệu Từ CSV (CIFAR-10 và CIFAR-100)

Project sử dụng dữ liệu thật từ training trên Colab. Để export:

1. **Đặt CSV files vào `data/logs/`** (từ notebooks training trên Colab)

2. **Export sang JSON:**
```bash
python scripts/export_metrics.py --csv-path data/logs/cifar10_sam.csv --dataset CIFAR-10 --optimizer SAM
python scripts/export_metrics.py --csv-path data/logs/cifar10_sgd.csv --dataset CIFAR-10 --optimizer SGD
python scripts/export_metrics.py --csv-path data/logs/cifar100_sam.csv --dataset CIFAR-100 --optimizer SAM
python scripts/export_metrics.py --csv-path data/logs/cifar100_sgd.csv --dataset CIFAR-100 --optimizer SGD
```

**Lưu ý:** 
- Cần `pandas` để export: `pip install pandas`
- MNIST giữ nguyên dữ liệu hiện tại (train local)

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





