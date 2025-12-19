# Tạo Dữ Liệu Mẫu

Để ứng dụng Streamlit có thể chạy, bạn cần tạo các file dữ liệu mẫu.

## Cách 1: Chạy Script Python (Khuyến nghị)

```bash
python scripts/create_all_data.py
```

Script này sẽ tạo tất cả các file JSON cần thiết:
- `experiments/cifar10/sam/metrics.json`
- `experiments/cifar10/sam/predictions.json`
- `experiments/cifar10/sgd/metrics.json`
- `experiments/cifar10/sgd/predictions.json`
- `experiments/cifar100/sam/metrics.json`
- `experiments/cifar100/sam/predictions.json`
- `experiments/cifar100/sgd/metrics.json`
- `experiments/cifar100/sgd/predictions.json`

## Cách 2: Tạo Loss Surface Files (NPY)

Để có visualization loss landscape, bạn cần numpy:

```bash
pip install numpy
python scripts/generate_sample_data.py
```

## Cấu Trúc Dữ Liệu

### metrics.json
```json
{
  "train_loss": [list of 100 values],
  "train_accuracy": [list of 100 values],
  "val_loss": [list of 100 values],
  "val_accuracy": [list of 100 values],
  "test_accuracy": 0.968,
  "epochs": 100
}
```

### predictions.json
```json
{
  "predictions": [list of 100 integers],
  "true_labels": [list of 100 integers],
  "confidences": [list of 100 arrays, each with num_classes values]
}
```

### loss_surface.npy
- NumPy array shape (50, 50) hoặc tương tự
- 2D loss surface để visualize

## Lưu Ý

- Tất cả các file đã được tạo sẵn với dữ liệu mẫu hợp lý
- Nếu bạn có dữ liệu thực từ training, thay thế các file này
- Dữ liệu mẫu dựa trên kết quả thực tế của WRN-28-10 trên CIFAR-10/100





