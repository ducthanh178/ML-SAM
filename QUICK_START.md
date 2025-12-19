# Hướng Dẫn Nhanh

## Tạo Dữ Liệu Mẫu

Chạy script sau để tạo tất cả dữ liệu mẫu:

```bash
python scripts/create_all_data.py
```

Nếu Python không có trong PATH, thử:
- `py scripts/create_all_data.py`
- `python3 scripts/create_all_data.py`
- Hoặc tìm đường dẫn đầy đủ đến python.exe

## Chạy Ứng Dụng

```bash
streamlit run app/app.py
```

## Cấu Trúc Dữ Liệu

Script sẽ tạo các file sau:
- `experiments/cifar10/sam/metrics.json` và `predictions.json`
- `experiments/cifar10/sgd/metrics.json` và `predictions.json`
- `experiments/cifar100/sam/metrics.json` và `predictions.json`
- `experiments/cifar100/sgd/metrics.json` và `predictions.json`

Để có loss landscape visualization, cần thêm:
- `experiments/*/sam/loss_surface.npy`
- `experiments/*/sgd/loss_surface.npy`

Chạy: `python scripts/generate_sample_data.py` (cần numpy)





