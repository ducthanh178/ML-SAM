# Hướng Dẫn Nhanh

## Export Dữ Liệu Từ CSV (CIFAR-10 và CIFAR-100)

Project sử dụng dữ liệu thật từ training trên Colab. Để export từ CSV logs:

1. **Đặt CSV files vào `data/logs/`:**
   - `cifar10_sam.csv` (hoặc `Log[WRN-28-10_CIFAR-10-SAM].csv`)
   - `cifar10_sgd.csv` (hoặc `Log[WRN-28-10_CIFAR-10-SGD].csv`)
   - `cifar100_sam.csv` (hoặc `Log[WRN-28-10_CIFAR-100-SAM].csv`)
   - `cifar100_sgd.csv` (hoặc `Log[WRN-28-10_CIFAR-100-SGD].csv`)

2. **Export sang JSON:**
```bash
python scripts/export_metrics.py --csv-path data/logs/cifar10_sam.csv --dataset CIFAR-10 --optimizer SAM
python scripts/export_metrics.py --csv-path data/logs/cifar10_sgd.csv --dataset CIFAR-10 --optimizer SGD
python scripts/export_metrics.py --csv-path data/logs/cifar100_sam.csv --dataset CIFAR-100 --optimizer SAM
python scripts/export_metrics.py --csv-path data/logs/cifar100_sgd.csv --dataset CIFAR-100 --optimizer SGD
```

**Lưu ý:** MNIST giữ nguyên dữ liệu hiện tại (train local).

## Chạy Ứng Dụng

```bash
streamlit run app/app.py
```





