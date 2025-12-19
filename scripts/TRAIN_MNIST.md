# Hướng Dẫn Train Models cho MNIST

## Cách 1: Train cả 2 models (SAM và SGD) cùng lúc

```bash
python scripts/train_mnist.py --train-both --epochs 10
```

## Cách 2: Train từng model riêng

### Train với SGD:
```bash
python scripts/train_mnist.py --optimizer SGD --epochs 10 --batch-size 128 --lr 0.1
```

### Train với SAM:
```bash
python scripts/train_mnist.py --optimizer SAM --epochs 10 --batch-size 128 --lr 0.1 --rho 0.05
```

## Parameters

- `--optimizer`: `SAM` hoặc `SGD` (mặc định: `SGD`)
- `--epochs`: Số epochs để train (mặc định: 10)
- `--batch-size`: Batch size (mặc định: 128)
- `--lr`: Learning rate (mặc định: 0.1)
- `--rho`: SAM rho parameter, chỉ dùng cho SAM (mặc định: 0.05)
- `--train-both`: Train cả 2 models cùng lúc

## Output

Sau khi train xong, các file sau sẽ được tạo:

```
experiments/mnist/
  sam/
    model.pth           # Model checkpoint
    metrics.json        # Training metrics
    predictions.json    # Predictions trên test set
  sgd/
    model.pth
    metrics.json
    predictions.json
```

## Lưu ý

- MNIST dataset sẽ tự động được download vào thư mục `./data/` lần đầu chạy
- Training trên GPU sẽ nhanh hơn nhiều (nếu có CUDA)
- Với 10 epochs, training thường mất vài phút trên CPU, vài giây trên GPU
- Để có kết quả tốt hơn, có thể train với nhiều epochs hơn (20-30)

## Kiểm tra models đã train

Sau khi train, chạy Streamlit app và vào trang "Digit Recognition" để test:

```bash
streamlit run app/app.py
```

