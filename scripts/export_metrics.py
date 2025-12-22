#!/usr/bin/env python3
"""
Export metrics từ CSV log files sang JSON format cho Streamlit app.

Script này đọc CSV files từ training notebooks và convert sang format JSON
mà Streamlit app sử dụng.

Usage:
    python scripts/export_metrics.py --csv-path data/logs/Log[WRN-28-10_CIFAR-10-SAM].csv --dataset CIFAR-10 --optimizer SAM
"""
import pandas as pd
import json
import argparse
from pathlib import Path


def convert_csv_to_metrics(csv_path: str, dataset: str, optimizer: str, output_dir: str = None):
    """
    Convert CSV log file sang metrics.json format.
    
    Args:
        csv_path: Đường dẫn đến file CSV log
        dataset: Tên dataset (CIFAR-10 hoặc CIFAR-100)
        optimizer: Tên optimizer (SAM hoặc SGD)
        output_dir: Thư mục output (mặc định: experiments/{dataset}/{optimizer})
    
    Returns:
        Đường dẫn đến file metrics.json đã tạo
    """
    # Đọc CSV
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")
    
    import sys
    import io
    # Fix encoding for Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Kiểm tra các cột cần thiết
    required_columns = ['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV missing columns: {missing_columns}. Available columns: {list(df.columns)}")
    
    print(f"Read {len(df)} rows from CSV")
    
    # Aggregate: Nếu có nhiều dòng cho cùng 1 Epoch, lấy giá trị trung bình
    df_agg = df.groupby('Epoch', as_index=False).mean()
    
    # Sort theo Epoch để đảm bảo thứ tự
    df_agg = df_agg.sort_values('Epoch')
    
    print(f"After aggregation: {len(df_agg)} epochs")
    
    # Convert sang format JSON
    # Lưu ý: Train_Acc và Test_Acc trong CSV đã là tỷ lệ (0-1), KHÔNG cần chia 100
    
    # Tìm epoch có Test_Acc cao nhất (best epoch)
    best_test_acc = df_agg['Test_Acc'].max()
    best_epoch_idx = df_agg['Test_Acc'].idxmax()
    best_epoch = int(df_agg.loc[best_epoch_idx, 'Epoch'])
    
    # Lấy tất cả metrics từ best epoch
    best_train_loss = float(df_agg.loc[best_epoch_idx, 'Train_Loss'])
    best_train_acc = float(df_agg.loc[best_epoch_idx, 'Train_Acc'])
    best_val_loss = float(df_agg.loc[best_epoch_idx, 'Test_Loss'])
    best_val_acc = float(df_agg.loc[best_epoch_idx, 'Test_Acc'])
    
    metrics = {
        "train_loss": [round(float(x), 4) for x in df_agg['Train_Loss'].tolist()],
        "train_accuracy": [round(float(x), 4) for x in df_agg['Train_Acc'].tolist()],
        "val_loss": [round(float(x), 4) for x in df_agg['Test_Loss'].tolist()],
        "val_accuracy": [round(float(x), 4) for x in df_agg['Test_Acc'].tolist()],
        # Best metrics từ cùng epoch có test accuracy cao nhất
        "test_accuracy": round(best_val_acc, 4),
        "best_train_accuracy": round(best_train_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_train_loss": round(best_train_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_epoch": best_epoch,
        "epochs": int(df_agg['Epoch'].max()) + 1  # Epoch bắt đầu từ 0
    }
    
    # Xác định output directory
    if output_dir is None:
        base_path = Path(__file__).parent.parent
        dataset_lower = dataset.lower().replace("-", "")
        optimizer_lower = optimizer.lower()
        output_dir = base_path / "experiments" / dataset_lower / optimizer_lower
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu metrics.json
    output_path = output_dir / "metrics.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Export successful!")
    print(f"   Input CSV: {csv_path}")
    print(f"   Output JSON: {output_path}")
    print(f"   Dataset: {dataset}, Optimizer: {optimizer}")
    print(f"   Epochs: {metrics['epochs']}")
    print(f"   Best epoch: {metrics['best_epoch']}")
    print(f"   Best test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   Best train accuracy: {metrics['best_train_accuracy']:.4f}")
    print(f"   Best val accuracy: {metrics['best_val_accuracy']:.4f}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Export metrics từ CSV log files sang JSON format'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Đường dẫn đến file CSV log (ví dụ: data/logs/Log[WRN-28-10_CIFAR-10-SAM].csv)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['CIFAR-10', 'CIFAR-100'],
        help='Tên dataset'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        required=True,
        choices=['SAM', 'SGD'],
        help='Tên optimizer'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Thư mục output (mặc định: experiments/{dataset}/{optimizer})'
    )
    
    args = parser.parse_args()
    
    try:
        convert_csv_to_metrics(
            csv_path=args.csv_path,
            dataset=args.dataset,
            optimizer=args.optimizer,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

