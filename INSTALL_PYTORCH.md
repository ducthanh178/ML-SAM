# Hướng Dẫn Cài Đặt PyTorch trên Windows

Nếu gặp lỗi `OSError: [WinError 126] The specified module could not be found` khi import torch, làm theo các bước sau:

## Giải Pháp 1: Cài PyTorch CPU Version (Khuyến nghị - nhẹ hơn)

CPU version không cần CUDA và ít dependencies hơn:

```bash
# Gỡ bỏ version cũ (nếu có)
pip uninstall torch torchvision -y

# Cài CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Giải Pháp 2: Cài PyTorch với pip thông thường

```bash
pip uninstall torch torchvision -y
pip install torch torchvision
```

## Giải Pháp 3: Cài Visual C++ Redistributables

Nếu vẫn lỗi, có thể thiếu Visual C++ Redistributables:

1. Download **Microsoft Visual C++ Redistributable**:
   - Link: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Hoặc tìm "Visual C++ Redistributable" trên Microsoft website

2. Cài đặt và restart máy

3. Cài lại PyTorch:
   ```bash
   pip install torch torchvision
   ```

## Giải Pháp 4: Sử dụng Conda (nếu có)

Nếu bạn có Anaconda/Miniconda:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

## Kiểm Tra Cài Đặt

Sau khi cài xong, kiểm tra:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

Nếu không có lỗi → thành công! ✅

## Lưu Ý

- **CPU version** đủ cho training MNIST (chậm hơn GPU nhưng vẫn OK)
- Nếu có NVIDIA GPU và muốn dùng CUDA, cần cài CUDA toolkit trước
- Với MNIST dataset nhỏ, CPU version vẫn train được trong vài phút

## Nếu Vẫn Gặp Lỗi

1. Đảm bảo đang dùng virtual environment (venv)
2. Restart terminal/IDE sau khi cài
3. Kiểm tra Python version (nên dùng Python 3.8-3.11)
4. Thử tạo venv mới và cài lại từ đầu

