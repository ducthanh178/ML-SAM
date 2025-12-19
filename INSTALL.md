# Hướng Dẫn Cài Đặt

## Bước 1: Cài đặt Dependencies

Cài đặt các thư viện cần thiết để chạy ứng dụng Streamlit:

```bash
pip install -r requirements.txt
```

Hoặc cài từng package:

```bash
pip install streamlit plotly numpy
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

## Tóm Tắt Dependencies

### Bắt buộc (để chạy app):
- ✅ `streamlit` - Web framework
- ✅ `plotly` - Interactive charts
- ✅ `numpy` - Numerical operations (cần cho loss surface)

### Tùy chọn:
- `numpy` - Chỉ cần nếu muốn tạo loss_surface.npy files

## Kiểm Tra Cài Đặt

Chạy lệnh sau để kiểm tra:

```bash
python -c "import streamlit, plotly, numpy; print('✅ All packages installed!')"
```

Nếu có lỗi, cài lại:
```bash
pip install --upgrade streamlit plotly numpy
```





