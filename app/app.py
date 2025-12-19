import streamlit as st

st.set_page_config(
    page_title="So Sánh SAM vs SGD",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 SAM vs SGD: Demo So Sánh Trực Quan")
st.markdown("""
### Sharpness-Aware Minimization vs Stochastic Gradient Descent

Demo tương tác này so sánh các optimizer **SAM** (Sharpness-Aware Minimization) và **SGD** 
dựa trên kết quả đã tính toán trước từ các thí nghiệm CIFAR-10 và CIFAR-100.

**Sử dụng sidebar** để khám phá các khía cạnh khác nhau của so sánh:
- 📊 **Tổng Quan**: Độ chính xác cuối cùng và đường cong training
- 🎯 **Độ Ổn Định Dự Đoán**: Phân phối confidence và độ ổn định
- 📉 **Generalization Gap**: So sánh train vs test accuracy
- 🏔️ **Loss Landscape**: Visualization 3D của loss surfaces
- ✍️ **Nhận Diện Chữ Số**: Nhận diện chữ số MNIST tương tác với so sánh SAM vs SGD

Tất cả dữ liệu được tải từ các file local (JSON, NPY) - không thực hiện training. 
Đối với nhận diện chữ số, cần train models trước (xem `scripts/TRAIN_MNIST.md`).
""")

st.markdown("---")

# Quick stats if available
try:
    from components.loaders import load_all_metrics
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CIFAR-10")
        try:
            metrics_c10 = load_all_metrics("CIFAR-10")
            if metrics_c10["SGD"].get("test_accuracy", 0) > 0:
                st.metric("SGD Test Acc", f"{metrics_c10['SGD']['test_accuracy']:.3f}")
                st.metric("SAM Test Acc", f"{metrics_c10['SAM']['test_accuracy']:.3f}")
            else:
                st.info("Chưa có dữ liệu")
        except:
            st.info("Chưa có dữ liệu")
    
    with col2:
        st.subheader("CIFAR-100")
        try:
            metrics_c100 = load_all_metrics("CIFAR-100")
            if metrics_c100["SGD"].get("test_accuracy", 0) > 0:
                st.metric("SGD Test Acc", f"{metrics_c100['SGD']['test_accuracy']:.3f}")
                st.metric("SAM Test Acc", f"{metrics_c100['SAM']['test_accuracy']:.3f}")
            else:
                st.info("Chưa có dữ liệu")
        except:
            st.info("Chưa có dữ liệu")
except:
    pass

st.markdown("---")

st.info("""
💡 **Mẹo**: Sử dụng sidebar để chọn các dataset và optimizer khác nhau. 
Các visualization sẽ tự động cập nhật để hiển thị sự khác biệt giữa SAM và SGD.
""")





