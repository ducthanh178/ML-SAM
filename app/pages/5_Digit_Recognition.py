import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Page config phải ở đầu nhất
st.set_page_config(page_title="Digit Recognition", page_icon="✍️", layout="wide")

# Add parent directory to path để import core modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Kiểm tra torch TRƯỚC khi import các modules khác
def check_torch_available():
    """Kiểm tra xem torch có sẵn và import được không."""
    try:
        import torch
        return True, None
    except Exception as e:
        return False, str(e)

torch_available, torch_error = check_torch_available()

if not torch_available:
    
    # Nếu torch không available, hiển thị warning ngay đầu trang
    st.error(f"""
    ⚠️ **PyTorch không thể import được!**
    
    Lỗi: `{torch_error}`
    
    **Giải pháp:**
    
    1. **Cài lại PyTorch (CPU version - khuyến nghị, nhẹ hơn):**
       ```bash
       pip uninstall torch torchvision -y
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       ```
    
    2. **Hoặc cài với pip thông thường:**
       ```bash
       pip uninstall torch torchvision -y
       pip install torch torchvision
       ```
    
    3. **Nếu vẫn lỗi, có thể cần cài Visual C++ Redistributables:**
       - Download từ: https://aka.ms/vs/17/release/vc_redist.x64.exe
       - Cài đặt và restart máy
       - Cài lại PyTorch
    
    📖 Xem chi tiết trong file `INSTALL_PYTORCH.md`
    """)
    
    st.info("""
    💡 **Lưu ý:** 
    - Bạn vẫn có thể sử dụng các tính năng khác của app (CIFAR-10/100 comparisons)
    - Chỉ tính năng Digit Recognition cần PyTorch
    """)
    st.stop()

# Nếu torch available, import các modules bình thường
try:
    from core.model_loader import compare_predictions_sam_vs_sgd, load_mnist_model, predict_digit
    from core.image_utils import preprocess_uploaded_image, preprocess_mnist_image
    from components.charts import plot_digit_prediction_comparison, plot_confidence_comparison_bars
except ImportError as e:
    st.error(f"❌ Lỗi import modules: {e}")
    st.stop()

st.title("✍️ Nhận Diện Chữ Số Viết Tay: SAM vs SGD")

st.markdown("""
### So sánh SAM và SGD trong Nhận Diện Chữ Số

Trang này so sánh cách **SAM** và **SGD** nhận diện chữ số viết tay:
- **Độ chính xác dự đoán**: Cả 2 model dự đoán số nào?
- **Confidence scores**: Model nào tự tin hơn?
- **Calibration**: Model nào có confidence tốt hơn (calibrated)?
- **Robustness**: Model nào ổn định hơn với input khác nhau?

**Upload ảnh hoặc sử dụng ảnh mẫu** để so sánh predictions của SAM và SGD.
""")

st.markdown("---")

# Input method selection
input_method = st.radio(
    "Chọn cách nhập ảnh:",
    options=["Upload ảnh", "Sử dụng ảnh mẫu từ MNIST"],
    horizontal=True
)

image_tensor = None
display_image = None

if input_method == "Upload ảnh":
    uploaded_file = st.file_uploader(
        "Upload ảnh chữ số (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload ảnh chữ số viết tay (0-9). Ảnh sẽ tự động được resize và convert sang grayscale."
    )
    
    if uploaded_file is not None:
        try:
            display_image = Image.open(uploaded_file)
            image_tensor = preprocess_uploaded_image(uploaded_file)
            st.success("✅ Ảnh đã được load thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý ảnh: {e}")

else:  # Sử dụng ảnh mẫu
    st.info("💡 Sử dụng ảnh mẫu từ MNIST test set (sẽ được implement sau) hoặc upload ảnh của bạn.")
    
    # Có thể thêm sample images từ MNIST sau
    sample_digit = st.selectbox(
        "Chọn chữ số mẫu (0-9):",
        options=list(range(10)),
        help="Chọn chữ số để xem prediction"
    )
    
    # TODO: Load sample image from MNIST test set
    st.warning("⚠️ Tính năng ảnh mẫu sẽ được thêm sau. Vui lòng upload ảnh của bạn.")

if image_tensor is not None:
    st.markdown("---")
    
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📷 Ảnh Input")
        if display_image:
            # Resize for display
            display_img_resized = display_image.resize((280, 280), Image.Resampling.LANCZOS)
            st.image(display_img_resized, caption="Ảnh đã upload", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Preprocessed Image")
        # Display preprocessed image (grayscale 28x28)
        if image_tensor is not None:
            img_array = image_tensor[0, 0].numpy()
            # Normalize để hiển thị
            img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            st.image(img_normalized, caption="28x28 grayscale (preprocessed)", use_container_width=True, clamp=True)
    
    st.markdown("---")
    
    # Predict với cả 2 models
    with st.spinner("🔄 Đang predict với SAM và SGD..."):
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            predictions = compare_predictions_sam_vs_sgd(image_tensor, device)
            
            pred_sam = predictions['SAM']
            pred_sgd = predictions['SGD']
            
            # Display results
            st.header("📊 Kết Quả So Sánh SAM vs SGD")
            
            # Main comparison metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "SGD Prediction",
                    f"**{pred_sgd['prediction']}**",
                    help="Chữ số được SGD dự đoán"
                )
                st.metric(
                    "SGD Confidence",
                    f"{pred_sgd['confidence']:.4f}",
                    delta=f"{pred_sgd['confidence']:.2%}",
                    help="Độ tự tin của SGD"
                )
            
            with col2:
                st.metric(
                    "SAM Prediction",
                    f"**{pred_sam['prediction']}**",
                    delta=f"{'✅ Same' if pred_sam['prediction'] == pred_sgd['prediction'] else '⚠️ Different'}",
                    help="Chữ số được SAM dự đoán"
                )
                st.metric(
                    "SAM Confidence",
                    f"{pred_sam['confidence']:.4f}",
                    delta=f"{pred_sam['confidence'] - pred_sgd['confidence']:.4f}",
                    delta_color="normal" if pred_sam['confidence'] > pred_sgd['confidence'] else "inverse",
                    help="Độ tự tin của SAM"
                )
            
            with col3:
                conf_diff = pred_sam['confidence'] - pred_sgd['confidence']
                st.metric(
                    "Confidence Difference",
                    f"{abs(conf_diff):.4f}",
                    delta=f"{conf_diff:+.4f}",
                    delta_color="normal" if conf_diff > 0 else "inverse",
                    help="SAM - SGD confidence difference"
                )
                
                # Prediction agreement
                agreement = "✅ Cùng prediction" if pred_sam['prediction'] == pred_sgd['prediction'] else "⚠️ Khác prediction"
                st.info(agreement)
            
            with col4:
                # Entropy (uncertainty measure - SAM thường có entropy cao hơn = calibrated hơn)
                entropy_sgd = -np.sum([p * np.log(p + 1e-10) for p in pred_sgd['all_probs']])
                entropy_sam = -np.sum([p * np.log(p + 1e-10) for p in pred_sam['all_probs']])
                
                st.metric(
                    "SGD Entropy",
                    f"{entropy_sgd:.4f}",
                    help="Entropy (uncertainty) - thấp = overconfident"
                )
                st.metric(
                    "SAM Entropy",
                    f"{entropy_sam:.4f}",
                    delta=f"{entropy_sam - entropy_sgd:+.4f}",
                    delta_color="normal" if entropy_sam > entropy_sgd else "inverse",
                    help="SAM thường có entropy cao hơn = calibrated hơn"
                )
            
            st.markdown("---")
            
            # Visualization 1: Confidence comparison bars
            st.subheader("📊 So Sánh Confidence Scores")
            fig_conf = plot_confidence_comparison_bars(pred_sam, pred_sgd)
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Insights
            if pred_sam['confidence'] > pred_sgd['confidence']:
                st.info("💡 **SAM có confidence cao hơn** - có thể cho thấy SAM tìm được solution tốt hơn.")
            elif pred_sam['confidence'] < pred_sgd['confidence']:
                st.info("💡 **SGD có confidence cao hơn** - nhưng điều này không nhất thiết tốt hơn (có thể overconfident).")
            else:
                st.info("💡 Cả 2 models có confidence tương đương.")
            
            st.markdown("---")
            
            # Visualization 2: Probability distributions
            st.subheader("📈 Phân Phối Xác Suất (Probability Distribution)")
            st.markdown("So sánh cách SAM và SGD phân bố xác suất cho 10 chữ số:")
            
            fig_dist = plot_digit_prediction_comparison(pred_sam, pred_sgd)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Detailed insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Đặc Điểm SGD")
                st.markdown(f"""
                - **Chữ số dự đoán**: {pred_sgd['prediction']}
                - **Confidence**: {pred_sgd['confidence']:.4f}
                - **Top 3 dự đoán**:
                  1. Chữ số {np.argsort(pred_sgd['all_probs'])[-1]}: {sorted(pred_sgd['all_probs'])[-1]:.4f}
                  2. Chữ số {np.argsort(pred_sgd['all_probs'])[-2]}: {sorted(pred_sgd['all_probs'])[-2]:.4f}
                  3. Chữ số {np.argsort(pred_sgd['all_probs'])[-3]}: {sorted(pred_sgd['all_probs'])[-3]:.4f}
                """)
                
                # Check if overconfident (very high confidence on wrong prediction)
                top_pred_sgd = np.argmax(pred_sgd['all_probs'])
                if pred_sgd['confidence'] > 0.95:
                    st.warning("⚠️ SGD có confidence rất cao - có thể là overconfident.")
            
            with col2:
                st.markdown("#### 🔵 Đặc Điểm SAM")
                st.markdown(f"""
                - **Chữ số dự đoán**: {pred_sam['prediction']}
                - **Confidence**: {pred_sam['confidence']:.4f}
                - **Top 3 dự đoán**:
                  1. Chữ số {np.argsort(pred_sam['all_probs'])[-1]}: {sorted(pred_sam['all_probs'])[-1]:.4f}
                  2. Chữ số {np.argsort(pred_sam['all_probs'])[-2]}: {sorted(pred_sam['all_probs'])[-2]:.4f}
                  3. Chữ số {np.argsort(pred_sam['all_probs'])[-3]}: {sorted(pred_sam['all_probs'])[-3]:.4f}
                """)
                
                # Check calibration
                if entropy_sam > entropy_sgd:
                    st.success("✅ SAM có entropy cao hơn - thường cho thấy calibration tốt hơn.")
            
            st.markdown("---")
            
            # Key takeaways
            st.header("💡 Điểm Quan Trọng: SAM vs SGD")
            
            if pred_sam['prediction'] == pred_sgd['prediction']:
                st.success(f"""
                ✅ **Cả 2 models dự đoán cùng chữ số: {pred_sam['prediction']}**
                
                So sánh confidence và entropy để xem model nào calibrated tốt hơn.
                """)
            else:
                st.warning(f"""
                ⚠️ **Các models dự đoán khác nhau:**
                - SGD: {pred_sgd['prediction']} (confidence: {pred_sgd['confidence']:.4f})
                - SAM: {pred_sam['prediction']} (confidence: {pred_sam['confidence']:.4f})
                
                Xem phân phối xác suất để hiểu tại sao.
                """)
            
            # General insights about SAM vs SGD
            st.info("""
            📚 **Tổng quan về SAM vs SGD trong Digit Recognition:**
            
            - **SAM (Sharpness-Aware Minimization)**: 
              - Tìm flatter minima → generalization tốt hơn
              - Confidence thường được calibrated tốt hơn
              - Ít overconfident hơn SGD
            
            - **SGD (Stochastic Gradient Descent)**:
              - Có thể tìm sharp minima → dễ overfit
              - Thường overconfident (confidence cao nhưng có thể sai)
              - Training accuracy cao nhưng test có thể kém hơn
            
            **Trong trường hợp này:**
            - Nếu SAM có entropy cao hơn → tốt hơn (calibrated)
            - Nếu cả 2 cùng prediction nhưng SAM confidence thấp hơn → có thể tốt hơn (không overconfident)
            """)
            
        except Exception as e:
            st.error(f"❌ Lỗi khi predict: {e}")
            st.exception(e)
            st.info("💡 Đảm bảo bạn đã train models và có file `experiments/mnist/sam/model.pth` và `experiments/mnist/sgd/model.pth`")

else:
    st.info("👆 Vui lòng upload ảnh hoặc chọn ảnh mẫu để bắt đầu nhận diện.")

st.markdown("---")

# Instructions
with st.expander("📖 Hướng Dẫn Sử Dụng"):
    st.markdown("""
    ### Cách sử dụng:
    1. **Upload ảnh chữ số**: Chọn file ảnh chứa chữ số viết tay (0-9)
       - Format: PNG, JPG, JPEG
       - Ảnh sẽ tự động được resize về 28x28 và convert sang grayscale
       - Nền trắng/chữ đen hoặc nền đen/chữ trắng đều được hỗ trợ
    
    2. **Xem kết quả so sánh**:
       - Prediction: Chữ số nào được dự đoán
       - Confidence: Độ tự tin (0-1)
       - Probability distribution: Phân phối xác suất cho 10 chữ số
       - Entropy: Độ uncertainty (SAM thường cao hơn = calibrated tốt hơn)
    
    3. **So sánh SAM vs SGD**:
       - Xem model nào dự đoán chính xác hơn
       - Xem model nào có confidence calibrated tốt hơn
       - Xem phân phối xác suất để hiểu sự khác biệt
    
    ### Lưu ý:
    - Models cần được train trước (chạy `scripts/train_mnist.py`)
    - Ảnh nên là chữ số viết tay rõ ràng, đơn lẻ
    - Background nên tương đối đồng nhất
    """)

