import streamlit as st
from components.sidebar import render_sidebar
from components.loaders import load_all_metrics
from components.charts import plot_accuracy_comparison, plot_training_curves

st.set_page_config(page_title="Tổng Quan", page_icon="📊", layout="wide")

# Sidebar
dataset, optimizer, checkpoint = render_sidebar()

# Main content
st.title("📊 Tổng Quan: SAM vs SGD")

st.markdown("""
### Vấn Đề: Generalization và Độ Ổn Định

Các mô hình deep learning thường gặp phải:
- **Overfitting**: Accuracy training cao nhưng hiệu suất test kém
- **Sharp minima**: Các nghiệm nhạy cảm với các nhiễu nhỏ
- **Generalization kém**: Khoảng cách lớn giữa accuracy train và test

**SAM (Sharpness-Aware Minimization)** giải quyết các vấn đề này bằng cách:
- Tìm các minima phẳng hơn để generalize tốt hơn
- Giảm generalization gap
- Cải thiện test accuracy trong khi vẫn duy trì train performance
""")

st.markdown("---")

# Load metrics
metrics = load_all_metrics(dataset)

if metrics["SGD"].get("test_accuracy", 0) == 0 and metrics["SAM"].get("test_accuracy", 0) == 0:
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng đảm bảo các file metrics.json đã được điền kết quả.")
else:
    # Final accuracy comparison
    st.header("🎯 So Sánh Accuracy Cuối Cùng")
    st.markdown("So sánh train và test accuracy cuối cùng của SGD vs SAM:")
    
    fig_accuracy = plot_accuracy_comparison(metrics["SGD"], metrics["SAM"])
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    # Sử dụng best metrics từ cùng epoch tốt nhất
    train_acc_sgd = metrics["SGD"].get("best_train_accuracy", metrics["SGD"].get("train_accuracy", [0])[-1] if metrics["SGD"].get("train_accuracy") else 0)
    test_acc_sgd = metrics["SGD"].get("test_accuracy", 0)
    train_acc_sam = metrics["SAM"].get("best_train_accuracy", metrics["SAM"].get("train_accuracy", [0])[-1] if metrics["SAM"].get("train_accuracy") else 0)
    test_acc_sam = metrics["SAM"].get("test_accuracy", 0)
    
    gap_sgd = train_acc_sgd - test_acc_sgd
    gap_sam = train_acc_sam - test_acc_sam
    
    with col1:
        st.metric("SGD Test Accuracy", f"{test_acc_sgd:.3f}")
        st.metric("SGD Generalization Gap", f"{gap_sgd:.3f}", delta=f"{gap_sgd:.3f}")
    
    with col2:
        st.metric("SAM Test Accuracy", f"{test_acc_sam:.3f}", 
                 delta=f"{test_acc_sam - test_acc_sgd:.3f}" if test_acc_sam > test_acc_sgd else None)
        st.metric("SAM Generalization Gap", f"{gap_sam:.3f}", 
                 delta=f"{gap_sam - gap_sgd:.3f}" if gap_sam < gap_sgd else None,
                 delta_color="inverse")
    
    with col3:
        improvement = ((test_acc_sam - test_acc_sgd) / test_acc_sgd * 100) if test_acc_sgd > 0 else 0
        st.metric("Improvement", f"{improvement:.2f}%", 
                 delta=f"{test_acc_sam - test_acc_sgd:.3f}")
        gap_reduction = ((gap_sgd - gap_sam) / gap_sgd * 100) if gap_sgd > 0 else 0
        st.metric("Gap Reduction", f"{gap_reduction:.2f}%")
    
    st.markdown("---")
    
    # Training curves
    st.header("📈 Đường Cong Training")
    st.markdown("Quan sát cách loss và accuracy thay đổi trong quá trình training:")
    
    fig_curves = plot_training_curves(metrics["SGD"], metrics["SAM"])
    st.plotly_chart(fig_curves, use_container_width=True)
    
    # Summary
    st.markdown("---")
    st.header("💡 Điểm Quan Trọng")
    
    if test_acc_sam > test_acc_sgd:
        st.success(f"✅ **SAM đạt test accuracy cao hơn {test_acc_sam - test_acc_sgd:.3f}** so với SGD")
    else:
        st.info("📊 So sánh các metrics ở trên để xem sự khác biệt")
    
    if gap_sam < gap_sgd:
        st.success(f"✅ **SAM giảm generalization gap {gap_sgd - gap_sam:.3f}** so với SGD")
    else:
        st.info("📊 Kiểm tra các metrics generalization gap ở trên")





