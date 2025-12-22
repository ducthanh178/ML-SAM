import streamlit as st
from components.sidebar import render_sidebar
from components.loaders import load_all_metrics
from components.charts import plot_generalization_gap, plot_training_curves

st.set_page_config(page_title="Khoảng Cách Generalization", page_icon="📉", layout="wide")

# Sidebar
dataset, optimizer, checkpoint = render_sidebar()

# Main content
st.title("📉 Generalization Gap")

st.markdown("""
### Hiểu Về Overfitting

**Generalization gap** là sự khác biệt giữa accuracy training và test:
- **Gap lớn**: Model overfit với dữ liệu training
- **Gap nhỏ**: Model generalize tốt với dữ liệu chưa thấy

**SGD** thường cho thấy generalization gap lớn hơn, cho thấy overfitting.
**SAM** thường giảm gap này bằng cách tìm các minima phẳng hơn để generalize tốt hơn.
""")

st.markdown("---")

# Load metrics
metrics = load_all_metrics(dataset)

if metrics["SGD"].get("test_accuracy", 0) == 0 and metrics["SAM"].get("test_accuracy", 0) == 0:
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng đảm bảo các file metrics.json đã được điền kết quả.")
else:
    # Generalization gap visualization
    st.header("🎯 So Sánh Train vs Test Accuracy")
    st.markdown("Visualize khoảng cách generalization cho cả 2 optimizers:")
    
    fig_gap = plot_generalization_gap(metrics["SGD"], metrics["SAM"])
    st.plotly_chart(fig_gap, use_container_width=True)
    
    # Calculate gaps - sử dụng best metrics từ cùng epoch tốt nhất
    train_acc_sgd = metrics["SGD"].get("best_train_accuracy", metrics["SGD"].get("train_accuracy", [0])[-1] if metrics["SGD"].get("train_accuracy") else 0)
    test_acc_sgd = metrics["SGD"].get("test_accuracy", 0)
    train_acc_sam = metrics["SAM"].get("best_train_accuracy", metrics["SAM"].get("train_accuracy", [0])[-1] if metrics["SAM"].get("train_accuracy") else 0)
    test_acc_sam = metrics["SAM"].get("test_accuracy", 0)
    
    gap_sgd = train_acc_sgd - test_acc_sgd
    gap_sam = train_acc_sam - test_acc_sam
    gap_reduction = gap_sgd - gap_sam
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SGD Train Acc", f"{train_acc_sgd:.3f}")
    
    with col2:
        st.metric("SGD Test Acc", f"{test_acc_sgd:.3f}")
    
    with col3:
        st.metric("SAM Train Acc", f"{train_acc_sam:.3f}")
    
    with col4:
        st.metric("SAM Test Acc", f"{test_acc_sam:.3f}")
    
    st.markdown("---")
    
    # Gap comparison
    st.header("📊 Phân Tích Generalization Gap")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "SGD Gap",
            f"{gap_sgd:.3f}",
            delta=f"{gap_sgd:.3f}",
            delta_color="inverse",
            help="Khác biệt giữa train và test accuracy"
        )
        if gap_sgd > 0.1:
            st.error("⚠️ Gap lớn cho thấy overfitting")
        elif gap_sgd > 0.05:
            st.warning("⚠️ Gap vừa phải")
        else:
            st.success("✅ Gap nhỏ")
    
    with col2:
        st.metric(
            "SAM Gap",
            f"{gap_sam:.3f}",
            delta=f"{gap_sam:.3f}",
            delta_color="inverse",
            help="Khác biệt giữa train và test accuracy"
        )
        if gap_sam > 0.1:
            st.error("⚠️ Gap lớn cho thấy overfitting")
        elif gap_sam > 0.05:
            st.warning("⚠️ Gap vừa phải")
        else:
            st.success("✅ Gap nhỏ")
    
    with col3:
        gap_reduction_pct = (gap_reduction / gap_sgd * 100) if gap_sgd > 0 else 0
        st.metric(
            "Gap Reduction",
            f"{gap_reduction:.3f}",
            delta=f"{gap_reduction_pct:.2f}%",
            help="SAM giảm gap bao nhiêu so với SGD"
        )
        if gap_reduction > 0:
            st.success(f"✅ SAM giảm gap {gap_reduction:.3f}")
        else:
            st.info("📊 So sánh các gap ở trên")
    
    st.markdown("---")
    
    # Training curves with gap visualization
    st.header("📈 Tiến Trình Training: Sự Tiến Hóa của Generalization Gap")
    st.markdown("Quan sát cách generalization gap thay đổi trong quá trình training:")
    
    # Calculate gap per epoch if available
    train_accs_sgd = metrics["SGD"].get("train_accuracy", [])
    val_accs_sgd = metrics["SGD"].get("val_accuracy", [])
    train_accs_sam = metrics["SAM"].get("train_accuracy", [])
    val_accs_sam = metrics["SAM"].get("val_accuracy", [])
    
    if train_accs_sgd and val_accs_sgd and train_accs_sam and val_accs_sam:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        epochs_sgd = list(range(1, len(train_accs_sgd) + 1))
        epochs_sam = list(range(1, len(train_accs_sam) + 1))
        
        gaps_sgd = [train_accs_sgd[i] - val_accs_sgd[i] for i in range(min(len(train_accs_sgd), len(val_accs_sgd)))]
        gaps_sam = [train_accs_sam[i] - val_accs_sam[i] for i in range(min(len(train_accs_sam), len(val_accs_sam)))]
        
        fig_gaps = go.Figure()
        
        fig_gaps.add_trace(go.Scatter(
            x=epochs_sgd[:len(gaps_sgd)],
            y=gaps_sgd,
            mode="lines+markers",
            name="SGD Gap",
            line=dict(color="#FF6B6B", width=2),
            marker=dict(size=6)
        ))
        
        fig_gaps.add_trace(go.Scatter(
            x=epochs_sam[:len(gaps_sam)],
            y=gaps_sam,
            mode="lines+markers",
            name="SAM Gap",
            line=dict(color="#4ECDC4", width=2),
            marker=dict(size=6)
        ))
        
        fig_gaps.update_layout(
            title="Generalization Gap Trong Quá Trình Training",
            title_x=0.5,
            xaxis_title="Epoch",
            yaxis_title="Gap (Train - Val Accuracy)",
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_gaps, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.header("💡 Điểm Quan Trọng")
    
    if gap_reduction > 0:
        st.success(f"""
        ✅ **SAM giảm generalization gap {gap_reduction:.3f}** ({gap_reduction_pct:.2f}%)
        
        Điều này cho thấy SAM tìm được các nghiệm generalize tốt hơn với dữ liệu chưa thấy,
        giảm overfitting so với SGD.
        """)
    else:
        st.info("📊 So sánh các generalization gap ở trên để xem sự khác biệt giữa SGD và SAM.")
    
    if gap_sgd > gap_sam:
        st.info(f"""
        📊 **SGD có gap lớn hơn {gap_sgd - gap_sam:.3f}** so với SAM.
        
        Điều này cho thấy SGD dễ bị overfitting hơn, ghi nhớ dữ liệu training
        thay vì học các pattern có thể generalize.
        """)





