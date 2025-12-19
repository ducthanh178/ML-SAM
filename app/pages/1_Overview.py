import streamlit as st
from components.sidebar import render_sidebar
from components.loaders import load_all_metrics
from components.charts import plot_accuracy_comparison, plot_training_curves

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# Sidebar
dataset, optimizer, checkpoint = render_sidebar()

# Main content
st.title("📊 Overview: SAM vs SGD")

st.markdown("""
### Problem: Generalization and Stability

Deep learning models often suffer from:
- **Overfitting**: High training accuracy but poor test performance
- **Sharp minima**: Solutions that are sensitive to small perturbations
- **Poor generalization**: Large gap between train and test accuracy

**SAM (Sharpness-Aware Minimization)** addresses these issues by:
- Finding flatter minima that generalize better
- Reducing the generalization gap
- Improving test accuracy while maintaining train performance
""")

st.markdown("---")

# Load metrics
metrics = load_all_metrics(dataset)

if metrics["SGD"].get("test_accuracy", 0) == 0 and metrics["SAM"].get("test_accuracy", 0) == 0:
    st.warning("⚠️ No data available. Please ensure metrics.json files are populated with results.")
else:
    # Final accuracy comparison
    st.header("🎯 Final Accuracy Comparison")
    st.markdown("Compare the final train and test accuracy of SGD vs SAM:")
    
    fig_accuracy = plot_accuracy_comparison(metrics["SGD"], metrics["SAM"])
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    train_acc_sgd = metrics["SGD"].get("train_accuracy", [0])[-1] if metrics["SGD"].get("train_accuracy") else 0
    test_acc_sgd = metrics["SGD"].get("test_accuracy", 0)
    train_acc_sam = metrics["SAM"].get("train_accuracy", [0])[-1] if metrics["SAM"].get("train_accuracy") else 0
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
    st.header("📈 Training Curves")
    st.markdown("Observe how loss and accuracy evolve during training:")
    
    fig_curves = plot_training_curves(metrics["SGD"], metrics["SAM"])
    st.plotly_chart(fig_curves, use_container_width=True)
    
    # Summary
    st.markdown("---")
    st.header("💡 Key Takeaways")
    
    if test_acc_sam > test_acc_sgd:
        st.success(f"✅ **SAM achieves {test_acc_sam - test_acc_sgd:.3f} higher test accuracy** than SGD")
    else:
        st.info("📊 Compare the metrics above to see the differences")
    
    if gap_sam < gap_sgd:
        st.success(f"✅ **SAM reduces generalization gap by {gap_sgd - gap_sam:.3f}** compared to SGD")
    else:
        st.info("📊 Check the generalization gap metrics above")





