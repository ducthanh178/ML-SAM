import streamlit as st

st.set_page_config(
    page_title="SAM vs SGD Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 SAM vs SGD: Visual Comparison Demo")
st.markdown("""
### Sharpness-Aware Minimization vs Stochastic Gradient Descent

This interactive demo compares **SAM** (Sharpness-Aware Minimization) and **SGD** optimizers
based on pre-computed results from CIFAR-10 and CIFAR-100 experiments.

**Navigate using the sidebar** to explore different aspects of the comparison:
- 📊 **Overview**: Final accuracy and training curves
- 🎯 **Prediction Stability**: Confidence distributions and stability
- 📉 **Generalization Gap**: Train vs test accuracy comparison
- 🏔️ **Loss Landscape**: 3D visualization of loss surfaces
- ✍️ **Digit Recognition**: Interactive MNIST digit recognition với SAM vs SGD comparison

All data is loaded from local files (JSON, NPY) - no training is performed. For digit recognition, models need to be trained first (see `scripts/TRAIN_MNIST.md`).
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
                st.info("No data available")
        except:
            st.info("No data available")
    
    with col2:
        st.subheader("CIFAR-100")
        try:
            metrics_c100 = load_all_metrics("CIFAR-100")
            if metrics_c100["SGD"].get("test_accuracy", 0) > 0:
                st.metric("SGD Test Acc", f"{metrics_c100['SGD']['test_accuracy']:.3f}")
                st.metric("SAM Test Acc", f"{metrics_c100['SAM']['test_accuracy']:.3f}")
            else:
                st.info("No data available")
        except:
            st.info("No data available")
except:
    pass

st.markdown("---")

st.info("""
💡 **Tip**: Use the sidebar to select different datasets and optimizers. 
The visualizations will update automatically to show the differences between SAM and SGD.
""")





