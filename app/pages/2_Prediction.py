import streamlit as st
import numpy as np
from components.sidebar import render_sidebar
from components.loaders import load_all_predictions
from components.charts import plot_confidence_comparison, plot_confidence_stability

st.set_page_config(page_title="Prediction Stability", page_icon="🎯", layout="wide")

# Sidebar
dataset, optimizer, checkpoint = render_sidebar()

# Main content
st.title("🎯 Prediction Stability")

st.markdown("""
### Understanding Prediction Confidence

This page visualizes how **SGD** and **SAM** differ in their prediction confidence:
- **SGD**: Often produces overconfident predictions with sharp confidence distributions
- **SAM**: Tends to have more stable and calibrated confidence scores

Compare the confidence distributions and stability across multiple samples.
""")

st.markdown("---")

# Load predictions
predictions = load_all_predictions(dataset)

if not predictions["SGD"].get("confidences") or not predictions["SAM"].get("confidences"):
    st.warning("⚠️ No prediction data available. Please ensure predictions.json files are populated.")
else:
    # Sample selector
    num_samples = len(predictions["SGD"].get("confidences", []))
    if num_samples > 0:
        sample_idx = st.slider(
            "Select Sample Index",
            min_value=0,
            max_value=num_samples - 1,
            value=0,
            help="Choose which sample to visualize"
        )
        
        st.markdown("---")
        
        # Confidence comparison for selected sample
        st.header("📊 Prediction Confidence Comparison")
        st.markdown(f"Comparing confidence distributions for sample {sample_idx}:")
        
        fig_confidence = plot_confidence_comparison(
            predictions["SGD"], 
            predictions["SAM"], 
            sample_idx
        )
        
        if fig_confidence:
            st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Show prediction details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SGD Prediction")
                conf_sgd = predictions["SGD"]["confidences"][sample_idx]
                pred_sgd = predictions["SGD"].get("predictions", [0])[sample_idx] if predictions["SGD"].get("predictions") else 0
                true_label = predictions["SGD"].get("true_labels", [0])[sample_idx] if predictions["SGD"].get("true_labels") else 0
                
                if isinstance(conf_sgd, list):
                    max_conf_sgd = max(conf_sgd)
                    pred_class_sgd = np.argmax(conf_sgd)
                else:
                    max_conf_sgd = conf_sgd[0] if len(conf_sgd) > 0 else 0
                    pred_class_sgd = pred_sgd
                
                st.metric("Predicted Class", pred_class_sgd)
                st.metric("True Class", true_label)
                st.metric("Max Confidence", f"{max_conf_sgd:.3f}")
                st.metric("Correct", "✅" if pred_class_sgd == true_label else "❌")
            
            with col2:
                st.subheader("SAM Prediction")
                conf_sam = predictions["SAM"]["confidences"][sample_idx]
                pred_sam = predictions["SAM"].get("predictions", [0])[sample_idx] if predictions["SAM"].get("predictions") else 0
                
                if isinstance(conf_sam, list):
                    max_conf_sam = max(conf_sam)
                    pred_class_sam = np.argmax(conf_sam)
                else:
                    max_conf_sam = conf_sam[0] if len(conf_sam) > 0 else 0
                    pred_class_sam = pred_sam
                
                st.metric("Predicted Class", pred_class_sam)
                st.metric("True Class", true_label)
                st.metric("Max Confidence", f"{max_conf_sam:.3f}")
                st.metric("Correct", "✅" if pred_class_sam == true_label else "❌")
        
        st.markdown("---")
        
        # Confidence stability
        st.header("📈 Confidence Stability Across Samples")
        st.markdown("Compare how confidence varies across multiple samples:")
        
        num_samples_plot = st.slider(
            "Number of Samples to Display",
            min_value=10,
            max_value=min(100, num_samples),
            value=min(20, num_samples),
            help="Select how many samples to include in the stability plot"
        )
        
        fig_stability = plot_confidence_stability(
            predictions["SGD"],
            predictions["SAM"],
            num_samples_plot
        )
        
        if fig_stability:
            st.plotly_chart(fig_stability, use_container_width=True)
            
            # Calculate statistics
            confidences_sgd = predictions["SGD"]["confidences"][:num_samples_plot]
            confidences_sam = predictions["SAM"]["confidences"][:num_samples_plot]
            
            max_conf_sgd = [max(conf) if isinstance(conf, list) else conf[0] if len(conf) > 0 else 0 
                           for conf in confidences_sgd]
            max_conf_sam = [max(conf) if isinstance(conf, list) else conf[0] if len(conf) > 0 else 0 
                           for conf in confidences_sam]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("SGD Mean Confidence", f"{np.mean(max_conf_sgd):.3f}")
                st.metric("SGD Std Confidence", f"{np.std(max_conf_sgd):.3f}")
            
            with col2:
                st.metric("SAM Mean Confidence", f"{np.mean(max_conf_sam):.3f}")
                st.metric("SAM Std Confidence", f"{np.std(max_conf_sam):.3f}")
            
            with col3:
                mean_diff = np.mean(max_conf_sam) - np.mean(max_conf_sgd)
                std_diff = np.std(max_conf_sam) - np.std(max_conf_sgd)
                st.metric("Mean Difference", f"{mean_diff:.3f}")
                st.metric("Std Difference", f"{std_diff:.3f}", 
                         delta=f"{std_diff:.3f}" if std_diff < 0 else None,
                         delta_color="inverse")
            
            st.info("💡 **Lower standard deviation indicates more stable predictions.** SAM typically shows lower variance in confidence scores.")





