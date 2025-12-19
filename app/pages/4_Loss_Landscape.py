import streamlit as st
import numpy as np
from components.sidebar import render_sidebar
from components.loaders import load_loss_surface
from components.charts import plot_loss_landscape

st.set_page_config(page_title="Loss Landscape", page_icon="🏔️", layout="wide")

# Sidebar
dataset, optimizer, checkpoint = render_sidebar()

# Main content
st.title("🏔️ Loss Landscape")

st.markdown("""
### Hiểu Về Sharp vs Flat Minima

**Loss landscape** visualize hình dạng của loss function xung quanh nghiệm:
- **Sharp minima**: Các thung lũng dốc, nhạy cảm với các nhiễu
- **Flat minima**: Các thung lũng rộng, nông, ổn định hơn

**SGD** thường tìm thấy sharp minima, trong khi **SAM** tìm các minima phẳng hơn để generalize tốt hơn.
""")

st.markdown("---")

# Load loss surfaces
loss_surface_sgd = load_loss_surface(dataset, "SGD")
loss_surface_sam = load_loss_surface(dataset, "SAM")

if loss_surface_sgd.size == 0 and loss_surface_sam.size == 0:
    st.warning("⚠️ Chưa có dữ liệu loss surface. Vui lòng đảm bảo các file loss_surface.npy tồn tại trong thư mục experiments.")
else:
    # Display both landscapes side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🔴 SGD Loss Landscape")
        if loss_surface_sgd.size > 0:
            fig_sgd = plot_loss_landscape(loss_surface_sgd, "SGD Loss Landscape")
            if fig_sgd:
                st.plotly_chart(fig_sgd, use_container_width=True)
            
            # Statistics
            st.subheader("Thống Kê")
            st.metric("Min Loss", f"{np.min(loss_surface_sgd):.4f}")
            st.metric("Max Loss", f"{np.max(loss_surface_sgd):.4f}")
            st.metric("Mean Loss", f"{np.mean(loss_surface_sgd):.4f}")
            st.metric("Std Loss", f"{np.std(loss_surface_sgd):.4f}")
            
            # Calculate sharpness (approximate as std or range)
            sharpness_sgd = np.std(loss_surface_sgd)
            st.metric("Sharpness (std)", f"{sharpness_sgd:.4f}")
        else:
            st.info("Chưa có dữ liệu cho SGD")
    
    with col2:
        st.header("🔵 SAM Loss Landscape")
        if loss_surface_sam.size > 0:
            fig_sam = plot_loss_landscape(loss_surface_sam, "SAM Loss Landscape")
            if fig_sam:
                st.plotly_chart(fig_sam, use_container_width=True)
            
            # Statistics
            st.subheader("Thống Kê")
            st.metric("Min Loss", f"{np.min(loss_surface_sam):.4f}")
            st.metric("Max Loss", f"{np.max(loss_surface_sam):.4f}")
            st.metric("Mean Loss", f"{np.mean(loss_surface_sam):.4f}")
            st.metric("Std Loss", f"{np.std(loss_surface_sam):.4f}")
            
            # Calculate sharpness
            sharpness_sam = np.std(loss_surface_sam)
            st.metric("Sharpness (std)", f"{sharpness_sam:.4f}")
        else:
            st.info("Chưa có dữ liệu cho SAM")
    
    st.markdown("---")
    
    # Comparison
    if loss_surface_sgd.size > 0 and loss_surface_sam.size > 0:
        st.header("📊 So Sánh")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sharpness_diff = sharpness_sgd - sharpness_sam
            st.metric(
                "Sharpness Difference",
                f"{sharpness_diff:.4f}",
                delta=f"{sharpness_diff:.4f}",
                delta_color="inverse" if sharpness_diff > 0 else "normal",
                help="Sharpness thấp hơn cho thấy minima phẳng hơn"
            )
        
        with col2:
            min_diff = np.min(loss_surface_sgd) - np.min(loss_surface_sam)
            st.metric(
                "Min Loss Difference",
                f"{min_diff:.4f}",
                delta=f"{min_diff:.4f}",
                help="Khác biệt trong giá trị loss tối thiểu"
            )
        
        with col3:
            range_sgd = np.max(loss_surface_sgd) - np.min(loss_surface_sgd)
            range_sam = np.max(loss_surface_sam) - np.min(loss_surface_sam)
            range_diff = range_sgd - range_sam
            st.metric(
                "Loss Range Difference",
                f"{range_diff:.4f}",
                delta=f"{range_diff:.4f}",
                help="Khác biệt trong phạm vi giá trị loss"
            )
        
        st.markdown("---")
        
        # Visual comparison with side-by-side 2D slices if 3D
        if loss_surface_sgd.ndim >= 2 and loss_surface_sam.ndim >= 2:
            st.header("🔍 2D Slice Comparison")
            st.markdown("So sánh các 2D slices qua loss landscape:")
            
            # Take middle slice
            if loss_surface_sgd.ndim == 2:
                slice_sgd = loss_surface_sgd
                slice_sam = loss_surface_sam if loss_surface_sam.ndim == 2 else loss_surface_sam.flatten().reshape(-1, 1)
            else:
                mid_idx = loss_surface_sgd.shape[0] // 2
                slice_sgd = loss_surface_sgd[mid_idx, :, :] if loss_surface_sgd.ndim == 3 else loss_surface_sgd
                mid_idx_sam = loss_surface_sam.shape[0] // 2 if loss_surface_sam.ndim >= 3 else 0
                slice_sam = loss_surface_sam[mid_idx_sam, :, :] if loss_surface_sam.ndim == 3 else loss_surface_sam
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig_slices = make_subplots(
                rows=1, cols=2,
                subplot_titles=("SGD 2D Slice", "SAM 2D Slice"),
                specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
            )
            
            fig_slices.add_trace(
                go.Heatmap(
                    z=slice_sgd,
                    colorscale="Viridis",
                    showscale=True,
                    name="SGD"
                ),
                row=1, col=1
            )
            
            fig_slices.add_trace(
                go.Heatmap(
                    z=slice_sam,
                    colorscale="Viridis",
                    showscale=True,
                    name="SAM"
                ),
                row=1, col=2
            )
            
            fig_slices.update_layout(
                height=500,
                title_text="2D Loss Landscape Slices",
                title_x=0.5
            )
            
            st.plotly_chart(fig_slices, use_container_width=True)
        
        st.markdown("---")
        
        # Key insights
        st.header("💡 Điểm Quan Trọng")
        
        if sharpness_sam < sharpness_sgd:
            st.success(f"""
            ✅ **SAM tìm thấy flatter minima** (sharpness: {sharpness_sam:.4f} vs {sharpness_sgd:.4f})
            
            Flatter minima ổn định hơn với các nhiễu và generalize tốt hơn với dữ liệu chưa thấy.
            Đây là nguyên lý cốt lõi đằng sau improved performance của SAM.
            """)
        else:
            st.info("📊 So sánh các sharpness metrics ở trên để xem sự khác biệt trong loss landscape geometry.")
        
        st.info("""
        📚 **Hiểu Về Visualization:**
        - **Steep valleys** (sharpness cao) = Sharp minima = Poor generalization
        - **Wide valleys** (sharpness thấp) = Flat minima = Better generalization
        - SAM explicitly optimizes cho flatter minima, dẫn đến better test performance
        """)





