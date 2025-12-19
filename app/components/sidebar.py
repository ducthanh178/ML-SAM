import streamlit as st


def render_sidebar():
    """Render sidebar với các options để chọn dataset và optimizer."""
    st.sidebar.title("🔬 SAM vs SGD Demo")
    st.sidebar.markdown("---")
    
    # Dataset selector
    dataset = st.sidebar.selectbox(
        "📊 Dataset",
        options=["CIFAR-10", "CIFAR-100"],
        index=0,
        help="Chọn dataset để xem kết quả so sánh"
    )
    
    st.sidebar.markdown("---")
    
    # Optimizer selector
    optimizer = st.sidebar.selectbox(
        "⚙️ Optimizer",
        options=["SAM", "SGD"],
        index=0,
        help="Chọn optimizer để xem chi tiết (hoặc so sánh cả 2 ở các trang khác)"
    )
    
    st.sidebar.markdown("---")
    
    # Checkpoint selector (optional, có thể để empty hoặc None)
    checkpoint = st.sidebar.text_input(
        "📁 Checkpoint (Optional)",
        value="",
        help="Nhập tên checkpoint nếu có (để trống nếu không dùng)"
    )
    
    st.sidebar.markdown("---")
    
    # Info
    st.sidebar.info("""
    **Hướng dẫn:**
    - Chọn dataset và optimizer từ menu trên
    - Các trang sẽ tự động cập nhật dữ liệu
    - Dữ liệu được load từ thư mục `experiments/`
    """)
    
    return dataset, optimizer, checkpoint

