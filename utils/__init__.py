import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
def get_figure_size(default_size=None):
    viz_type = st.session_state.get("current_visualization", "default")
    size_mappings = {
        "small": {
            "default": (5, 3.5),
            "wide": (6, 3),
            "complex_function": (4.5, 4.5),
            "harmonic_flow_wide": (6, 3),
            "conformal_map_wide": (6, 3),
            "integral_contour": (4.5, 4.5),
        },
        "medium": {
            "default": (7, 5),
            "wide": (8, 4),
            "complex_function": (6, 6),
            "harmonic_flow_wide": (8, 4),
            "conformal_map_wide": (8, 4),
            "integral_contour": (6, 6),
        },
        "large": {
            "default": (9, 7),
            "wide": (10, 5),
            "complex_function": (8, 8),
            "harmonic_flow_wide": (10, 5),
            "conformal_map_wide": (10, 5),
            "integral_contour": (8, 8),
        }
    }
    if "default_image_size" in st.session_state:
        size_preference = st.session_state.get("default_image_size", "medium")
        st.session_state["visualization_size"] = size_preference
    else:
        size_preference = st.session_state.get("visualization_size", "medium")
    if size_preference not in size_mappings:
        size_preference = "medium"
    if viz_type in size_mappings[size_preference]:
        return size_mappings[size_preference][viz_type]
    else:
        if default_size:
            return default_size
        return size_mappings[size_preference]["default"]
def create_viz_container(title=None):
    container = st.container()
    with container:
        if title:
            st.markdown(f"<h3 style='color: var(--neon-blue); margin-bottom: 10px; font-size: 1.3rem;'>{title}</h3>", 
                        unsafe_allow_html=True)
        st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
    return container
def close_viz_container():
    st.markdown("</div>", unsafe_allow_html=True)
def apply_image_container(fig, caption=None, max_height=None):
    if max_height is None:
        max_height = st.session_state.get("max_viz_height", 400)
    with st.container():
        st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='plot-container' style='max-height: {max_height}px;'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        if caption:
            st.markdown(f"<div class='viz-caption'>{caption}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    plt.close(fig)
def create_3d_toggle():
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col2:
            show_3d = st.checkbox("Show 3D visualization", value=False, key=f"3d_toggle_{st.session_state.get('current_visualization', 'default')}")
        with col1:
            if show_3d:
                st.info("3D visualization will allow you to explore the function's magnitude as a surface. "
                        "Drag to rotate, scroll to zoom, and double-click to reset view.")
    return show_3d
def load_plotly_requirements():
    try:
        import plotly
        return True
    except ImportError:
        st.error("Plotly is required for 3D visualizations but is not installed.")
        st.code("pip install plotly", language="bash")
        return False
def create_animation_container(title=None, description=None):
    container = st.container()
    with container:
        if title:
            st.markdown(f"<h3 style='color: var(--neon-green); margin-bottom: 10px; font-size: 1.3rem;'>{title}</h3>", 
                        unsafe_allow_html=True)
        if description:
            st.markdown(f"<p style='color: var(--text-secondary); margin-bottom: 10px;'>{description}</p>", 
                        unsafe_allow_html=True)
        st.markdown("<div class='viz-container animated-border'>", unsafe_allow_html=True)
    return container
def close_animation_container():
    st.markdown("</div>", unsafe_allow_html=True) 

