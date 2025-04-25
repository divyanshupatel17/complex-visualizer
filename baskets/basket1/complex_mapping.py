import streamlit as st
import numpy as np
try:
    from models.complex_function_grapher import visualize_complex_function
except ImportError:
    def visualize_complex_function(func_str, domain_min, domain_max, resolution, show_colorwheel):
        st.error("Complex Function Grapher module not found. Please check your installation.")
        st.code("Error importing: models.complex_function_grapher")
try:
    from models.conformal_map_animator import visualize_conformal_map
except ImportError:
    def visualize_conformal_map(func_str, domain_min, domain_max, resolution):
        st.error("Conformal Map Animator module not found. Please check your installation.")
        st.code("Error importing: models.conformal_map_animator")
try:
    from models.harmonic_flow_predictor import visualize_harmonic_flow
except ImportError:
    def visualize_harmonic_flow(func_str, domain_min, domain_max, resolution):
        st.error("Harmonic Flow Predictor module not found. Please check your installation.")
        st.code("Error importing: models.harmonic_flow_predictor")
try:
    from models.integral_contour_interpreter import visualize_integral_contour
except ImportError:
    def visualize_integral_contour(func_str, domain_min, domain_max, resolution):
        st.error("Integral Contour Interpreter module not found. Please check your installation.")
        st.code("Error importing: models.integral_contour_interpreter")
COMPLEX_MODELS = {
    "complex visualizer": {
        "description": "Visualize analytic functions with domain coloring to explore their behavior",
        "examples": ["z**2", "z**3 - 1", "sin(z)", "exp(z)", "1/(z**2 + 1)", "z * log(z)"],
        "default": "z**2"
    },
    "Complex Integration": {
        "description": "Compute and visualize contour integrals in the complex plane",
        "examples": ["1/(z-1)", "z**2-1", "1/z", "exp(z)", "sin(z)/z", "z/(z**2+1)"],
        "default": "1/(z-1)"
    }
}
def display():
    st.markdown("<h2 class='basket-header'>Complex Mapping</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: var(--text-tertiary);'>Explore Complex Functions</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Complex Visualizer", "Complex Integration"])
    with tab1:
        st.markdown("<h3>Complex Visualizer</h3>", unsafe_allow_html=True)
        st.markdown("Visualize analytic functions with domain coloring to explore their behavior")
        
        st.markdown("### Quick Examples")
        if st.button("z**2", key="complex_example_top_z2"):
            st.session_state["complex_func"] = "z**2"
        
        if st.button("Select complex visualizer", type="primary"):
            st.session_state["selected_model"] = "Complex Visualizer"
            st.markdown("<div class='success-message'>Complex Visualizer selected</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Function Examples")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("z**2", key="complex_example_z2"):
                st.session_state["complex_func"] = "z**2"
        with col2:
            if st.button("z**2 + 1", key="complex_example_z2_plus_1"):
                st.session_state["complex_func"] = "z**2 + 1"
        with col3:
            if st.button("sin(z)", key="complex_example_sin"):
                st.session_state["complex_func"] = "sin(z)"
        with col4:
            if st.button("exp(z)", key="complex_example_exp"):
                st.session_state["complex_func"] = "exp(z)"
        with st.form(key="complex_form"):
            if "complex_func" not in st.session_state:
                st.session_state["complex_func"] = "z**2"
            func_str = st.text_input(
                "f(z) = ",
                value=st.session_state["complex_func"],
                help="Enter a complex function using z, e.g., z**2"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                domain_min = st.slider("Domain Minimum", -10.0, -0.1, -5.0)
            with col2:
                domain_max = st.slider("Domain Maximum", 0.1, 10.0, 5.0)
            with col3:
                resolution = st.slider("Resolution", 100, 1000, 500, step=100)
            submit = st.form_submit_button("Visualize")
            if submit:
                if domain_min >= domain_max:
                    st.error("Domain Minimum must be less than Domain Maximum.")
                else:
                    st.session_state["complex_func"] = func_str
                    visualize_complex_function(func_str, domain_min, domain_max, resolution)
    with tab2:
        st.markdown("<h3>Complex Integration</h3>", unsafe_allow_html=True)
        st.markdown("Compute and visualize contour integrals of complex functions")
        
        st.markdown("### Quick Examples")
        if st.button("1/(z-1)", key="integral_example_header_btn"):
            st.session_state["integral_func"] = "1/(z-1)"
        
        if st.button("Select complex integration", type="primary"):
            st.session_state["selected_model"] = "Complex Integration"
            st.markdown("<div class='success-message'>Complex Integration selected</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Function Examples")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("1/(z-1)", key="integral_example_col_1_z_minus_1"):
                st.session_state["integral_func"] = "1/(z-1)"
        with col2:
            if st.button("1/z", key="integral_example_1_z"):
                st.session_state["integral_func"] = "1/z"
        with col3:
            if st.button("z**2", key="integral_example_z2"):
                st.session_state["integral_func"] = "z**2"
        with col4:
            if st.button("exp(z)/z", key="integral_example_exp_z"):
                st.session_state["integral_func"] = "exp(z)/z"
        with st.form(key="integral_form"):
            if "integral_func" not in st.session_state:
                st.session_state["integral_func"] = "1/(z-1)"
            func_str = st.text_input(
                "f(z) = ",
                value=st.session_state["integral_func"],
                help="Enter a complex function using z, e.g., 1/(z-1)"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                domain_min = st.slider("Domain Minimum", -10.0, -0.1, -5.0, key="int_min")
            with col2:
                domain_max = st.slider("Domain Maximum", 0.1, 10.0, 5.0, key="int_max")
            with col3:
                resolution = st.slider("Resolution", 100, 1000, 500, step=100, key="int_res")
            submit = st.form_submit_button("Visualize")
            if submit:
                if domain_min >= domain_max:
                    st.error("Domain Minimum must be less than Domain Maximum.")
                else:
                    st.session_state["integral_func"] = func_str
                    visualize_integral_contour(func_str, domain_min, domain_max, resolution)

