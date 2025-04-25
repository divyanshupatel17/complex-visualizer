import streamlit as st
import numpy as np
from scipy.linalg import null_space, orth
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from models.matrix_transformation_visualizer import visualize_matrix_transform
except ImportError:
    def visualize_matrix_transform(matrix, vector, show_basis):
        st.error("Matrix Transformation Visualizer module not found. Please check your installation.")
        st.code("Error importing: models.matrix_transformation_visualizer")
try:
    from models.basis_classifier import classify_basis
except ImportError:
    def classify_basis(matrix):
        st.error("Basis Classifier module not found. Please check your installation.")
        st.code("Error importing: models.basis_classifier")
try:
    from models.subspace_explorer import explore_subspace
except ImportError:
    def explore_subspace(matrix, vector):
        st.error("Subspace Explorer module not found. Please check your installation.")
        st.code("Error importing: models.subspace_explorer")
try:
    from models.inversion_animation import animate_inversion
except ImportError:
    def animate_inversion(matrix):
        st.error("Inversion Animation Engine module not found. Please check your installation.")
        st.code("Error importing: models.inversion_animation")
MATRIX_MODELS = {
    "Matrix Transformation Visualizer": {
        "description": "Visualize how matrices transform vectors and spaces in 2D and 3D",
        "examples": {
            "Rotation": "[[0, -1], [1, 0]]",
            "Scaling": "[[2, 0], [0, 0.5]]",
            "Shear": "[[1, 1], [0, 1]]",
            "Reflection": "[[1, 0], [0, -1]]"
        },
        "default": "[[1, 0], [0, 1]]"
    },
    "Basis Classifier": {
        "description": "Analyze and classify matrix bases, including orthogonality and independence",
        "examples": {
            "Orthogonal": "[[1, 0], [0, 1]]",
            "Dependent": "[[1, 2], [2, 4]]",
            "Independent": "[[1, 1], [1, -1]]"
        },
        "default": "[[1, 0], [0, 1]]"
    },
    "Subspace Explorer": {
        "description": "Explore column space, null space, and other subspaces of matrices",
        "examples": {
            "Full Rank": "[[1, 0], [0, 1]]",
            "Rank Deficient": "[[1, 2], [2, 4]]",
            "3D Subspace": "[[1, 0, 0], [0, 1, 0], [0, 0, 0]]"
        },
        "default": "[[1, 0], [0, 1]]"
    },
    "Inversion Animation Engine": {
        "description": "Visualize matrix inversion and its geometric interpretation",
        "examples": {
            "2x2 Invertible": "[[1, 2], [3, 4]]",
            "3x3 Invertible": "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
            "Singular": "[[1, 1], [1, 1]]"
        },
        "default": "[[1, 0], [0, 1]]"
    }
}
def parse_matrix(matrix_str):
    try:
        matrix_str = matrix_str.strip()
        rows = matrix_str.split('],')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '').strip()
            elements = [float(x.strip()) for x in row.split(',')]
            matrix.append(elements)
        return np.array(matrix)
    except Exception as e:
        st.error(f"Error parsing matrix: {str(e)}")
        return None
def display():
    st.markdown("<h2 class='basket-header'>Matrixland: Linear Algebra Visualizer</h2>", unsafe_allow_html=True)
    st.markdown()
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "Matrix Transformation Visualizer"
    st.markdown("
    tabs = st.tabs(list(MATRIX_MODELS.keys()))
    for i, (model_name, model_info) in enumerate(MATRIX_MODELS.items()):
        with tabs[i]:
            st.markdown(f"**{model_name}**")
            st.markdown(model_info["description"])
            st.markdown("Examples:")
            for name, example in model_info["examples"].items():
                st.code(f"{name}: {example}")
            if st.button(f"Select {model_name}", key=f"select_btn_{model_name}"):
                st.session_state["selected_model"] = model_name
                st.rerun()
    selected_model = st.session_state.get("selected_model", "Matrix Transformation Visualizer")
    st.markdown(f, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("
    model_info = MATRIX_MODELS[selected_model]
    st.markdown("
    example_cols = st.columns(len(model_info["examples"]))
    for i, (name, example) in enumerate(model_info["examples"].items()):
        if example_cols[i].button(name, key=f"example_{i}"):
            st.session_state["matrix_input"] = example
    with st.form(key="matrix_form"):
        matrix_input = st.text_area(
            "Enter Matrix (use [[a, b], [c, d]] format):",
            value=st.session_state.get("matrix_input", model_info["default"]),
            help="Enter a matrix in Python list format"
        )
        if selected_model == "Matrix Transformation Visualizer":
            vector_input = st.text_input(
                "Enter Vector (optional, use [x, y] format):",
                value="[1, 1]",
                help="Enter a vector to transform"
            )
            show_basis = st.checkbox("Show Basis Vectors", value=True)
        submitted = st.form_submit_button("Visualize", type="primary")
        if submitted:
            try:
                st.session_state["matrix_input"] = matrix_input
                matrix = parse_matrix(matrix_input)
                if matrix is None:
                    st.error("Invalid matrix format")
                    return
                if selected_model == "Matrix Transformation Visualizer":
                    vector = parse_matrix(vector_input)
                    if vector is None:
                        st.error("Invalid vector format")
                        return
                    with st.spinner("Generating transformation visualization..."):
                        visualize_matrix_transform(matrix, vector, show_basis)
                elif selected_model == "Basis Classifier":
                    with st.spinner("Analyzing matrix basis..."):
                        classify_basis(matrix)
                elif selected_model == "Subspace Explorer":
                    vector = parse_matrix(vector_input) if "vector_input" in locals() else None
                    with st.spinner("Exploring subspaces..."):
                        explore_subspace(matrix, vector)
                elif selected_model == "Inversion Animation Engine":
                    with st.spinner("Generating inversion animation..."):
                        animate_inversion(matrix)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    with st.expander("Learn about Linear Algebra"):
        st.markdown() 

