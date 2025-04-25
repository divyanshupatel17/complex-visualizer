import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import re
from baskets.basket1.complex_mapping import display as complex_mapping_display
from models.subspace_explorer import explore_subspace
from models.inversion_animation import animate_inversion
from models.eigen_motion import simulate_eigen_motion
from models.equation_solver import solve_equation, solve_system
from models.cayley_hamilton import check_cayley_hamilton
from models.inner_product_intuition import inner_product_intuition
from models.gram_schmidt_animator import gram_schmidt_animator
from utils.sidebar import create_sidebar
from utils import create_viz_container, close_viz_container, create_animation_container, close_animation_container
st.set_page_config(
    page_title="CVLA Interactive AI Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
def format_example(title, code, description=None):
    html = f"""
    <div class="example-section">
        <div class="example-title">{title}</div>
    """
    
    if description:
        html += f'<p style="color: #ECEFCA; font-size: 0.9rem; margin: 5px 0;">{description}</p>'
    
    html += f"""
        <pre class="example-code">{code}</pre>
    </div>
    """
    
    return html
def load_css():
    css = """
    <style>
        :root {
            --neon-blue: #00f3ff;
            --neon-purple: #ff00ff;
            --neon-green: #00ff00;
            --neon-yellow: #ffff00;
            --dark-bg: #0a0a0a;
            --darker-bg: #030303;
            --text-primary: #ffffff;
            --text-secondary: #ECEFCA;
            --text-tertiary: #7AE2CF;
            --text-inactive: #94B4C1;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
def main():
    load_css()
    selected_basket = create_sidebar()
    st.markdown("<h1 class='grid-background'>CVLA Interactive AI Lab</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>An AI-powered interactive platform for exploring Complex Variables & Linear Algebra visually</p>", unsafe_allow_html=True)
    selected_description = ""
    if selected_basket == "Complex Mapping":
        selected_description = "Visualize complex functions and mappings in the complex plane"
    elif selected_basket == "Complex Integration":
        selected_description = "Compute and visualize contour integrals in the complex plane"
    elif selected_basket == "Matrixland":
        selected_description = "Explore matrix operations and vector spaces visually"
    elif selected_basket == "Eigen Exploratorium":
        selected_description = "Visualize eigenvalues, eigenvectors, and matrix transformations"
    elif selected_basket == "Inner Product Lab":
        selected_description = "Explore inner products and orthogonalization processes"
    st.markdown(f"<div class='subheader'>{selected_description}</div>", unsafe_allow_html=True)
    if "max_viz_height" not in st.session_state:
        st.session_state["max_viz_height"] = 400
    if "default_image_size" in st.session_state:
        st.session_state["visualization_size"] = st.session_state["default_image_size"]
    else:
        st.session_state["visualization_size"] = "medium"
    if selected_basket == "Complex Mapping":
        complex_mapping_display()
    elif selected_basket == "Matrixland":
        st.markdown("<h2 class='basket-header'>Matrixland & Vector Playground</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Subspace Explorer", "Matrix Inversion"])
        with tab1:
            st.markdown("<h3 class='basket-header'>Subspace Explorer</h3>", unsafe_allow_html=True)
            st.markdown("### Example Matrices")
            st.markdown("Try these matrices to explore different subspace properties.", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div style='text-align: center;'>Identity matrix</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}")
            with col2:
                st.markdown("<div style='text-align: center;'>Rotation matrix</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}")
            with col3:
                st.markdown("<div style='text-align: center;'>Projection matrix</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}")
            st.markdown("### Matrix Dimensions")
            n_rows = st.number_input(
                "Number of rows:",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="subspace_rows"
            )
            n_cols = st.number_input(
                "Number of columns:",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="subspace_cols"
            )
            st.markdown("### Matrix Entries")
            matrix = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    with cols[j]:
                        matrix[i, j] = st.number_input(
                            f"a{i+1}{j+1}", 
                            value=1.0 if i == j else 0.0,
                            key=f"subspace_a{i+1}{j+1}",
                            format="%.2f",
                            label_visibility="collapsed"
                        )
            st.markdown("### Defined Matrix")
            matrix_latex = r"\begin{bmatrix} "
            for i in range(n_rows):
                for j in range(n_cols):
                    matrix_latex += f"{matrix[i, j]:.2f}"
                    if j < n_cols - 1:
                        matrix_latex += " & "
                if i < n_rows - 1:
                    matrix_latex += r" \\ "
            matrix_latex += r" \end{bmatrix}"
            st.latex(matrix_latex)
            if st.checkbox("Include Vector", value=False):
                st.markdown("### Vector Input")
                vector_dim = n_rows
                vector = np.zeros(vector_dim)
                cols = st.columns(min(vector_dim, 6))
                for i in range(vector_dim):
                    with cols[i % len(cols)]:
                        vector[i] = st.number_input(
                            f"v{i+1}", 
                            value=1.0 if i == 0 else 0.0,
                            key=f"subspace_v{i+1}",
                            format="%.2f",
                            label_visibility="visible"
                        )
                st.markdown("### Defined Vector")
                if vector_dim <= 10:
                    vector_latex = r"\mathbf{v} = \begin{bmatrix} "
                    for i in range(vector_dim):
                        vector_latex += f"{vector[i]:.2f}"
                        if i < vector_dim - 1:
                            vector_latex += r" \\ "
                    vector_latex += r" \end{bmatrix}"
                    st.latex(vector_latex)
                else:
                    st.dataframe(pd.DataFrame(vector, columns=["Value"]), use_container_width=True)
            else:
                vector = None
            if st.button("Explore Subspace"):
                explore_subspace(matrix, vector)
        with tab2:
            st.markdown("<h3 class='basket-header'>Matrix Inversion</h3>", unsafe_allow_html=True)
            st.markdown("### Example Matrices")
            st.markdown("Try these matrices to see how conditioning affects inversion.", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div style='text-align: center;'>Well-conditioned matrix</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 4 & 1 \\ 3 & 2 \end{bmatrix}")
            with col2:
                st.markdown("<div style='text-align: center;'>Ill-conditioned matrix</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 1.0 & 0.9999 \\ 1.0 & 1.0001 \end{bmatrix}")
            st.markdown("### Matrix Dimensions")
            n_rows = st.number_input(
                "Number of rows:",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="inversion_rows"
            )
            n_cols = st.number_input(
                "Number of columns:",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="inversion_cols"
            )
            if n_rows != n_cols:
                st.warning("For matrix inversion, matrix must be square. Adjusting columns to match rows.")
                n_cols = n_rows
            st.markdown("### Matrix Entries")
            matrix = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    with cols[j]:
                        matrix[i, j] = st.number_input(
                            f"a{i+1}{j+1}", 
                            value=1.0 if i == j else 0.0,
                            key=f"inversion_a{i+1}{j+1}",
                            format="%.2f",
                            label_visibility="collapsed"
                        )
            st.markdown("### Defined Matrix")
            if n_rows > 6:
                df = pd.DataFrame(matrix)
                st.dataframe(df, use_container_width=True)
            else:
                matrix_latex = r"\begin{bmatrix} "
                for i in range(n_rows):
                    for j in range(n_cols):
                        matrix_latex += f"{matrix[i, j]:.2f}"
                        if j < n_cols - 1:
                            matrix_latex += " & "
                    if i < n_rows - 1:
                        matrix_latex += r" \\ "
                matrix_latex += r" \end{bmatrix}"
                st.latex(matrix_latex)
            try:
                det = np.linalg.det(matrix)
                if abs(det) < 1e-10:
                    st.warning(f"Matrix determinant is {det:.6f}, which is close to zero. The matrix may not be invertible.")
                else:
                    st.info(f"Matrix determinant: {det:.6f}")
                if st.button("Animate Inversion"):
                    anim_container = create_animation_container(
                        title="Matrix Inversion Animation",
                        description="Watch how the matrix is transformed during the inversion process."
                    )
                    animate_inversion(matrix)
                    close_animation_container()
            except np.linalg.LinAlgError:
                st.error("Cannot compute determinant. Please check your matrix entries.")
    elif selected_basket == "Eigen Exploratorium":
        st.markdown("<h2 class='basket-header'>Eigen Exploratorium</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs([
            "Eigenvalue Explorer",
            "Equation Solver AI"
        ])
        with tab1:
            st.markdown("<h3 class='basket-header'>Eigenvalue Explorer</h3>", unsafe_allow_html=True)
            with st.expander("About Eigenvalues and the Cayley-Hamilton Theorem", expanded=False):
                st.markdown("Eigenvalues and eigenvectors are fundamental concepts in linear algebra that help us understand how matrices transform vectors.")
            
            st.markdown("### Matrix Dimensions")
            n_rows = st.number_input(
                "Number of rows:",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="eigen_rows"
            )
            n_cols = st.number_input(
                "Number of columns:",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="eigen_cols"
            )
            if n_rows != n_cols:
                st.warning("For eigenvalue analysis, matrix must be square. Adjusting columns to match rows.")
                n_cols = n_rows
            
            st.markdown("### Matrix Entries")
            matrix = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    with cols[j]:
                        matrix[i, j] = st.number_input(
                            f"a{i+1}{j+1}", 
                            value=1.0 if i == j else 0.0,
                            key=f"eigen_a{i+1}{j+1}",
                            format="%.2f",
                            label_visibility="collapsed"
                        )
            
            st.markdown("### Defined Matrix")
            if n_rows > 6:
                df = pd.DataFrame(matrix)
                st.dataframe(df, use_container_width=True)
            else:
                matrix_latex = r"\begin{bmatrix} "
                for i in range(n_rows):
                    for j in range(n_cols):
                        matrix_latex += f"{matrix[i, j]:.2f}"
                        if j < n_cols - 1:
                            matrix_latex += " & "
                    if i < n_rows - 1:
                        matrix_latex += r" \\ "
                matrix_latex += r" \end{bmatrix}"
                st.latex(matrix_latex)
            try:
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                
                st.markdown("### Characteristic Polynomial")
                
                st.markdown("<div class='subheader'>Properties of the Matrix</div>", unsafe_allow_html=True)
                
                st.markdown("### Eigenvalues & Determinant")
                det_value = np.linalg.det(matrix)
                trace_value = np.trace(matrix)
                if n_rows == 2:
                    poly_str = f"λ² - {trace_value:.4f}λ + {det_value:.4f}"
                    st.latex(f"p_A(\\lambda) = {poly_str}")
                elif n_rows == 3:
                    trace = trace_value
                    det = det_value
                    minor_sum = 0
                    for i in range(3):
                        minor = np.delete(np.delete(matrix, i, 0), i, 1)
                        minor_sum += np.linalg.det(minor)
                    poly_str = f"λ³ - {trace:.4f}λ² + {minor_sum:.4f}λ - {det:.4f}"
                    st.latex(f"p_A(\\lambda) = {poly_str}")
                else:
                    st.info("Characteristic polynomial is available for visualization but not displayed in full for matrices larger than 3×3")
                
                st.markdown("### Eigenvalue Analysis")
                eig_cols = st.columns(min(len(eigenvalues), 4))
                for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                    if abs(eigenval.imag) < 1e-10:
                        eigenval_str = f"{eigenval.real:.4f}"
                    else:
                        eigenval_str = f"{eigenval.real:.4f} + {eigenval.imag:.4f}i"
                    with eig_cols[i % len(eig_cols)]:
                        st.markdown(f"**λ{i+1} = {eigenval_str}**")
                if n_rows <= 4:
                    eigenvectors_cols = st.columns(min(n_rows, 4))
                    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                        with eigenvectors_cols[i % len(eigenvectors_cols)]:
                            st.markdown(f"<div style='text-align: center;'>Eigenvector {i+1}</div>", unsafe_allow_html=True)
                            vec_latex = r"\mathbf{v}_{" + str(i+1) + r"} = \begin{bmatrix} "
                            for j, val in enumerate(eigenvec):
                                if abs(val.imag) < 1e-10:
                                    vec_latex += f"{val.real:.2f}"
                                else:
                                    vec_latex += f"{val.real:.2f} + {val.imag:.2f}i"
                                if j < len(eigenvec) - 1:
                                    vec_latex += r" \\ "
                            vec_latex += r" \end{bmatrix}"
                            st.latex(vec_latex)
                else:
                    st.write("Eigenvectors are available for computation but not displayed in full for matrices larger than 4×4")
                
                st.markdown("### Visualization")
                visualization_option = st.radio(
                    "Select visualization:", 
                    ["Eigenvector Visualization", "Cayley-Hamilton Verification"]
                )
                if visualization_option == "Eigenvector Visualization":
                    st.write("This visualization shows how eigenvectors transform under the matrix.")
                    if st.button("Simulate Eigen Motion"):
                        anim_container = create_animation_container(
                            title="Eigenvector Motion Simulation",
                            description="This animation shows how eigenvectors transform under matrix multiplication."
                        )
                        simulate_eigen_motion(matrix)
                        close_animation_container()
                else:
                    st.write("Verify that the matrix satisfies its own characteristic polynomial.")
                    if st.button("Verify Cayley-Hamilton"):
                        viz_container = create_viz_container("Cayley-Hamilton Verification")
                        check_cayley_hamilton(matrix)
                        close_viz_container()
            except np.linalg.LinAlgError:
                st.error("Could not compute eigenvalues for this matrix. Please check if the matrix entries are valid.")
        with tab2:
            st.markdown("<h3 class='basket-header'>Equation Solver AI</h3>", unsafe_allow_html=True)
            eq_type = st.radio("Choose equation type:", ["Single Equation", "System of Equations"])
            if eq_type == "Single Equation":
                st.markdown("Enter a single equation in the form 'expression = expression'.")
                equation = st.text_input("Enter equation:")
                if st.button("Solve Equation"):
                    solve_equation(equation)
            else:
                st.markdown("Enter a system of equations separated by newlines.")
                equations = st.text_area("Enter system of equations:")
                if st.button("Solve System"):
                    lines = [line.strip() for line in equations.split('\n')]
                    if len(lines) == 1 and lines[0].count('=') > 1:
                        processed_input = re.sub(r'(=\s*[-+\d\w\s\.]+)(\s+[a-zA-Z]+\s*[-+])', r'\1\n\2', lines[0])
                        lines = [line.strip() for line in processed_input.split('\n')]
                    equation_list = []
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = []
                        current_part = ""
                        in_equation = False
                        for char in line:
                            current_part += char
                            if char == '=':
                                in_equation = True
                            elif in_equation and char.isalpha() and current_part.strip().endswith(" "):
                                parts.append(current_part.rstrip())
                                current_part = char
                                in_equation = False
                        if current_part:
                            parts.append(current_part)
                        if not parts:
                            parts = [line]
                        for part in parts:
                            if '=' in part:
                                equation_list.append(part.strip())
                    if not equation_list:
                        st.error("Please enter valid equations in the form 'expression = expression'")
                    else:
                        try:
                            solve_system(equation_list)
                        except Exception as e:
                            st.error(f"Error solving system: {str(e)}")
    elif selected_basket == "Inner Product Lab":
        st.title("Inner Product Lab")
        tab1, tab2 = st.tabs([
            "Inner Product Intuition Machine",
            "Gram-Schmidt Animator"
        ])
        with tab1:
            st.header("Inner Product Intuition Machine")
            st.write("Visualize and understand inner products in different spaces.")
            space_type = st.radio("Space Type", ["Real", "Complex"])
            vector_size = st.selectbox("Vector Size", ["2D", "3D"])
            if vector_size == "2D":
                col1, col2 = st.columns(2)
                with col1:
                    v1_x = st.number_input("v1_x", value=1.0)
                    v1_y = st.number_input("v1_y", value=0.0)
                    v1 = np.array([v1_x, v1_y])
                with col2:
                    v2_x = st.number_input("v2_x", value=0.0)
                    v2_y = st.number_input("v2_y", value=1.0)
                    v2 = np.array([v2_x, v2_y])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    v1_x = st.number_input("v1_x", value=1.0)
                    v1_y = st.number_input("v1_y", value=0.0)
                    v1_z = st.number_input("v1_z", value=0.0)
                    v1 = np.array([v1_x, v1_y, v1_z])
                with col2:
                    v2_x = st.number_input("v2_x", value=0.0)
                    v2_y = st.number_input("v2_y", value=1.0)
                    v2_z = st.number_input("v2_z", value=0.0)
                    v2 = np.array([v2_x, v2_y, v2_z])
            inner_product_intuition(v1, v2, space_type.lower())
        with tab2:
            st.header("Gram-Schmidt Animator")
            st.write("Visualize the Gram-Schmidt process step by step.")
            
            st.markdown("### Example Vectors")
            st.markdown("Try these vectors to see a clear orthogonalization process.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div style='text-align: center;'>Vector 1</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}")
            with col2:
                st.markdown("<div style='text-align: center;'>Vector 2</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}")
            with col3:
                st.markdown("<div style='text-align: center;'>Vector 3</div>", unsafe_allow_html=True)
                st.latex(r"\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}")
            space_type = st.radio("Space Type", ["Real", "Complex"], key="gram_schmidt")
            n_vectors = st.slider("Number of Vectors", 2, 3, 2)
            vectors = []
            for i in range(n_vectors):
                st.subheader(f"Vector {i+1}")
                if n_vectors == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.number_input(f"x_{i+1}", value=1.0 if i == 0 else 0.0)
                    with col2:
                        y = st.number_input(f"y_{i+1}", value=0.0 if i == 0 else 1.0)
                    vectors.append(np.array([x, y]))
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x = st.number_input(f"x_{i+1}", value=1.0 if i == 0 else 0.0)
                    with col2:
                        y = st.number_input(f"y_{i+1}", value=0.0 if i == 1 else 0.0)
                    with col3:
                        z = st.number_input(f"z_{i+1}", value=0.0 if i == 2 else 0.0)
                    vectors.append(np.array([x, y, z]))
            gram_schmidt_animator(vectors, space_type.lower())
if __name__ == "__main__":
    main()

