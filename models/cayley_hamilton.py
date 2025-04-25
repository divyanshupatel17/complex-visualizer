import streamlit as st
import numpy as np
from scipy.linalg import eigvals
import plotly.graph_objects as go
import numpy.polynomial.polynomial as poly

def check_cayley_hamilton(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        st.error("Matrix must be square for Cayley-Hamilton theorem")
        return
    
    if matrix.shape[0] > 10:
        st.warning("Large matrices may have numerical stability issues. Results should be interpreted with caution.")
    
    n = matrix.shape[0]
    if n <= 5:
        latex_matrix = "\\begin{bmatrix} "
        for i in range(n):
            row = " & ".join([f"{matrix[i, j]:.2f}" for j in range(n)])
            latex_matrix += row
            if i < n - 1:
                latex_matrix += " \\\\ "
        latex_matrix += " \\end{bmatrix}"
        st.latex(f"A = {latex_matrix}")
    else:
        st.write("Matrix A:")
        st.write(matrix)
    
    char_poly = np.poly(matrix)
    poly_str = f"p(λ) = λ^{n}"
    for i in range(n):
        coef = char_poly[i]
        if abs(coef) > 1e-10:
            if i == 0:
                poly_str += f" + {coef:.4f}"
            elif i == 1:
                poly_str += f" + {coef:.4f}λ"
            else:
                poly_str += f" + {coef:.4f}λ^{i}"
    st.latex(poly_str)
    
    result = np.zeros_like(matrix)
    A_power = np.eye(n)
    for coef in char_poly:
        result += coef * A_power
        A_power = A_power @ matrix
    
    if n <= 5:
        latex_result = "\\begin{bmatrix} "
        for i in range(n):
            row = " & ".join([f"{result[i, j]:.4f}" for j in range(n)])
            latex_result += row
            if i < n - 1:
                latex_result += " \\\\ "
        latex_result += " \\end{bmatrix}"
        st.latex(f"p(A) = {latex_result}")
    else:
        st.write("p(A) =")
        st.write(result)
    
    tolerance = 1e-10 * np.linalg.norm(matrix)
    if np.allclose(result, np.zeros_like(result), atol=tolerance):
        st.success("Cayley-Hamilton theorem verified! p(A) = 0")
    else:
        st.error("Cayley-Hamilton theorem failed! p(A) ≠ 0")
        st.info(f"Maximum deviation from zero: {np.max(np.abs(result)):.2e}")
    
    eigenvalues = eigvals(matrix)
    st.subheader("Eigenvalues")
    eig_str = "λ = ["
    for i, eig in enumerate(eigenvalues):
        if i > 0:
            eig_str += ", "
        if np.iscomplex(eig):
            real_part = np.real(eig)
            imag_part = np.imag(eig)
            if imag_part >= 0:
                eig_str += f"{real_part:.2f} + {imag_part:.2f}i"
            else:
                eig_str += f"{real_part:.2f} - {abs(imag_part):.2f}i"
        else:
            eig_str += f"{eig:.2f}"
    eig_str += "]"
    st.latex(eig_str)
    
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Unit Circle'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.real(eigenvalues),
        y=np.imag(eigenvalues),
        mode='markers+text',
        marker=dict(
            size=10, 
            color=['blue' if abs(e) <= 1 else 'red' for e in eigenvalues]
        ),
        text=[f"λ{i+1}" for i in range(len(eigenvalues))],
        textposition="top center",
        name='Eigenvalues'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Origin'
    ))
    
    fig.update_layout(
        title='Eigenvalues in Complex Plane',
        xaxis_title='Real',
        yaxis_title='Imaginary',
        showlegend=True,
        width=600,
        height=600
    )
    st.plotly_chart(fig)
    
    st.subheader("Additional Analysis")
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)
    st.latex(f"\\det(A) = {det:.2f}")
    st.latex(r"\text{tr}(A) = " + f"{trace:.2f}")
    
    st.subheader("Matrix Properties")
    eigvals_, eigvecs = np.linalg.eig(matrix)
    if np.linalg.matrix_rank(eigvecs) == n:
        st.success("Matrix is diagonalizable")
    else:
        st.warning("Matrix is not diagonalizable")
    
    if np.allclose(matrix @ matrix.T.conj(), matrix.T.conj() @ matrix):
        st.success("Matrix is normal")
    else:
        st.warning("Matrix is not normal")
    
    rank = np.linalg.matrix_rank(matrix)
    st.write(f"Matrix rank: {rank} (out of {n})")
    
    if np.all(np.abs(eigenvalues) < 1):
        st.success("System is stable (all eigenvalues inside unit circle)")
    elif np.all(np.abs(eigenvalues) <= 1):
        st.warning("System is marginally stable (eigenvalues on unit circle)")
    else:
        st.error("System is unstable (eigenvalues outside unit circle)")
    
    st.subheader("Minimal Polynomial")
    st.info("The minimal polynomial is the polynomial of lowest degree such that p(A) = 0.") 

