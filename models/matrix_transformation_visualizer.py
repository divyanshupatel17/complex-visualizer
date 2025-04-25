import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_matrix_transform(matrix, vector, show_basis=True):
    if matrix.shape[0] not in [2, 3] or matrix.shape[1] not in [2, 3]:
        st.error("Matrix must be 2x2 or 3x3")
        return
    
    if vector.shape[0] != matrix.shape[0]:
        st.error("Vector dimensions must match matrix dimensions")
        return
    
    transformed_vector = matrix @ vector
    basis_vectors = np.eye(matrix.shape[0])
    transformed_basis = matrix @ basis_vectors
    
    if matrix.shape[0] == 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Space", "Transformed Space"))
        fig.add_trace(
            go.Scatter(
                x=[0, vector[0]], 
                y=[0, vector[1]],
                mode='lines+markers',
                name='Vector',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        if show_basis:
            for i, (basis, color) in enumerate(zip(basis_vectors.T, ['red', 'green'])):
                fig.add_trace(
                    go.Scatter(
                        x=[0, basis[0]], 
                        y=[0, basis[1]],
                        mode='lines',
                        name=f'Basis {i+1}',
                        line=dict(color=color, width=2, dash='dash')
                    ),
                    row=1, col=1
                )
        
        fig.add_trace(
            go.Scatter(
                x=[0, transformed_vector[0]], 
                y=[0, transformed_vector[1]],
                mode='lines+markers',
                name='Transformed Vector',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ),
            row=1, col=2
        )
        
        if show_basis:
            for i, (basis, color) in enumerate(zip(transformed_basis.T, ['red', 'green'])):
                fig.add_trace(
                    go.Scatter(
                        x=[0, basis[0]], 
                        y=[0, basis[1]],
                        mode='lines',
                        name=f'Transformed Basis {i+1}',
                        line=dict(color=color, width=2, dash='dash')
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            showlegend=True,
            height=500,
            width=1000
        )
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Space", "Transformed Space"),
                          specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
        fig.add_trace(
            go.Scatter3d(
                x=[0, vector[0]], 
                y=[0, vector[1]],
                z=[0, vector[2]],
                mode='lines+markers',
                name='Vector',
                line=dict(color='blue', width=3),
                marker=dict(size=5)
            ),
            row=1, col=1
        )
        
        if show_basis:
            for i, (basis, color) in enumerate(zip(basis_vectors.T, ['red', 'green', 'purple'])):
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, basis[0]], 
                        y=[0, basis[1]],
                        z=[0, basis[2]],
                        mode='lines',
                        name=f'Basis {i+1}',
                        line=dict(color=color, width=2, dash='dash')
                    ),
                    row=1, col=1
                )
        
        fig.add_trace(
            go.Scatter3d(
                x=[0, transformed_vector[0]], 
                y=[0, transformed_vector[1]],
                z=[0, transformed_vector[2]],
                mode='lines+markers',
                name='Transformed Vector',
                line=dict(color='blue', width=3),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        
        if show_basis:
            for i, (basis, color) in enumerate(zip(transformed_basis.T, ['red', 'green', 'purple'])):
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, basis[0]], 
                        y=[0, basis[1]],
                        z=[0, basis[2]],
                        mode='lines',
                        name=f'Transformed Basis {i+1}',
                        line=dict(color=color, width=2, dash='dash')
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            showlegend=True,
            height=600,
            width=1000,
            scene=dict(
                aspectmode='cube'
            )
        )
    
    st.plotly_chart(fig)
    
    st.markdown("### Matrix Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Matrix")
        st.latex(r"A = " + np.array2string(matrix, separator=', '))
        st.markdown("#### Determinant")
        det = np.linalg.det(matrix)
        st.latex(r"\det(A) = " + f"{det:.4f}")
        if abs(det) < 1e-10:
            st.warning("Matrix is singular (determinant ≈ 0)")
        elif det < 0:
            st.info("Matrix includes a reflection (determinant < 0)")
    
    with col2:
        st.markdown("#### Eigenvalues")
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            st.latex(r"\lambda = " + np.array2string(eigenvalues, separator=', '))
            if np.all(np.isreal(eigenvalues)):
                st.info("All eigenvalues are real")
            else:
                st.info("Matrix has complex eigenvalues")
        except:
            st.error("Could not compute eigenvalues")
        
        st.markdown("#### Rank")
        rank = np.linalg.matrix_rank(matrix)
        st.latex(r"\text{rank}(A) = " + str(rank)) 

