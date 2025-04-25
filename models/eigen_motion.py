import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import eig
def simulate_eigen_motion(matrix, duration=5, steps=100):
    try:
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=float)
        if matrix.shape[0] != matrix.shape[1]:
            st.error("Matrix must be square")
            return
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            st.error("Matrix contains NaN or infinite values")
            return
        max_val = np.max(np.abs(matrix))
        if max_val > 0:
            matrix = matrix / max_val
        eigenvalues, eigenvectors = eig(matrix)
        st.markdown("### Eigenvalues and Eigenvectors")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            if np.iscomplex(val):
                val_str = f"{np.real(val):.2f} {'+' if np.imag(val) >= 0 else '-'} {abs(np.imag(val)):.2f}i"
            else:
                val_str = f"{val:.2f}"
            vec_str = []
            for x in vec:
                if np.iscomplex(x):
                    vec_str.append(f"{np.real(x):.2f} {'+' if np.imag(x) >= 0 else '-'} {abs(np.imag(x)):.2f}i")
                else:
                    vec_str.append(f"{x:.2f}")
            st.latex(f"\\lambda_{i+1} = {val_str}")
            vec_matrix = "\\begin{bmatrix} " + " \\\\ ".join(vec_str) + " \\end{bmatrix}"
            st.latex(f"\\mathbf{{v}}_{i+1} = {vec_matrix}")
        frames = []
        t = np.linspace(0, 2*np.pi, steps)
        if matrix.shape[0] == 2:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            unit_circle = np.vstack((circle_x, circle_y))
            transformed_circle = matrix @ unit_circle
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                name='Unit Circle',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=transformed_circle[0],
                y=transformed_circle[1],
                mode='lines',
                name='Transformed Circle',
                line=dict(color='red', width=2)
            ))
            for i, vec in enumerate(eigenvectors.T):
                scale = 2.0
                scaled_vec = scale * np.real(vec)
                fig.add_trace(go.Scatter(
                    x=[0, scaled_vec[0]],
                    y=[0, scaled_vec[1]],
                    mode='lines+markers',
                    name=f'Eigenvector {i+1}',
                    line=dict(color='green', width=2)
                ))
            fig.update_layout(
                title='Matrix Transformation Visualization',
                xaxis_title='X',
                yaxis_title='Y',
                showlegend=True
            )
            st.markdown("This visualization shows the distribution of eigenvalues in the complex plane.")
        elif matrix.shape[0] == 3:
            phi = np.linspace(0, np.pi, 20)
            theta = np.linspace(0, 2*np.pi, 20)
            phi, theta = np.meshgrid(phi, theta)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
            transformed_points = matrix @ points
            x_trans = transformed_points[0].reshape(x.shape)
            y_trans = transformed_points[1].reshape(y.shape)
            z_trans = transformed_points[2].reshape(z.shape)
            fig = go.Figure()
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.3,
                colorscale='Blues',
                name='Unit Sphere'
            ))
            fig.add_trace(go.Surface(
                x=x_trans, y=y_trans, z=z_trans,
                opacity=0.3,
                colorscale='Reds',
                name='Transformed Sphere'
            ))
            for i, vec in enumerate(eigenvectors.T):
                scale = 2.0
                scaled_vec = scale * np.real(vec)
                fig.add_trace(go.Scatter3d(
                    x=[0, scaled_vec[0]],
                    y=[0, scaled_vec[1]],
                    z=[0, scaled_vec[2]],
                    mode='lines+markers',
                    name=f'Eigenvector {i+1}',
                    line=dict(color='green', width=4)
                ))
            fig.update_layout(
                title='3D Matrix Transformation Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                showlegend=True
            )
        else:
            fig = go.Figure()
            real_parts = np.real(eigenvalues)
            imag_parts = np.imag(eigenvalues)
            magnitudes = np.abs(eigenvalues)
            fig.add_trace(go.Scatter(
                x=real_parts,
                y=imag_parts,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=magnitudes,
                    colorscale='Viridis',
                    colorbar=dict(title='Magnitude'),
                    showscale=True
                ),
                text=[f"λ{i+1}" for i in range(len(eigenvalues))],
                textposition="top center",
                name='Eigenvalues'
            ))
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                line=dict(color='rgba(100,100,100,0.3)', dash='dash'),
                name='Unit Circle'
            ))
            fig.add_shape(
                type="line",
                x0=-2, y0=0, x1=2, y1=0,
                line=dict(color="rgba(100,100,100,0.3)", width=1)
            )
            fig.add_shape(
                type="line",
                x0=0, y0=-2, x1=0, y1=2,
                line=dict(color="rgba(100,100,100,0.3)", width=1)
            )
            fig.update_layout(
                title=f'Eigenvalue Distribution for {matrix.shape[0]}×{matrix.shape[0]} Matrix',
                xaxis_title='Real Part',
                yaxis_title='Imaginary Part',
                xaxis=dict(range=[-2, 2]),
                yaxis=dict(range=[-2, 2], scaleanchor="x", scaleratio=1),
                showlegend=True
            )
            st.markdown("This visualization shows the distribution of eigenvalues in the complex plane.")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in eigen motion simulation: {str(e)}")
        st.error("Please check your matrix and try again.") 

