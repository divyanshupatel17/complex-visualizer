import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import plotly.graph_objects as go
from utils import get_figure_size, apply_image_container, create_3d_toggle
def visualize_matrix_transformation(matrix_str, domain_min, domain_max, resolution):
    try:
        if "current_visualization" not in st.session_state:
            st.session_state["current_visualization"] = "matrix_transformation"
        try:
            matrix = np.array(eval(matrix_str))
            if matrix.ndim != 2:
                raise ValueError("Matrix must be 2D")
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be square")
        except Exception as e:
            st.error(f"Invalid matrix format: {str(e)}")
            st.info()
            return
        resolution = min(max(resolution, 10), 50)
        x = np.linspace(domain_min, domain_max, resolution)
        y = np.linspace(domain_min, domain_max, resolution)
        X, Y = np.meshgrid(x, y)
        if matrix.shape[0] == 2:
            points = np.vstack([X.flatten(), Y.flatten()])
            transformed = matrix @ points
            X_trans = transformed[0].reshape(X.shape)
            Y_trans = transformed[1].reshape(Y.shape)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size())
            ax1.set_title('Original Grid')
            ax1.grid(True)
            ax1.set_xlim(domain_min, domain_max)
            ax1.set_ylim(domain_min, domain_max)
            ax1.set_aspect('equal')
            ax2.set_title('Transformed Grid')
            ax2.grid(True)
            ax2.set_xlim(domain_min, domain_max)
            ax2.set_ylim(domain_min, domain_max)
            ax2.set_aspect('equal')
            for i in range(resolution):
                ax1.plot(X[i, :], Y[i, :], 'b-', alpha=0.3)
                ax1.plot(X[:, i], Y[:, i], 'b-', alpha=0.3)
                ax2.plot(X_trans[i, :], Y_trans[i, :], 'r-', alpha=0.3)
                ax2.plot(X_trans[:, i], Y_trans[:, i], 'r-', alpha=0.3)
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            circle_trans = matrix @ np.vstack([circle_x, circle_y])
            ax1.plot(circle_x, circle_y, 'k-', label='Unit Circle')
            ax2.plot(circle_trans[0], circle_trans[1], 'k-', label='Transformed Circle')
            try:
                eigvals, eigvecs = np.linalg.eig(matrix)
                for i, (val, vec) in enumerate(zip(eigvals, eigvecs.T)):
                    if np.isreal(val) and np.isreal(vec).all():
                        vec = vec.real
                        ax1.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.2, fc='g', ec='g', label=f'Eigenvector {i+1}')
                        trans_vec = matrix @ vec
                        ax2.arrow(0, 0, trans_vec[0], trans_vec[1], head_width=0.1, head_length=0.2, fc='g', ec='g', label=f'Scaled Eigenvector {i+1}')
            except:
                pass
            ax1.legend()
            ax2.legend()
            apply_image_container(fig, f"2D Matrix Transformation: {matrix_str}", max_height=400)
            with st.expander("Matrix Properties"):
                st.markdown("
                st.markdown(f"**Determinant**: {np.linalg.det(matrix):.4f}")
                st.markdown(f"**Trace**: {np.trace(matrix):.4f}")
                try:
                    st.markdown(f"**Eigenvalues**: {np.linalg.eigvals(matrix)}")
                except:
                    st.markdown("Could not compute eigenvalues")
                if np.linalg.det(matrix) != 0:
                    st.markdown(f"**Inverse Matrix**:\n```\n{np.linalg.inv(matrix)}\n```")
        elif matrix.shape[0] == 3:
            z = np.linspace(domain_min, domain_max, resolution)
            X, Y, Z = np.meshgrid(x, y, z)
            points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
            transformed = matrix @ points
            X_trans = transformed[0].reshape(X.shape)
            Y_trans = transformed[1].reshape(Y.shape)
            Z_trans = transformed[2].reshape(Z.shape)
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.3),
                name='Original Points'
            ))
            fig.add_trace(go.Scatter3d(
                x=X_trans.flatten(), y=Y_trans.flatten(), z=Z_trans.flatten(),
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.3),
                name='Transformed Points'
            ))
            phi = np.linspace(0, np.pi, 20)
            theta = np.linspace(0, 2*np.pi, 20)
            phi, theta = np.meshgrid(phi, theta)
            x_sphere = np.sin(phi) * np.cos(theta)
            y_sphere = np.sin(phi) * np.sin(theta)
            z_sphere = np.cos(phi)
            sphere_points = np.vstack([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])
            sphere_trans = matrix @ sphere_points
            fig.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale='Blues',
                opacity=0.3,
                name='Unit Sphere'
            ))
            fig.add_trace(go.Surface(
                x=sphere_trans[0].reshape(x_sphere.shape),
                y=sphere_trans[1].reshape(y_sphere.shape),
                z=sphere_trans[2].reshape(z_sphere.shape),
                colorscale='Reds',
                opacity=0.3,
                name='Transformed Sphere'
            ))
            fig.update_layout(
                title=f"3D Matrix Transformation: {matrix_str}",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode='cube'
                ),
                width=800,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Matrix Properties"):
                st.markdown("
                st.markdown(f"**Determinant**: {np.linalg.det(matrix):.4f}")
                st.markdown(f"**Trace**: {np.trace(matrix):.4f}")
                try:
                    st.markdown(f"**Eigenvalues**: {np.linalg.eigvals(matrix)}")
                except:
                    st.markdown("Could not compute eigenvalues")
                if np.linalg.det(matrix) != 0:
                    st.markdown(f"**Inverse Matrix**:\n```\n{np.linalg.inv(matrix)}\n```")
        with st.expander("Understanding Matrix Transformations"):
            st.markdown()
    except Exception as e:
        st.error(f"Error visualizing matrix transformation: {str(e)}")
        st.info(f"Matrix input: {matrix_str}")
if __name__ == "__main__":
    st.title("Matrix Transformation Visualizer")
    with st.form("matrix_form"):
        matrix_str = st.text_input("Enter matrix", value="[[1,0],[0,1]]")
        col1, col2, col3 = st.columns(3)
        with col1:
            domain_min = st.slider("Domain Min", -5.0, 0.0, -2.0)
        with col2:
            domain_max = st.slider("Domain Max", 0.0, 5.0, 2.0)
        with col3:
            resolution = st.slider("Resolution", 10, 50, 20)
        submitted = st.form_submit_button("Visualize", type="primary")
        if submitted:
            visualize_matrix_transformation(matrix_str, domain_min, domain_max, resolution) 

