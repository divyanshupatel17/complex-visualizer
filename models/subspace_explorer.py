import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from scipy.linalg import null_space, orth, svd
from sklearn.decomposition import PCA
from utils import create_viz_container, close_viz_container
def explore_subspace(matrix, vector=None):
    if matrix.shape[0] > 3 or matrix.shape[1] > 3:
        st.info(f"Standard 3D visualization is not available for {matrix.shape[0]}×{matrix.shape[1]} matrices. Alternative visualizations will be provided.")
        standard_viz = False
    else:
        standard_viz = True
    rank = np.linalg.matrix_rank(matrix)
    if matrix.shape[0] == matrix.shape[1]:
        try:
            det = np.linalg.det(matrix)
            is_singular = abs(det) < 1e-10
        except Exception as e:
            st.error(f"Error computing determinant: {str(e)}")
            det = None
            is_singular = None
    else:
        det = None
        is_singular = True
    try:
        col_space = orth(matrix)
        null_space_basis = null_space(matrix)
    except Exception as e:
        st.error(f"Error computing subspaces: {str(e)}")
        col_space = None
        null_space_basis = None
    st.markdown("### Matrix Analysis")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### Matrix Representation")
        if matrix.shape[0] > 6 or matrix.shape[1] > 6:
            df = pd.DataFrame(matrix)
            st.dataframe(df, use_container_width=True)
        else:
            matrix_latex = r"A = \begin{bmatrix} "
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    matrix_latex += f"{matrix[i, j]:.2f}"
                    if j < matrix.shape[1] - 1:
                        matrix_latex += " & "
                if i < matrix.shape[0] - 1:
                    matrix_latex += r" \\ "
            matrix_latex += r" \end{bmatrix}"
            st.latex(matrix_latex)
    with col2:
        metrics_container = "<div class='matrix-metrics'>"
        matrix_type = f"{'Square' if matrix.shape[0] == matrix.shape[1] else 'Rectangular'}"
        metrics_container += f"<p><b>Matrix Type:</b> {matrix_type} ({matrix.shape[0]}×{matrix.shape[1]})</p>"
        metrics_container += f"<p><b>Rank:</b> {rank}</p>"
        if null_space_basis is not None:
            metrics_container += f"<p><b>Dimension of Null Space:</b> {null_space_basis.shape[1]}</p>"
            metrics_container += f"<p><b>Dimension of Column Space:</b> {col_space.shape[1]}</p>"
        if matrix.shape[0] == matrix.shape[1] and det is not None:
            metrics_container += f"<p><b>Determinant:</b> {det:.4f}</p>"
            if is_singular:
                metrics_container += f"<p><b>Status:</b> <span style='color:red'>Singular</span></p>"
            else:
                metrics_container += f"<p><b>Status:</b> <span style='color:green'>Invertible</span></p>"
        if rank == min(matrix.shape[0], matrix.shape[1]):
            metrics_container += f"<p><b>Full {('row' if matrix.shape[0] <= matrix.shape[1] else 'column')} rank:</b> <span style='color:green'>Yes</span></p>"
        else:
            metrics_container += f"<p><b>Full {('row' if matrix.shape[0] <= matrix.shape[1] else 'column')} rank:</b> <span style='color:red'>No</span></p>"
        if vector is not None:
            try:
                if matrix.shape[0] == matrix.shape[1]:
                    x = np.linalg.solve(matrix, vector)
                    residual = np.linalg.norm(matrix @ x - vector)
                    if residual < 1e-10:
                        metrics_container += f"<p><b>Linear system:</b> <span style='color:green'>Unique solution exists</span></p>"
                    else:
                        metrics_container += f"<p><b>Linear system:</b> <span style='color:red'>No exact solution</span></p>"
                else:
                    x, residuals, rank, s = np.linalg.lstsq(matrix, vector, rcond=None)
                    if len(residuals) == 0 or residuals[0] < 1e-10:
                        metrics_container += f"<p><b>Linear system:</b> <span style='color:green'>Exact solution exists</span></p>"
                    else:
                        metrics_container += f"<p><b>Linear system:</b> <span style='color:orange'>Least squares solution</span></p>"
            except Exception as e:
                metrics_container += f"<p><b>Linear system:</b> <span style='color:red'>Error computing solution</span></p>"
        metrics_container += "</div>"
        st.markdown(metrics_container, unsafe_allow_html=True)
    if null_space_basis is not None and null_space_basis.size > 0 and col_space is not None and col_space.size > 0:
        col3, col4 = st.columns(2)
        with col3:
            with st.expander("Column Space Basis"):
                if col_space.shape[1] <= 5:
                    col_space_latex = r"\text{Col}(A) = \text{span}\left\{"
                    for i, vec in enumerate(col_space.T):
                        col_space_latex += r"\begin{bmatrix} "
                        for j in range(len(vec)):
                            col_space_latex += f"{vec[j]:.2f}"
                            if j < len(vec) - 1:
                                col_space_latex += r" \\ "
                        col_space_latex += r" \end{bmatrix}"
                        if i < col_space.shape[1] - 1:
                            col_space_latex += ", "
                    col_space_latex += r"\right\}"
                    st.latex(col_space_latex)
                else:
                    st.dataframe(pd.DataFrame(col_space), use_container_width=True)
        with col4:
            with st.expander("Null Space Basis"):
                if null_space_basis.shape[1] <= 5:
                    null_space_latex = r"\text{Null}(A) = \text{span}\left\{"
                    for i, vec in enumerate(null_space_basis.T):
                        null_space_latex += r"\begin{bmatrix} "
                        for j in range(len(vec)):
                            null_space_latex += f"{vec[j]:.2f}"
                            if j < len(vec) - 1:
                                null_space_latex += r" \\ "
                        null_space_latex += r" \end{bmatrix}"
                        if i < null_space_basis.shape[1] - 1:
                            null_space_latex += ", "
                    null_space_latex += r"\right\}"
                    st.latex(null_space_latex)
                else:
                    st.dataframe(pd.DataFrame(null_space_basis), use_container_width=True)
    st.markdown("### Visual Representations")
    viz_tabs = st.tabs(["Standard Visualization", "Heatmap", "SVD Analysis"])
    with viz_tabs[0]:
        st.markdown("*Geometric representation showing how the matrix transforms vectors, with basis vectors as arrows from origin.*")
        if standard_viz:
            if col_space is not None and col_space.size > 0:
                viz_container = create_viz_container("Column Space Visualization")
                if matrix.shape[0] == 2:
                    fig = go.Figure()
                    for i, col in enumerate(col_space.T):
                        fig.add_trace(
                            go.Scatter(
                                x=[0, col[0]], 
                                y=[0, col[1]],
                                mode='lines',
                                name=f'Basis Vector {i+1}',
                                line=dict(color='blue', width=3)
                            )
                        )
                    if vector is not None and len(vector) == 2:
                        fig.add_trace(
                            go.Scatter(
                                x=[0, vector[0]], 
                                y=[0, vector[1]],
                                mode='lines+markers',
                                name='Input Vector',
                                line=dict(color='green', width=3),
                                marker=dict(size=8)
                            )
                        )
                    fig.update_layout(
                        title="Column Space (2D)",
                        xaxis_title="x",
                        yaxis_title="y",
                        height=400,
                        width=400,
                        margin=dict(l=40, r=40, t=40, b=40),
                        showlegend=True,
                        legend=dict(x=0.02, y=0.98),
                        plot_bgcolor='rgba(240, 240, 240, 0.8)',
                    )
                    fig.update_xaxes(
                        scaleanchor="y", 
                        scaleratio=1,
                        showgrid=True,
                        zeroline=True,
                        zerolinewidth=2,
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        zeroline=True,
                        zerolinewidth=2,
                    )
                elif matrix.shape[0] == 3:
                    fig = go.Figure()
                    for i, col in enumerate(col_space.T):
                        fig.add_trace(
                            go.Scatter3d(
                                x=[0, col[0]], 
                                y=[0, col[1]],
                                z=[0, col[2]],
                                mode='lines',
                                name=f'Basis Vector {i+1}',
                                line=dict(color='blue', width=5)
                            )
                        )
                    if vector is not None and len(vector) == 3:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[0, vector[0]], 
                                y=[0, vector[1]],
                                z=[0, vector[2]],
                                mode='lines+markers',
                                name='Input Vector',
                                line=dict(color='green', width=5),
                                marker=dict(size=6)
                            )
                        )
                    fig.update_layout(
                        title="Column Space (3D)",
                        height=400,
                        width=400,
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=True,
                        scene=dict(
                            xaxis_title="x",
                            yaxis_title="y",
                            zaxis_title="z",
                            aspectmode='cube',
                            xaxis=dict(showgrid=True, zeroline=True, nticks=10),
                            yaxis=dict(showgrid=True, zeroline=True, nticks=10),
                            zaxis=dict(showgrid=True, zeroline=True, nticks=10)
                        )
                    )
                st.plotly_chart(fig, use_container_width=True)
                close_viz_container()
        else:
            st.info("For matrices larger than 3×3, we provide alternative visualizations:")
            if matrix.shape[1] > 1:
                try:
                    if matrix.shape[1] <= 10:
                        n_points = 200
                        domain_points = np.random.normal(0, 1, (n_points, matrix.shape[1]))
                        transformed_points = domain_points @ matrix.T
                        pca = PCA(n_components=2)
                        transformed_2d = pca.fit_transform(transformed_points)
                        vis_container = create_viz_container("PCA-based Dimension Reduction Visualization")
                        fig = px.scatter(
                            x=transformed_2d[:, 0], 
                            y=transformed_2d[:, 1],
                            color=np.linalg.norm(domain_points, axis=1),
                            title=f"2D Projection of Matrix Action (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})",
                            labels={"color": "Input Norm"}
                        )
                        fig.update_layout(
                            xaxis_title="Principal Component 1",
                            yaxis_title="Principal Component 2",
                            height=500,
                            width=700
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"This visualization shows how the matrix transforms random points from its domain to its range, projected to 2D using PCA.")
                        close_viz_container()
                except Exception as e:
                    st.error(f"Could not create dimension reduction visualization: {str(e)}")
            else:
                st.warning("Dimension reduction visualization requires matrices with at least 2 columns.")
    with viz_tabs[1]:
        st.markdown("*Color-coded representation of matrix values where intensity shows magnitude, making patterns and structure visible.*")
        viz_container = create_viz_container("Matrix Heatmap")
        fig = px.imshow(
            matrix,
            labels=dict(x="Column", y="Row", color="Value"),
            x=[f"Col {i+1}" for i in range(matrix.shape[1])],
            y=[f"Row {i+1}" for i in range(matrix.shape[0])],
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            title="Matrix Heatmap",
            height=500,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True)
        if matrix.shape[0] == matrix.shape[1]:
            try:
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                eigenvectors_real = np.real(eigenvectors)
                fig2 = px.imshow(
                    eigenvectors_real,
                    labels=dict(x="Eigenvector", y="Component", color="Value"),
                    x=[f"λ₍{i+1}₎={eigenvalues[i]:.2f}" for i in range(len(eigenvalues))],
                    y=[f"Comp {i+1}" for i in range(eigenvectors.shape[0])],
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0
                )
                fig2.update_layout(
                    title="Eigenvector Components Heatmap",
                    height=500,
                    width=700
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("Eigenvector components shown as a heatmap, where each column represents an eigenvector.")
            except Exception as e:
                st.error(f"Could not compute eigenvalue visualization: {str(e)}")
        close_viz_container()
    with viz_tabs[2]:
        st.markdown("*Decomposition that reveals matrix's fundamental properties through singular values, showing condition number, rank, and low-rank approximations.*")
        viz_container = create_viz_container("Singular Value Decomposition")
        try:
            U, s, Vt = svd(matrix)
            st.markdown("### Singular Value Decomposition Analysis")
            df_sing = pd.DataFrame({
                'Index': range(1, len(s) + 1),
                'Singular Value': s
            })
            fig = px.bar(
                df_sing, 
                x='Index', 
                y='Singular Value',
                title='Singular Values Distribution'
            )
            fig.update_layout(
                xaxis_title="Singular Value Index",
                yaxis_title="Magnitude",
                height=400,
                width=600
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Singular Values:**")
            sing_val_text = ", ".join([f"{val:.4f}" for val in s])
            st.markdown(f"`{sing_val_text}`")
            if len(s) > 0:
                condition_number = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
                st.markdown(f"**Condition Number:** {condition_number:.4f}")
                if condition_number < 10:
                    st.success("The matrix is well-conditioned.")
                elif condition_number < 1000:
                    st.warning("The matrix is moderately ill-conditioned.")
                else:
                    st.error("The matrix is severely ill-conditioned.")
            numerical_rank = sum(s > 1e-10)
            st.markdown(f"**Numerical Rank:** {numerical_rank}")
            if matrix.shape[0] <= 10 and matrix.shape[1] <= 10:
                col1, col2 = st.columns(2)
                with col1:
                    rank_select = st.slider(
                        "Rank for Approximation", 
                        min_value=1, 
                        max_value=min(len(s), 5),
                        value=1
                    )
                s_approx = np.zeros_like(s)
                s_approx[:rank_select] = s[:rank_select]
                matrix_approx = U @ np.diag(s_approx) @ Vt
                with col1:
                    fig1 = px.imshow(
                        matrix,
                        title="Original Matrix",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.imshow(
                        matrix_approx,
                        title=f"Rank-{rank_select} Approximation",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                error = np.linalg.norm(matrix - matrix_approx, 'fro') / np.linalg.norm(matrix, 'fro')
                st.markdown(f"**Relative Approximation Error:** {error:.4f} ({error*100:.2f}%)")
        except Exception as e:
            st.error(f"Could not compute SVD analysis: {str(e)}")
        close_viz_container() 

