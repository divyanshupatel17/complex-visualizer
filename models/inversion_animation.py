import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import inv
import pandas as pd
def animate_inversion(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        st.error("Matrix must be square for inversion")
        return
    det = np.linalg.det(matrix)
    if abs(det) < 1e-10:
        st.error("Matrix is not invertible (determinant is zero)")
        return
    try:
        inverse = inv(matrix)
    except:
        st.error("Could not compute matrix inverse")
        return
    st.markdown("### Matrix Inversion Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Matrix")
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
        st.markdown("#### Determinant")
        st.latex(r"\det(A) = " + f"{det:.4f}")
    with col2:
        st.markdown("#### Inverse Matrix")
        inverse_latex = r"A^{-1} = \begin{bmatrix} "
        for i in range(inverse.shape[0]):
            for j in range(inverse.shape[1]):
                inverse_latex += f"{inverse[i, j]:.2f}"
                if j < inverse.shape[1] - 1:
                    inverse_latex += " & "
            if i < inverse.shape[0] - 1:
                inverse_latex += r" \\ "
        inverse_latex += r" \end{bmatrix}"
        st.latex(inverse_latex)
        product = matrix @ inverse
        st.markdown("#### Verification (A × A⁻¹)")
        if matrix.shape[0] <= 5:
            product_latex = r"A \cdot A^{-1} = \begin{bmatrix} "
            for i in range(product.shape[0]):
                for j in range(product.shape[1]):
                    product_latex += f"{product[i, j]:.2f}"
                    if j < product.shape[1] - 1:
                        product_latex += " & "
                if i < product.shape[0] - 1:
                    product_latex += r" \\ "
            product_latex += r" \end{bmatrix}"
            st.latex(product_latex)
        else:
            st.info("Matrix product is approximately the identity matrix")
            diag_values = np.diag(product)
            off_diag_max = np.max(np.abs(product - np.diag(diag_values)))
            st.write(f"Diagonal elements: {diag_values}")
            st.write(f"Maximum off-diagonal element: {off_diag_max:.6f}")
    if matrix.shape[0] == 2:
        visualize_2d_inversion(matrix, inverse)
    elif matrix.shape[0] == 3:
        visualize_3d_inversion(matrix, inverse)
    else:
        visualize_numerical_inversion(matrix, inverse)
    with st.expander("Matrix Properties"):
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            st.markdown("#### Eigenvalues")
            if len(eigenvalues) <= 10:
                eigenvalues_latex = r"\lambda = \begin{bmatrix} "
                for i, val in enumerate(eigenvalues):
                    eigenvalues_latex += f"{val.real:.2f}{'+' if val.imag >= 0 else ''}{val.imag:.2f}i"
                    if i < len(eigenvalues) - 1:
                        eigenvalues_latex += " & "
                eigenvalues_latex += r" \end{bmatrix}"
                st.latex(eigenvalues_latex)
            else:
                st.write(eigenvalues)
            if matrix.shape[0] <= 5:
                st.markdown("#### Eigenvectors")
                for i, vec in enumerate(eigenvectors.T):
                    vec_latex = r"\mathbf{v}_{" + str(i+1) + "} = \begin{bmatrix} "
                    for j, val in enumerate(vec):
                        vec_latex += f"{val.real:.2f}{'+' if val.imag >= 0 else ''}{val.imag:.2f}i"
                        if j < len(vec) - 1:
                            vec_latex += r" \\ "
                    vec_latex += r" \end{bmatrix}"
                    st.latex(vec_latex)
        except:
            st.warning("Could not compute eigenvalues and eigenvectors")
        try:
            cond = np.linalg.cond(matrix)
            st.markdown("#### Condition Number")
            st.latex(r"\kappa(A) = " + f"{cond:.4f}")
            if cond > 1000:
                st.warning("Matrix is ill-conditioned")
            else:
                st.success("Matrix is well-conditioned")
        except:
            st.warning("Could not compute condition number")
def visualize_2d_inversion(matrix, inverse):
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=unit_square[:, 0],
            y=unit_square[:, 1],
            mode='lines',
            name='Original Square',
            line=dict(color='blue', width=2)
        )
    )
    transformed_square = unit_square @ matrix
    fig.add_trace(
        go.Scatter(
            x=transformed_square[:, 0],
            y=transformed_square[:, 1],
            mode='lines',
            name='Transformed Square',
            line=dict(color='red', width=2)
        )
    )
    frames = []
    steps = 20
    for i in range(steps + 1):
        t = i / steps
        interpolated_matrix = (1 - t) * matrix + t * inverse
        interpolated_square = unit_square @ interpolated_matrix
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=unit_square[:, 0],
                        y=unit_square[:, 1],
                        mode='lines',
                        line=dict(color='blue', width=2)
                    ),
                    go.Scatter(
                        x=interpolated_square[:, 0],
                        y=interpolated_square[:, 1],
                        mode='lines',
                        line=dict(color='red', width=2)
                    )
                ],
                name=f'frame_{i}'
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                    )
                ]
            )
        ],
        width=600,
        height=600,
        title="2D Matrix Inversion Animation",
        xaxis=dict(
            title="x",
            scaleanchor="y", 
            scaleratio=1
        ),
        yaxis=dict(
            title="y"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
def visualize_3d_inversion(matrix, inverse):
    unit_cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1],
        [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]
    ])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=unit_cube[:, 0],
            y=unit_cube[:, 1],
            z=unit_cube[:, 2],
            mode='lines',
            name='Original Cube',
            line=dict(color='blue', width=2)
        )
    )
    transformed_cube = unit_cube @ matrix
    fig.add_trace(
        go.Scatter3d(
            x=transformed_cube[:, 0],
            y=transformed_cube[:, 1],
            z=transformed_cube[:, 2],
            mode='lines',
            name='Transformed Cube',
            line=dict(color='red', width=2)
        )
    )
    frames = []
    steps = 20
    for i in range(steps + 1):
        t = i / steps
        interpolated_matrix = (1 - t) * matrix + t * inverse
        interpolated_cube = unit_cube @ interpolated_matrix
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=unit_cube[:, 0],
                        y=unit_cube[:, 1],
                        z=unit_cube[:, 2],
                        mode='lines',
                        line=dict(color='blue', width=2)
                    ),
                    go.Scatter3d(
                        x=interpolated_cube[:, 0],
                        y=interpolated_cube[:, 1],
                        z=interpolated_cube[:, 2],
                        mode='lines',
                        line=dict(color='red', width=2)
                    )
                ],
                name=f'frame_{i}'
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                    )
                ]
            )
        ],
        width=600,
        height=600,
        title="3D Matrix Inversion Animation",
        scene=dict(
            aspectmode='cube',
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
def visualize_numerical_inversion(matrix, inverse):
    st.markdown("### Numerical Visualization of Matrix Inversion")
    st.write("For matrices larger than 3x3, we provide a numerical visualization of the inversion process.")
    steps = 10
    step = st.slider("Animation Step", 0, steps, 0)
    t = step / steps
    interpolated_matrix = (1 - t) * matrix + t * inverse
    progress_bar = st.progress(step / steps)
    st.write(f"Interpolation: {int(t * 100)}% from original matrix to inverse")
    st.markdown("#### Interpolated Matrix")
    if matrix.shape[0] <= 6:
        interp_latex = r"\begin{bmatrix} "
        for i in range(interpolated_matrix.shape[0]):
            for j in range(interpolated_matrix.shape[1]):
                interp_latex += f"{interpolated_matrix[i, j]:.2f}"
                if j < interpolated_matrix.shape[1] - 1:
                    interp_latex += " & "
            if i < interpolated_matrix.shape[0] - 1:
                interp_latex += r" \\ "
        interp_latex += r" \end{bmatrix}"
        st.latex(interp_latex)
    else:
        df = pd.DataFrame(interpolated_matrix)
        st.dataframe(df, use_container_width=True)
    det_interp = np.linalg.det(interpolated_matrix)
    st.markdown(f"**Determinant of interpolated matrix:** {det_interp:.4f}")
    det_values = []
    for i in range(steps + 1):
        t_i = i / steps
        interp_i = (1 - t_i) * matrix + t_i * inverse
        det_i = np.linalg.det(interp_i)
        det_values.append(det_i)
    det_chart = pd.DataFrame({
        'Step': range(steps + 1),
        'Determinant': det_values
    })
    st.line_chart(det_chart.set_index('Step'))
    st.markdown("#### Matrix Heatmap Visualization")
    fig = go.Figure(data=go.Heatmap(
        z=interpolated_matrix,
        colorscale='Viridis',
        showscale=True
    ))
    fig.update_layout(
        title=f"Matrix at {int(t * 100)}% interpolation",
        height=400,
        width=400,
        xaxis=dict(title="Column"),
        yaxis=dict(title="Row")
    )
    st.plotly_chart(fig, use_container_width=True) 

