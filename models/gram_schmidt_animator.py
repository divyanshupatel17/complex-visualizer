import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import norm

def gram_schmidt_animator(vectors, space_type='real'):
    if not vectors:
        st.error("No vectors provided")
        return
    
    try:
        vectors = [np.array(v, dtype=complex if space_type == 'complex' else float) for v in vectors]
    except Exception as e:
        st.error(f"Error converting vectors: {str(e)}")
        return
    
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        st.error("All vectors must have the same dimension")
        return
    
    if any(np.all(np.abs(v) < 1e-10) for v in vectors):
        st.error("Zero vectors are not allowed")
        return
    
    max_norm = max(norm(v) for v in vectors)
    if max_norm > 0:
        vectors = [v / max_norm for v in vectors]
    
    matrix = np.column_stack(vectors)
    if np.linalg.matrix_rank(matrix) < len(vectors):
        st.error("Vectors must be linearly independent")
        return
    
    orthogonal = []
    orthonormal = []
    for i, v in enumerate(vectors):
        u = v.copy()
        for j in range(i):
            if space_type == 'real':
                proj = np.dot(u, orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j]) * orthogonal[j]
            else:
                proj = np.vdot(u, orthogonal[j]) / np.vdot(orthogonal[j], orthogonal[j]) * orthogonal[j]
            u = u - proj
        orthogonal.append(u)
        u_norm = norm(u)
        if u_norm > 1e-10:
            orthonormal.append(u / u_norm)
        else:
            st.error("Zero vector produced in Gram-Schmidt process")
            return
    
    st.subheader("Gram-Schmidt Process Steps")
    for i, (v, u, e) in enumerate(zip(vectors, orthogonal, orthonormal)):
        st.markdown(f"### Step {i+1}")
        st.latex(f"\\mathbf{{v}}_{i+1} = {np.array2string(v, precision=2)}")
        st.latex(f"\\mathbf{{u}}_{i+1} = {np.array2string(u, precision=2)}")
        st.latex(f"\\mathbf{{e}}_{i+1} = {np.array2string(e, precision=2)}")
        
        if len(v) <= 3:
            fig = go.Figure()
            v_real = np.real(v)
            u_real = np.real(u)
            e_real = np.real(e)
            
            fig.add_trace(go.Scatter3d(
                x=[0, v_real[0]], y=[0, v_real[1]], z=[0, v_real[2]] if len(v) == 3 else [0, 0],
                mode='lines+markers',
                name=f'v{i+1} (real part)',
                line=dict(color='red', width=4),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[0, u_real[0]], y=[0, u_real[1]], z=[0, u_real[2]] if len(u) == 3 else [0, 0],
                mode='lines+markers',
                name=f'u{i+1} (real part)',
                line=dict(color='blue', width=4),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[0, e_real[0]], y=[0, e_real[1]], z=[0, e_real[2]] if len(e) == 3 else [0, 0],
                mode='lines+markers',
                name=f'e{i+1} (real part)',
                line=dict(color='green', width=4),
                marker=dict(size=4)
            ))
            
            for j in range(i):
                prev_u = np.real(orthogonal[j])
                fig.add_trace(go.Scatter3d(
                    x=[0, prev_u[0]], y=[0, prev_u[1]], z=[0, prev_u[2]] if len(prev_u) == 3 else [0, 0],
                    mode='lines',
                    name=f'u{j+1} (real part)',
                    line=dict(color='gray', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f'Step {i+1} Visualization (Real Parts Only)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z' if len(v) == 3 else '',
                    aspectmode='data'
                ),
                showlegend=True
            )
            st.plotly_chart(fig)
            
            if space_type == 'complex':
                v_imag = np.imag(v)
                u_imag = np.imag(u)
                e_imag = np.imag(e)
                
                fig_imag = go.Figure()
                fig_imag.add_trace(go.Scatter3d(
                    x=[0, v_imag[0]], y=[0, v_imag[1]], z=[0, v_imag[2]] if len(v) == 3 else [0, 0],
                    mode='lines+markers',
                    name=f'v{i+1} (imaginary part)',
                    line=dict(color='red', width=4),
                    marker=dict(size=4)
                ))
                
                fig_imag.add_trace(go.Scatter3d(
                    x=[0, u_imag[0]], y=[0, u_imag[1]], z=[0, u_imag[2]] if len(u) == 3 else [0, 0],
                    mode='lines+markers',
                    name=f'u{i+1} (imaginary part)',
                    line=dict(color='blue', width=4),
                    marker=dict(size=4)
                ))
                
                fig_imag.add_trace(go.Scatter3d(
                    x=[0, e_imag[0]], y=[0, e_imag[1]], z=[0, e_imag[2]] if len(e) == 3 else [0, 0],
                    mode='lines+markers',
                    name=f'e{i+1} (imaginary part)',
                    line=dict(color='green', width=4),
                    marker=dict(size=4)
                ))
                
                for j in range(i):
                    prev_u = np.imag(orthogonal[j])
                    fig_imag.add_trace(go.Scatter3d(
                        x=[0, prev_u[0]], y=[0, prev_u[1]], z=[0, prev_u[2]] if len(prev_u) == 3 else [0, 0],
                        mode='lines',
                        name=f'u{j+1} (imaginary part)',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                
                fig_imag.update_layout(
                    title=f'Step {i+1} Visualization (Imaginary Parts Only)',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z' if len(v) == 3 else '',
                        aspectmode='data'
                    ),
                    showlegend=True
                )
                st.plotly_chart(fig_imag)
    
    st.subheader("Final Orthonormal Basis")
    for i, e in enumerate(orthonormal):
        st.latex(f"\\mathbf{{e}}_{i+1} = {np.array2string(e, precision=2)}")
    
    st.subheader("Orthonormality Verification")
    for i in range(len(orthonormal)):
        for j in range(len(orthonormal)):
            if space_type == 'real':
                inner_prod = np.dot(orthonormal[i], orthonormal[j])
            else:
                inner_prod = np.vdot(orthonormal[i], orthonormal[j])
            
            if i == j:
                if abs(inner_prod - 1) < 1e-10:
                    st.success(f"e{i+1} is normalized")
                else:
                    st.error(f"e{i+1} is not normalized (inner product = {inner_prod:.2e})")
            else:
                if abs(inner_prod) < 1e-10:
                    st.success(f"e{i+1} and e{j+1} are orthogonal")
                else:
                    st.error(f"e{i+1} and e{j+1} are not orthogonal (inner product = {inner_prod:.2e})") 

