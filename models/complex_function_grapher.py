import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
import plotly.graph_objects as go
import io
from PIL import Image
import base64
import time
def get_figure_size():
    return (8, 6)
def apply_image_container(fig, caption, max_height):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    st.markdown(
        f"""<div style="text-align: center; max-height: {max_height}px; overflow-y: auto;">
        <img src="data:image/png;base64,{img_str}" alt="{caption}" style="max-height: {max_height}px;"/>
        <p style="font-size: 0.8em; color: #ccc;">{caption}</p>
        </div>""",
        unsafe_allow_html=True
    )
    plt.close(fig)
def create_3d_toggle():
    return st.sidebar.checkbox("Show 3D Visualization", value=False)
def visualize_complex_function(func_str, domain_min, domain_max, resolution, show_colorwheel=True):
    try:
        if "current_visualization" not in st.session_state:
            st.session_state["current_visualization"] = "complex_function"
        z = sp.Symbol('z')
        transformations = standard_transformations + (implicit_multiplication_application,)
        func_str = func_str.strip().lower().replace('^', '**').replace(' ', '')
        local_dict = {
            'z': z, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp,
            'log': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E, 'i': sp.I, 'j': sp.I
        }
        try:
            expr = parse_expr(func_str, local_dict=local_dict, transformations=transformations)
        except (SyntaxError, TypeError, ValueError) as e:
            st.error(f"Invalid function syntax: {str(e)}")
            st.info("Examples: z**2, sin(z), exp(z), log(z), z**2 + 1/z")
            return
        try:
            func = lambdify(z, expr, modules=['numpy'])
        except Exception as e:
            st.error(f"Cannot convert function to numerical form: {str(e)}")
            return
        resolution = min(max(resolution, 100), 500)
        real = np.linspace(domain_min, domain_max, resolution)
        imag = np.linspace(domain_min, domain_max, resolution)
        real_grid, imag_grid = np.meshgrid(real, imag)
        z_grid = real_grid + 1j * imag_grid
        try:
            w = func(z_grid)
            valid_mask = np.isfinite(w)
            w[~valid_mask] = 0
        except (ValueError, ZeroDivisionError) as e:
            st.error(f"Function evaluation failed: {str(e)}")
            st.info("The function may have singularities in this domain.")
            return
        mag = np.abs(w)
        phase = np.angle(w)
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        mag_normalized = np.tanh(0.5 * np.log1p(mag))
        
        st.sidebar.markdown("### Visualization Options")
        
        color_scheme = st.sidebar.selectbox("Color Scheme", ["Standard", "Enhanced", "Contrast"], index=0)
        show_contours = st.sidebar.checkbox("Show Contours", value=True)
        contour_density = st.sidebar.slider("Contour Density", 5, 15, 8)
        show_3d = create_3d_toggle()
        hsv = np.zeros((resolution, resolution, 3))
        hsv[:, :, 0] = phase_normalized
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag_normalized
        if color_scheme == "Enhanced":
            hsv[:, :, 1] = 0.9
            hsv[:, :, 2] = mag_normalized * 1.1
        elif color_scheme == "Contrast":
            hsv[:, :, 2] = np.power(mag_normalized, 0.8)
        rgb = hsv_to_rgb(hsv)
        if show_3d:
            st.subheader("3D Visualization")
            sample_rate = max(1, resolution // 100)
            fig_3d = go.Figure(data=[
                go.Surface(
                    z=np.log1p(mag[::sample_rate, ::sample_rate]),
                    x=real_grid[::sample_rate, ::sample_rate],
                    y=imag_grid[::sample_rate, ::sample_rate],
                    colorscale='viridis',
                    surfacecolor=phase_normalized[::sample_rate, ::sample_rate],
                    colorbar=dict(title="Phase", tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=["-π", "-π/2", "0", "π/2", "π"])
                )
            ])
            fig_3d.update_layout(
                title=f"3D Surface: |f(z)| for f(z) = {func_str}",
                width=700, height=500,
                scene=dict(
                    xaxis_title="Re(z)", yaxis_title="Im(z)", zaxis_title="log(|f(z)|)",
                    aspectratio=dict(x=1, y=1, z=0.5)
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown("*3D view shows magnitude as height and phase as color.*")
        st.subheader("2D Visualization (Domain Coloring)")
        fig, ax = plt.subplots(figsize=get_figure_size())
        ax.imshow(rgb, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
        if show_contours:
            levels_mag = np.logspace(np.log10(max(1e-5, np.min(mag[mag > 0]))), np.log10(np.max(mag)), contour_density)
            ax.contour(real_grid, imag_grid, mag, levels=levels_mag, colors='black', alpha=0.5, linewidths=0.5)
            levels_phase = np.linspace(-np.pi, np.pi, contour_density)
            ax.contour(real_grid, imag_grid, phase, levels=levels_phase, colors='white', alpha=0.5, linewidths=0.5)
        show_analysis = st.sidebar.checkbox("Show Zeros/Poles", value=False)
        if show_analysis:
            try:
                zeros = sp.solve(expr, z)
                poles = sp.solve(1/expr, z)
                for zero in zeros:
                    if zero.is_complex and abs(zero) < domain_max:
                        ax.plot(float(zero.re), float(zero.im), 'o', color='green', markersize=6, label='Zero')
                for pole in poles:
                    if pole.is_complex and abs(pole) < domain_max:
                        ax.plot(float(pole.re), float(pole.im), 'x', color='red', markersize=6, label='Pole')
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc='upper right')
            except Exception:
                st.sidebar.warning("Unable to compute zeros/poles.")
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(f'f(z) = {func_str}')
        ax.grid(False)
        if show_colorwheel:
            wheel_ax = fig.add_axes([0.85, 0.15, 0.15, 0.15], projection='polar')
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.linspace(0, 1, 10)
            Theta, R = np.meshgrid(theta, r)
            wheel_hsv = np.zeros((10, 100, 3))
            wheel_hsv[:, :, 0] = Theta / (2*np.pi)
            wheel_hsv[:, :, 1] = 1.0
            wheel_hsv[:, :, 2] = 1.0
            wheel_ax.pcolormesh(Theta, R, hsv_to_rgb(wheel_hsv)[:, :, 0], shading='auto')
            wheel_ax.set_xticks(np.linspace(0, 2*np.pi, 8))
            wheel_ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
            wheel_ax.set_yticks([])
            wheel_ax.set_title('Phase', fontsize=10)
        apply_image_container(fig, f"Domain coloring for f(z) = {func_str}", max_height=400)
        st.subheader("Animation")
        num_frames = 20
        animate_automatically = st.checkbox("Play Animation Automatically", value=False)
        if animate_automatically:
            anim_placeholder = st.empty()
            stop_button = st.button("Stop Animation")
            frame_idx = 0
            while animate_automatically and not stop_button:
                t = frame_idx / (num_frames - 1)
                anim_hsv = hsv.copy()
                anim_hsv[:, :, 0] = (phase_normalized + t) % 1.0
                anim_rgb = hsv_to_rgb(anim_hsv)
                anim_fig, anim_ax = plt.subplots(figsize=(6, 5))
                anim_ax.imshow(anim_rgb, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
                anim_ax.set_xlabel('Re(z)')
                anim_ax.set_ylabel('Im(z)')
                anim_ax.set_title(f'f(z) = {func_str} (Phase Shift: {t:.2f})')
                anim_ax.grid(False)
                with anim_placeholder.container():
                    apply_image_container(anim_fig, f"Phase-shifted domain coloring (Frame {frame_idx + 1}/{num_frames})", max_height=300)
                frame_idx = (frame_idx + 1) % num_frames
                time.sleep(0.1)
                if stop_button:
                    break
            if stop_button:
                st.info("Animation stopped.")
        else:
            frame_idx = st.slider("Animation Frame", 0, num_frames - 1, 0, key="anim_slider")
            t = frame_idx / (num_frames - 1)
            anim_hsv = hsv.copy()
            anim_hsv[:, :, 0] = (phase_normalized + t) % 1.0
            anim_rgb = hsv_to_rgb(anim_hsv)
            anim_fig, anim_ax = plt.subplots(figsize=(6, 5))
            anim_ax.imshow(anim_rgb, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
            anim_ax.set_xlabel('Re(z)')
            anim_ax.set_ylabel('Im(z)')
            anim_ax.set_title(f'f(z) = {func_str} (Phase Shift: {t:.2f})')
            anim_ax.grid(False)
            apply_image_container(anim_fig, f"Phase-shifted domain coloring (Frame {frame_idx + 1}/{num_frames})", max_height=300)
        with st.expander("Domain Coloring Interpretation"):
            st.markdown("""
            **How to interpret domain coloring:**
            - **Color (Hue)** represents the phase (argument) of the complex number
            - **Brightness** represents the magnitude (absolute value)
            - **Contour lines** show constant magnitude (black) and constant phase (white)
            
            The animation gradually rotates the phase to help visualize the behavior of the function.
            """)
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.info("Please check your input or try a different function.")
if __name__ == "__main__":
    st.title("Complex Function Visualizer")
    with st.form("visualization_form"):
        func_str = st.text_input("Enter function f(z)", value="z**2")
        col1, col2, col3 = st.columns(3)
        with col1:
            domain_min = st.slider("Domain Min", -10.0, 0.0, -5.0)
        with col2:
            domain_max = st.slider("Domain Max", 0.0, 10.0, 5.0)
        with col3:
            resolution = st.slider("Resolution", 100, 500, 300)
        show_colorwheel = st.checkbox("Show Color Wheel", value=True)
        submitted = st.form_submit_button("Visualize", type="primary")
        if submitted:
            visualize_complex_function(func_str, domain_min, domain_max, resolution, show_colorwheel)

