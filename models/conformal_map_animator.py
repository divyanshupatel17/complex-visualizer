import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
import io
from PIL import Image
import base64
from utils import get_figure_size, apply_image_container
def visualize_conformal_map(func_str, domain_min, domain_max, resolution):
    try:
        st.session_state["current_visualization"] = "conformal_map_wide"
        z = sp.Symbol('z')
        transformations = (standard_transformations + 
                          (implicit_multiplication_application,))
        expr = parse_expr(func_str, local_dict={'z': z}, transformations=transformations)
        func = lambdify(z, expr, modules=['numpy'])
        grid_density = min(30, resolution // 15)
        real_lines = np.linspace(domain_min, domain_max, grid_density)
        imag_lines = np.linspace(domain_min, domain_max, grid_density)
        fig_size = get_figure_size()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        ax1.set_title('Original Domain (z-plane)')
        ax1.set_xlabel('Re(z)')
        ax1.set_ylabel('Im(z)')
        ax1.set_xlim(domain_min, domain_max)
        ax1.set_ylim(domain_min, domain_max)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        for i, x in enumerate(real_lines):
            h = i / len(real_lines)
            color = hsv_to_rgb([h, 0.7, 0.9])
            ax1.axvline(x=x, color=color, linestyle='-', linewidth=1, alpha=0.8)
        for i, y in enumerate(imag_lines):
            h = i / len(imag_lines)
            color = hsv_to_rgb([h, 0.7, 0.9])
            ax1.axhline(y=y, color=color, linestyle='-', linewidth=1, alpha=0.8)
        ax2.set_title(f'Transformed Domain (w-plane)\nw = f(z) = {func_str}')
        ax2.set_xlabel('Re(w)')
        ax2.set_ylabel('Im(w)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        t = np.linspace(domain_min, domain_max, resolution)
        all_w_real = []
        all_w_imag = []
        for i, x in enumerate(real_lines):
            z_line = x + 1j * t
            w_line = func(z_line)
            valid_idx = np.isfinite(w_line)
            w_line = w_line[valid_idx]
            if len(w_line) > 0:
                h = i / len(real_lines)
                color = hsv_to_rgb([h, 0.7, 0.9])
                ax2.plot(w_line.real, w_line.imag, color=color, linewidth=1, alpha=0.8)
                all_w_real.extend(w_line.real)
                all_w_imag.extend(w_line.imag)
        for i, y in enumerate(imag_lines):
            z_line = t + 1j * y
            w_line = func(z_line)
            valid_idx = np.isfinite(w_line)
            w_line = w_line[valid_idx]
            if len(w_line) > 0:
                h = i / len(imag_lines)
                color = hsv_to_rgb([h, 0.7, 0.9])
                ax2.plot(w_line.real, w_line.imag, color=color, linewidth=1, alpha=0.8)
                all_w_real.extend(w_line.real)
                all_w_imag.extend(w_line.imag)
        if all_w_real and all_w_imag:
            filtered_real = np.array(all_w_real)
            filtered_imag = np.array(all_w_imag)
            q1_real, q3_real = np.percentile(filtered_real, [25, 75])
            q1_imag, q3_imag = np.percentile(filtered_imag, [25, 75])
            iqr_real = q3_real - q1_real
            iqr_imag = q3_imag - q1_imag
            lower_real = q1_real - 2 * iqr_real
            upper_real = q3_real + 2 * iqr_real
            lower_imag = q1_imag - 2 * iqr_imag
            upper_imag = q3_imag + 2 * iqr_imag
            padding = 0.1
            ax2.set_xlim(lower_real - padding * (upper_real - lower_real),
                        upper_real + padding * (upper_real - lower_real))
            ax2.set_ylim(lower_imag - padding * (upper_imag - lower_imag),
                        upper_imag + padding * (upper_imag - lower_imag))
        plt.tight_layout()
        apply_image_container(fig, caption=f"Static conformal mapping for f(z) = {func_str}")
        st.markdown("")
        anim_size = (fig_size[0] * 0.8, fig_size[1] * 0.8)
        frames = []
        num_frames = 21
        if all_w_real and all_w_imag:
            real_lim = ax2.get_xlim()
            imag_lim = ax2.get_ylim()
        else:
            real_lim = [domain_min, domain_max]
            imag_lim = [domain_min, domain_max]
        for frame in range(num_frames):
            frame_fig, frame_ax = plt.subplots(figsize=anim_size)
            frame_ax.set_aspect('equal')
            frame_ax.grid(True, alpha=0.3)
            frame_ax.set_title(f'Transformation: w = f(z) = {func_str}')
            frame_ax.set_xlabel('Re')
            frame_ax.set_ylabel('Im')
            t = frame / (num_frames - 1)
            frame_ax.set_xlim((1-t)*domain_min + t*real_lim[0], (1-t)*domain_max + t*real_lim[1])
            frame_ax.set_ylim((1-t)*domain_min + t*imag_lim[0], (1-t)*domain_max + t*imag_lim[1])
            for i, x in enumerate(real_lines):
                z_line = x + 1j * np.linspace(domain_min, domain_max, min(resolution, 100))
                w_line = func(z_line)
                valid_idx = np.isfinite(w_line)
                z_valid = z_line[valid_idx]
                w_valid = w_line[valid_idx]
                if len(w_valid) > 0:
                    h = i / len(real_lines)
                    color = hsv_to_rgb([h, 0.7, 0.9])
                    interpolated = (1 - t) * z_valid + t * w_valid
                    frame_ax.plot(interpolated.real, interpolated.imag, color=color, 
                                linewidth=1, alpha=0.8)
            for i, y in enumerate(imag_lines):
                z_line = np.linspace(domain_min, domain_max, min(resolution, 100)) + 1j * y
                w_line = func(z_line)
                valid_idx = np.isfinite(w_line)
                z_valid = z_line[valid_idx]
                w_valid = w_line[valid_idx]
                if len(w_valid) > 0:
                    h = i / len(imag_lines)
                    color = hsv_to_rgb([h, 0.7, 0.9])
                    interpolated = (1 - t) * z_valid + t * w_valid
                    frame_ax.plot(interpolated.real, interpolated.imag, color=color, 
                                linewidth=1, alpha=0.8)
            buf = io.BytesIO()
            plt.tight_layout()
            frame_fig.savefig(buf, format='png', dpi=70)
            buf.seek(0)
            frames.append(buf.read())
            plt.close(frame_fig)
        with st.container():
            if frames:
                html = f"<img src='data:image/gif;base64,{{0}}' alt='Animation'>"
                from PIL import Image
                images = [Image.open(io.BytesIO(frame)) for frame in frames]
                gif_buffer = io.BytesIO()
                images[0].save(
                    gif_buffer, 
                    format='GIF',
                    save_all=True,
                    append_images=images[1:],
                    optimize=True,
                    duration=100,
                    loop=0
                )
                gif_buffer.seek(0)
                gif_base64 = base64.b64encode(gif_buffer.read()).decode('utf-8')
                st.markdown(f"<img src='data:image/gif;base64,{gif_base64}' alt='Animation'>", unsafe_allow_html=True)
                st.markdown(f"<div class='image-caption'>Animation of w = f(z) = {func_str}</div>", unsafe_allow_html=True)
            else:
                st.warning("Could not generate animation frames.")
        with st.expander("Understanding Conformal Mappings"):
            st.markdown("Conformal mappings preserve angles between curves. This visualization shows how the grid lines in the z-plane are transformed into the w-plane while maintaining their angle intersections.")
        with st.expander("Properties of this Conformal Map"):
            try:
                w0 = complex(func(0))
                if np.isfinite(w0):
                    st.markdown(f"f(0) = {w0:.3g}")
                else:
                    st.markdown("z = 0 is a singularity of this function")
            except:
                st.markdown("z = 0 may be a singularity of this function")
            if "1/z" in func_str or "/z" in func_str:
                st.markdown("This function maps infinity to a finite point or vice versa.")
    except Exception as e:
        st.error(f"Error visualizing conformal map: {str(e)}")
        st.info(f"Function input: f(z) = {func_str}")
        if "sympify" in str(e):
            st.markdown("Please check your function syntax. Use 'z' as the variable.")
        elif "singular" in str(e).lower() or "domain" in str(e).lower():
            st.warning(f"The function may have singularities in the domain [{domain_min}, {domain_max}].")
            st.markdown("Try adjusting the domain to avoid points where the function is not defined.") 

