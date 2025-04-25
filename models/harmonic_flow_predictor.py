import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
from utils import get_figure_size, apply_image_container
def visualize_harmonic_flow(func_str, domain_min, domain_max, resolution):
    try:
        st.session_state["current_visualization"] = "harmonic_flow_wide"
        z = sp.Symbol('z', complex=True)
        transformations = (standard_transformations + 
                          (implicit_multiplication_application,))
        expr = parse_expr(func_str, local_dict={'z': z}, transformations=transformations)
        func = lambdify(z, expr, modules=['numpy'])
        resolution = min(resolution, 40)
        real = np.linspace(domain_min, domain_max, resolution)
        imag = np.linspace(domain_min, domain_max, resolution)
        real_grid, imag_grid = np.meshgrid(real, imag)
        z_grid = real_grid + 1j * imag_grid
        w = func(z_grid)
        u = np.real(w)
        v = np.imag(w)
        eps = 1e-6
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                du_dx[j, i] = (u[j, i+1] - u[j, i-1]) / (2 * (real[i+1] - real[i]))
                du_dy[j, i] = (u[j+1, i] - u[j-1, i]) / (2 * (imag[j+1] - imag[j]))
                dv_dx[j, i] = (v[j, i+1] - v[j, i-1]) / (2 * (real[i+1] - real[i]))
                dv_dy[j, i] = (v[j+1, i] - v[j-1, i]) / (2 * (imag[j+1] - imag[j]))
        fig_size = get_figure_size()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        contour_real = ax1.contourf(real_grid, imag_grid, u, 15, cmap='viridis', alpha=0.7)
        colorbar1 = plt.colorbar(contour_real, ax=ax1)
        colorbar1.set_label('Real Part Value')
        ax1.streamplot(real_grid, imag_grid, du_dx, du_dy, density=1, color='white', 
                      arrowsize=1, linewidth=1)
        ax1.set_title('Real Part with Gradient Flow')
        ax1.set_xlabel('Re(z)')
        ax1.set_ylabel('Im(z)')
        contour_imag = ax2.contourf(real_grid, imag_grid, v, 15, cmap='plasma', alpha=0.7)
        colorbar2 = plt.colorbar(contour_imag, ax=ax2)
        colorbar2.set_label('Imaginary Part Value')
        ax2.streamplot(real_grid, imag_grid, dv_dx, dv_dy, density=1, color='white', 
                      arrowsize=1, linewidth=1)
        ax2.set_title('Imaginary Part with Gradient Flow')
        ax2.set_xlabel('Re(z)')
        ax2.set_ylabel('Im(z)')
        plt.tight_layout()
        st.markdown(f"Function: f(z) = {func_str}")
        st.markdown(f"**Domain**: [{domain_min}, {domain_max}] × [{domain_min}, {domain_max}]")
        apply_image_container(fig, caption=f"Harmonic flow visualization for f(z) = {func_str}")
        cr_check = np.abs(du_dx - dv_dy) + np.abs(du_dy + dv_dx)
        cr_satisfied = np.mean(cr_check[1:-1, 1:-1]) < 0.1
        with st.expander("Harmonic Analysis"):
            st.markdown("This analysis examines whether the function satisfies the Cauchy-Riemann equations, which is a necessary condition for analyticity.")
            if cr_satisfied:
                st.success("The function appears to be analytic in this domain (Cauchy-Riemann equations satisfied).")
                st.markdown("The gradient flows show how the function values change across the complex plane.")
            else:
                st.warning("The function may not be analytic in parts of this domain.")
        with st.expander("Learn about Harmonic Functions"):
            st.markdown("Harmonic functions are solutions to Laplace's equation. The real and imaginary parts of analytic functions are harmonic functions.")
    except Exception as e:
        st.error(f"Error visualizing harmonic flow: {str(e)}")
        st.info(f"Function input: f(z) = {func_str}")
        if "sympify" in str(e):
            st.markdown("Please check your function syntax. Make sure to use z as the complex variable.")
        elif "singular" in str(e).lower() or "domain" in str(e).lower():
            st.warning(f"The function may have singularities in the domain [{domain_min}, {domain_max}].")
            st.markdown("Try adjusting the domain range to avoid singularities like division by zero.") 

