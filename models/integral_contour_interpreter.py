import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
from scipy import integrate
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
def get_figure_size():
    return (8, 8)
def apply_image_container(fig, caption):
    st.pyplot(fig)
    st.caption(caption)
def parse_complex_function(func_str):
    z = sp.Symbol('z', complex=True)
    try:
        transformations = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(func_str, local_dict={'z': z}, transformations=transformations)
        if any(f in str(expr) for f in ['conjugate', 'Abs']):
            st.warning("Function may be non-analytic (e.g., contains conjugate or |z|). Results may be unreliable.")
        if any(f in str(expr) for f in ['log', 'sqrt', 'asin', 'acos', 'atan']):
            st.warning("Function may have branch cuts (e.g., log(z)). Ensure the contour avoids branch points.")
        return expr, z
    except Exception as e:
        st.error(f"Error parsing function: {str(e)}")
        st.stop()
def evaluate_function(func, z_sym, grid):
    try:
        func_numpy = lambdify(z_sym, func, modules=["numpy"])
        return func_numpy(grid)
    except Exception as e:
        st.error(f"Error evaluating function: {str(e)}")
        st.stop()
def compute_residues(func, z_sym, domain_min, domain_max):
    try:
        poles = []
        residues = []
        denom = sp.denom(func)
        factors = sp.factor_list(denom)[1]
        for factor, multiplicity in factors:
            if factor.has(z_sym):
                try:
                    roots = sp.solve(factor, z_sym)
                    for root in roots:
                        try:
                            x, y = float(sp.re(root)), float(sp.im(root))
                            if not (domain_min <= x <= domain_max and domain_min <= y <= domain_max):
                                continue
                            if multiplicity == 1:
                                residue = sp.limit((z_sym - root) * func, z_sym, root)
                            else:
                                expr = (z_sym - root) ** multiplicity * func
                                for _ in range(multiplicity - 1):
                                    expr = sp.diff(expr, z_sym)
                                residue = sp.limit(expr, z_sym, root) / sp.factorial(multiplicity - 1)
                            try:
                                residue = complex(residue)
                                root = complex(root)
                            except:
                                pass
                            poles.append(root)
                            residues.append(residue)
                        except Exception as e:
                            st.warning(f"Could not compute residue at z = {root}: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not solve factor {factor}: {str(e)}")
        if not poles and sp.denom(func) != 1:
            st.warning("Function may have essential singularities or complex poles not detected.")
        return poles, residues
    except Exception as e:
        st.warning(f"Could not compute residues: {str(e)}")
        return [], []
def check_contour_poles(contour_points, poles, min_distance=0.1):
    for pole in poles:
        try:
            distances = np.abs(contour_points - complex(pole))
            if np.any(distances < min_distance):
                st.warning(f"Contour is too close to pole at {complex(pole):.4f}. Numerical integration may be unstable.")
                return False
        except:
            continue
    return True
def is_pole_enclosed(pole, contour_points):
    try:
        pole_complex = complex(pole)
        winding_number = 0
        for j in range(len(contour_points)-1):
            z1 = contour_points[j] - pole_complex
            z2 = contour_points[j+1] - pole_complex
            if abs(z1) < 1e-10 or abs(z2) < 1e-10:
                st.warning(f"Pole at {pole_complex:.4f} is too close to contour. Enclosure check unreliable.")
                return None
            angle = np.angle(z2) - np.angle(z1)
            if angle > np.pi:
                angle -= 2*np.pi
            elif angle < -np.pi:
                angle += 2*np.pi
            winding_number += angle
        turns = winding_number / (2 * np.pi)
        return abs(turns) > 0.5 and abs(abs(turns) - round(abs(turns))) < 0.01
    except:
        st.warning(f"Could not determine if pole at {pole} is inside the contour.")
        return None
def check_self_intersection(points):
    def segments_intersect(p1, p2, q1, q2):
        def ccw(A, B, C):
            return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    n = len(points)
    for i in range(n-1):
        for j in range(i+2, n-1):
            if segments_intersect(points[i], points[i+1], points[j], points[j+1]):
                return True
    return False
def create_contour_path(contour_type, params, num_points=200):
    if contour_type == "Circle":
        center, radius = params['center'], params['radius']
        theta = np.linspace(0, 2*np.pi, num_points)
        return center + radius * np.exp(1j * theta)
    elif contour_type == "Rectangle":
        x_min, x_max, y_min, y_max = params['x_min'], params['x_max'], params['y_min'], params['y_max']
        if x_min >= x_max or y_min >= y_max:
            st.error("Invalid rectangle bounds: Ensure Real Min < Real Max and Imag Min < Imag Max.")
            st.stop()
        points_per_side = num_points // 4
        bottom = np.linspace(x_min + 1j*y_min, x_max + 1j*y_min, points_per_side)
        right = np.linspace(x_max + 1j*y_min, x_max + 1j*y_max, points_per_side)
        top = np.linspace(x_max + 1j*y_max, x_min + 1j*y_max, points_per_side)
        left = np.linspace(x_min + 1j*y_max, x_min + 1j*y_min, points_per_side)
        return np.concatenate([bottom, right, top, left])
    elif contour_type == "Custom":
        points = params['points']
        if len(points) < 3:
            st.error("Custom contour requires at least 3 points.")
            st.stop()
        points.append(points[0])
        if check_self_intersection(points):
            st.warning("Custom contour has self-intersections, which may lead to incorrect results.")
        contour_x = []
        contour_y = []
        points_per_segment = num_points // (len(points) - 1)
        for i in range(len(points) - 1):
            x_seg = np.linspace(points[i].real, points[i+1].real, points_per_segment)
            y_seg = np.linspace(points[i].imag, points[i+1].imag, points_per_segment)
            contour_x.extend(x_seg)
            contour_y.extend(y_seg)
        return np.array(contour_x) + 1j * np.array(contour_y)
    else:
        st.error("Unsupported contour type.")
        st.stop()
def compute_contour_integral(func_numpy, contour_points):
    def integrand(t):
        idx = int(t * (len(contour_points) - 1) / (2 * np.pi))
        z = contour_points[idx]
        dz = contour_points[(idx + 1) % len(contour_points)] - contour_points[idx]
        return func_numpy(z) * dz * len(contour_points) / (2 * np.pi)
    try:
        real, _ = integrate.quad(lambda t: np.real(integrand(t)), 0, 2 * np.pi)
        imag, _ = integrate.quad(lambda t: np.imag(integrand(t)), 0, 2 * np.pi)
        return real + 1j * imag
    except Exception as e:
        st.error(f"Error computing integral: {str(e)}")
        return None
def compute_open_path_integral(func_numpy, start, end, num_points=200):
    t = np.linspace(0, 1, num_points)
    z = start + t * (end - start)
    dz = (end - start) / (num_points - 1)
    f_values = func_numpy(z)
    integral = 0.5 * np.sum((f_values[:-1] + f_values[1:]) * dz)
    return integral
def format_complex(z):
    if z is None:
        return "Error"
    z = complex(z)
    if abs(z.imag) < 1e-10:
        return f"{z.real:.4f}"
    elif abs(z.real) < 1e-10:
        return f"{z.imag:.4f}i"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.4f} {sign} {abs(z.imag):.4f}i"
def hsv_to_rgb(h, s, v):
    h = np.asarray(h)
    s = np.asarray(s)
    v = np.asarray(v)
    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    rgb = np.zeros(h.shape + (3,))
    mask = i == 0
    rgb[mask, 0] = v[mask]
    rgb[mask, 1] = t[mask]
    rgb[mask, 2] = p[mask]
    mask = i == 1
    rgb[mask, 0] = q[mask]
    rgb[mask, 1] = v[mask]
    rgb[mask, 2] = p[mask]
    mask = i == 2
    rgb[mask, 0] = p[mask]
    rgb[mask, 1] = v[mask]
    rgb[mask, 2] = t[mask]
    mask = i == 3
    rgb[mask, 0] = p[mask]
    rgb[mask, 1] = q[mask]
    rgb[mask, 2] = v[mask]
    mask = i == 4
    rgb[mask, 0] = t[mask]
    rgb[mask, 1] = p[mask]
    rgb[mask, 2] = v[mask]
    mask = i == 5
    rgb[mask, 0] = v[mask]
    rgb[mask, 1] = p[mask]
    rgb[mask, 2] = q[mask]
    return rgb
def visualize_complex_function(func_str, domain_min, domain_max, resolution):
    func, z_sym = parse_complex_function(func_str)
    x = np.linspace(domain_min, domain_max, resolution)
    y = np.linspace(domain_min, domain_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    W = evaluate_function(func, z_sym, Z)
    mag = np.abs(W)
    arg = np.angle(W)
    mag = np.log1p(mag) / np.log1p(mag.max() + 1e-10)
    h = (arg + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = mag
    rgb = hsv_to_rgb(h, s, v)
    singularity_mask = np.isnan(W) | np.isinf(W)
    rgb[singularity_mask, 0] = 1.0
    rgb[singularity_mask, 1] = 0.0
    rgb[singularity_mask, 2] = 0.0
    fig, ax = plt.subplots(figsize=get_figure_size())
    ax.imshow(rgb, extent=[domain_min, domain_max, domain_min, domain_max], origin='lower')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(f"Domain Coloring: f(z) = {func_str}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    apply_image_container(fig, f"Domain coloring of f(z) = {func_str}. Hue represents phase, brightness represents magnitude.")
def visualize_integral_contour(func_str, domain_min, domain_max, resolution):
    st.markdown("### Complex Integration Visualizer")
    func, z_sym = parse_complex_function(func_str)
    try:
        st.markdown(f"**Function**: $f(z) = {sp.latex(func)}$")
    except:
        st.markdown(f"**Function**: f(z) = {func_str}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Computing poles and residues...")
    progress_bar.progress(20)
    poles, residues = compute_residues(func, z_sym, domain_min, domain_max)
    status_text.text("Creating visualization grid...")
    progress_bar.progress(40)
    x = np.linspace(domain_min, domain_max, resolution)
    y = np.linspace(domain_min, domain_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    W = evaluate_function(func, z_sym, Z)
    mag = np.abs(W)
    mag = np.where(np.isnan(mag) | np.isinf(mag), mag.max(), mag)
    log_mag = np.log1p(mag)
    st.markdown("### Choose a Contour")
    contour_type = st.selectbox("Contour Type", ["Circle", "Rectangle", "Custom", "Open Path"], index=0)
    contour_params = {}
    contour_points = None
    is_closed = contour_type != "Open Path"
    if contour_type == "Circle":
        col1, col2, col3 = st.columns(3)
        with col1:
            center_x = st.number_input("Center (Real)", min_value=float(domain_min), max_value=float(domain_max), value=0.0, step=0.1, format="%.2f")
        with col2:
            center_y = st.number_input("Center (Imag)", min_value=float(domain_min), max_value=float(domain_max), value=0.0, step=0.1, format="%.2f")
        with col3:
            radius = st.number_input("Radius", min_value=0.1, max_value=float(domain_max - domain_min)/2, value=1.0, step=0.1, format="%.2f")
        contour_params = {'center': complex(center_x, center_y), 'radius': radius}
        st.markdown(f"**Contour**: $C = {{z : |z - ({center_x} + {center_y}i)| = {radius}}}$ (Counterclockwise)")
        contour_points = create_contour_path(contour_type, contour_params)
    elif contour_type == "Rectangle":
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("Real Min", min_value=float(domain_min), max_value=float(domain_max), value=float(domain_min/2), step=0.1, format="%.2f")
            y_min = st.number_input("Imag Min", min_value=float(domain_min), max_value=float(domain_max), value=float(domain_min/2), step=0.1, format="%.2f")
        with col2:
            x_max = st.number_input("Real Max", min_value=float(domain_min), max_value=float(domain_max), value=float(domain_max/2), step=0.1, format="%.2f")
            y_max = st.number_input("Imag Max", min_value=float(domain_min), max_value=float(domain_max), value=float(domain_max/2), step=0.1, format="%.2f")
        contour_params = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        st.markdown(f"**Contour**: Rectangle with vertices at $({x_min} + {y_min}i)$, $({x_max} + {y_min}i)$, $({x_max} + {y_max}i)$, $({x_min} + {y_max}i)$ (Counterclockwise)")
        contour_points = create_contour_path(contour_type, contour_params)
    elif contour_type == "Custom":
        if "custom_points" not in st.session_state:
            st.session_state.custom_points = [complex(0,0), complex(1,0), complex(1,1), complex(0,1)]
        st.markdown("Define a custom contour by adding points below:")
        points_text = "\n".join([f"{i+1}. ({p.real:.2f}, {p.imag:.2f})" for i, p in enumerate(st.session_state.custom_points)])
        st.text_area("Contour Points", points_text, height=100, disabled=True)
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_x = st.number_input("X", value=0.0, step=0.1, format="%.2f")
        with col2:
            new_y = st.number_input("Y", value=0.0, step=0.1, format="%.2f")
        with col3:
            if st.button("Add Point"):
                st.session_state.custom_points.append(complex(new_x, new_y))
                st.rerun()
        if len(st.session_state.custom_points) > 3 and st.button("Remove Last Point"):
            st.session_state.custom_points.pop()
            st.rerun()
        if st.button("Reset to Square"):
            st.session_state.custom_points = [complex(0,0), complex(1,0), complex(1,1), complex(0,1)]
            st.rerun()
        contour_params = {'points': st.session_state.custom_points}
        points_str = ", ".join([f"({p.real:.2f} + {p.imag:.2f}i)" for p in contour_params['points']])
        st.markdown(f"**Contour**: Connected path through {points_str} (Counterclockwise)")
        contour_points = create_contour_path(contour_type, contour_params)
    elif contour_type == "Open Path":
        col1, col2 = st.columns(2)
        with col1:
            start_x = st.number_input("Start Real", min_value=float(domain_min), max_value=float(domain_max), value=float(domain_min/2), step=0.1, format="%.2f")
            start_y = st.number_input("Start Imag", min_value=float(domain_min), max_value=float(domain_max), value=0.0, step=0.1, format="%.2f")
        with col2:
            end_x = st.number_input("End Real", min_value=float(domain_min), max_value=float(domain_max), value=float(domain_max/2), step=0.1, format="%.2f")
            end_y = st.number_input("End Imag", min_value=float(domain_min), max_value=float(domain_max), value=0.0, step=0.1, format="%.2f")
        start = complex(start_x, start_y)
        end = complex(end_x, end_y)
        st.markdown(f"**Path**: Line from $({start_x} + {start_y}i)$ to $({end_x} + {end_y}i)$")
        st.info("Open-path integrals depend on the path and endpoints, not the Residue Theorem.")
        t = np.linspace(0, 1, 200)
        contour_points = start + t * (end - start)
    status_text.text("Computing integral...")
    progress_bar.progress(60)
    func_numpy = lambdify(z_sym, func, modules=["numpy"])
    integral_value = None
    integral_str = "Error"
    if is_closed:
        if check_contour_poles(contour_points, poles):
            integral_value = compute_contour_integral(func_numpy, contour_points)
            integral_str = format_complex(integral_value)
    else:
        integral_value = compute_open_path_integral(func_numpy, start, end)
        integral_str = format_complex(integral_value)
    enclosed_poles = []
    enclosed_residues = []
    if is_closed and poles:
        status_text.text("Checking pole enclosure...")
        progress_bar.progress(80)
        for pole, residue in zip(poles, residues):
            if is_pole_enclosed(pole, contour_points):
                enclosed_poles.append(pole)
                enclosed_residues.append(residue)
    status_text.text("Creating visualization...")
    progress_bar.progress(90)
    fig, ax = plt.subplots(figsize=get_figure_size())
    arg = np.angle(W)
    h = (arg + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = log_mag / np.log1p(log_mag.max() + 1e-10)
    rgb = hsv_to_rgb(h, s, v)
    singularity_mask = np.isnan(W) | np.isinf(W)
    rgb[singularity_mask, 0] = 1.0
    rgb[singularity_mask, 1] = 0.0
    rgb[singularity_mask, 2] = 0.0
    ax.imshow(rgb, extent=[domain_min, domain_max, domain_min, domain_max], origin='lower')
    if poles:
        pole_x = [complex(pole).real for pole in poles]
        pole_y = [complex(pole).imag for pole in poles]
        ax.scatter(pole_x, pole_y, color='white', s=100, marker='x', label='Poles')
    contour_x = np.real(contour_points)
    contour_y = np.imag(contour_points)
    ax.plot(contour_x, contour_y, 'r-', linewidth=2, label='Contour')
    if is_closed:
        num_arrows = 8
        arrow_indices = np.linspace(0, len(contour_x)-1, num_arrows, dtype=int)
        for i in arrow_indices:
            idx = i % len(contour_x)
            next_idx = (idx + 5) % len(contour_x)
            dx = contour_x[next_idx] - contour_x[idx]
            dy = contour_y[next_idx] - contour_y[idx]
            ax.arrow(contour_x[idx], contour_y[idx], dx*0.1, dy*0.1, 
                  head_width=0.1, head_length=0.15, fc='r', ec='r')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    closed_integral_latex = r'$\oint_C f(z) dz$'
    open_integral_latex = r'$\int_C f(z) dz$'
    integral_latex = closed_integral_latex if is_closed else open_integral_latex
    ax.set_title(f"Integral: {integral_latex} = {integral_str}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    progress_bar.progress(100)
    status_text.text("Visualization complete!")
    apply_image_container(fig, f"Integral of f(z) = {func_str} along {'contour' if is_closed else 'path'} C")
    with st.expander("Integral Results"):
        st.markdown(f"**Numerical Integral**: {integral_latex} = {integral_str}")
        if is_closed:
            if enclosed_poles:
                expected_value = 2 * np.pi * 1j * sum(enclosed_residues)
                expected_str = format_complex(expected_value)
                st.markdown(f"**Theoretical Integral (Residue Theorem)**: $2\\pi i \\sum \\text{{Res}} = {expected_str}$")
                if integral_value is not None:
                    error = abs(integral_value - expected_value)
                    st.markdown(f"**Absolute Error**: {error:.4e}")
                    if abs(expected_value) > 1e-10:
                        rel_error = error / abs(expected_value)
                        st.markdown(f"**Relative Error**: {rel_error:.4e}")
                    if error < 1e-6:
                        st.success("Numerical result matches theoretical prediction with high accuracy!")
                    elif error < 1e-3:
                        st.info("Numerical result is reasonably close to theoretical prediction.")
                    else:
                        st.warning("Numerical result differs significantly. Check contour proximity to poles.")
                st.markdown("### Enclosed Poles and Residues")
                for i, (pole, residue) in enumerate(zip(enclosed_poles, enclosed_residues)):
                    st.markdown(f"**Pole {i+1}**: z = {format_complex(pole)}, Residue = {format_complex(residue)}")
            else:
                st.markdown("**No poles enclosed. Integral should be zero.**")
                if integral_value is not None and abs(integral_value) < 1e-6:
                    st.success("Numerical result is correctly close to zero!")
                elif integral_value is not None:
                    st.warning(f"Numerical result ({integral_str}) is not close to zero. Check contour proximity to poles.")
        else:
            st.markdown("**Note**: Open-path integrals depend on the path and function. No Residue Theorem verification is performed.")
def visualize_linear_transformation(matrix_str, domain_min, domain_max, resolution):
    try:
        matrix = np.array(eval(matrix_str), dtype=complex)
        if matrix.shape != (2, 2):
            st.error("Matrix must be 2x2.")
            st.stop()
        x = np.linspace(domain_min, domain_max, resolution)
        y = np.linspace(domain_min, domain_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        Z_flat = Z.flatten()
        W = np.array([matrix @ [z.real, z.imag] for z in Z_flat])
        W = W[:, 0] + 1j * W[:, 1]
        W = W.reshape(Z.shape)
        mag = np.abs(W)
        arg = np.angle(W)
        mag = np.log1p(mag) / np.log1p(mag.max() + 1e-10)
        h = (arg + np.pi) / (2 * np.pi)
        s = np.ones_like(h)
        v = mag
        rgb = hsv_to_rgb(h, s, v)
        fig, ax = plt.subplots(figsize=get_figure_size())
        ax.imshow(rgb, extent=[domain_min, domain_max, domain_min, domain_max], origin='lower')
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(f"Linear Transformation: A = {matrix.tolist()}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        apply_image_container(fig, f"Linear transformation z -> Az with A = {matrix.tolist()}")
    except Exception as e:
        st.error(f"Error parsing matrix: {str(e)}")
def main():
    st.title("Complex Variables and Linear Algebra Explorer")
    st.markdown("Select one of the models below to explore complex analysis and linear algebra concepts.")
    models = {
        "Complex Visualizer": {
            "description": "Visualize complex functions using domain coloring.",
            "examples": ["z**2", "sin(z)", "exp(z)", "1/(z-1)"],
            "default": "z**2"
        },
        "Complex Integration": {
            "description": "Compute and visualize contour integrals.",
            "examples": ["1/(z-1)", "z**2", "exp(z)/z"],
            "default": "1/(z-1)"
        },
        "Linear Transformation": {
            "description": "Visualize linear transformations in the complex plane.",
            "examples": ["[[1, 0], [0, 1]]", "[[0, -1], [1, 0]]", "[[2, 1], [1, 2]]"],
            "default": "[[1, 0], [0, 1]]"
        }
    }
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Complex Visualizer"
    st.markdown("### Select a Model")
    tabs = st.tabs(list(models.keys()))
    for i, (model_name, model_info) in enumerate(models.items()):
        with tabs[i]:
            st.markdown(f"**{model_name}**")
            st.markdown(model_info["description"])
            if st.button(f"Select {model_name}", key=f"select_{model_name}"):
                st.session_state.selected_model = model_name
                st.rerun()
    with st.form(key=f"{st.session_state.selected_model}_form"):
        selected_model = st.session_state.selected_model
        st.markdown(f"**Selected**: {selected_model}")
        model_info = models[selected_model]
        st.markdown("### Enter Parameters")
        example_cols = st.columns(4)
        for i, example in enumerate(model_info["examples"]):
            if example_cols[i % 4].button(example, key=f"example_{i}"):
                st.session_state["input"] = example
        with st.form(key="input_form"):
            if selected_model == "Linear Transformation":
                input_value = st.text_area(
                    "2x2 Matrix (as [[a, b], [c, d]])",
                    value=st.session_state.get("input", model_info["default"]),
                    help="Enter a 2x2 matrix, e.g., [[1, 0], [0, 1]]"
                )
            else:
                input_value = st.text_input(
                    "f(z) = ",
                    value=st.session_state.get("input", model_info["default"]),
                    help="Enter a complex function using z, e.g., z**2"
                )
            col1, col2, col3 = st.columns(3)
            with col1:
                domain_min = st.slider("Domain Min", -10.0, -0.1, -5.0)
            with col2:
                domain_max = st.slider("Domain Max", 0.1, 10.0, 5.0)
            with col3:
                resolution = st.slider("Resolution", 100, 1000, 500, step=100)
            submit = st.form_submit_button("Visualize")
            if submit:
                if domain_min >= domain_max:
                    st.error("Domain Min must be less than Domain Max.")
                    st.stop()
                try:
                    st.session_state["input"] = input_value
                    if selected_model == "Complex Visualizer":
                        visualize_complex_function(input_value, domain_min, domain_max, resolution)
                    elif selected_model == "Complex Integration":
                        visualize_integral_contour(input_value, domain_min, domain_max, resolution)
                    elif selected_model == "Linear Transformation":
                        visualize_linear_transformation(input_value, domain_min, domain_max, resolution)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
if __name__ == "__main__":
    main()

