import streamlit as st
import base64

def create_sidebar():
    with st.sidebar:
        # Add course information card at the top
        st.markdown("""
        <div style="background-color: rgba(35,35,50,0.7); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid rgba(100,149,237,0.5);">
            <h3 style="text-align: center; color: #00f3ff; font-size: 1.1rem; margin-bottom: 10px;">CVLA INTERACTIVE AI LAB</h3>
            <p style="text-align: center; color: #ECEFCA; font-size: 0.8rem; margin-bottom: 10px;">AN AI-POWERED EDUCATIONAL PLATFORM FOR COMPLEX VARIABLES & LINEAR ALGEBRA</p>
            <hr style="border-color: rgba(100,149,237,0.5); margin: 10px 0;">
            <p style="color: #ECEFCA; font-size: 0.8rem; margin: 5px 0;"><strong>Course:</strong> Complex Variable and Linear Algebra</p>
            <p style="color: #ECEFCA; font-size: 0.8rem; margin: 5px 0;"><strong>Faculty:</strong> Vijay Kumar P</p>
            <p style="color: #ECEFCA; font-size: 0.8rem; margin: 5px 0;"><strong>Students:</strong></p>
            <ul style="color: #ECEFCA; font-size: 0.8rem; margin: 0; padding-left: 20px;">
                <li>Akshat Pal (23BRS1353)</li>
                <li>Ashutosh Gunjal (23BRS1354)</li>
                <li>Divyanshu Patel (23BAI1214)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("", unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)
        if "selected_basket" not in st.session_state:
            st.session_state["selected_basket"] = "Complex Mapping"
        
        st.markdown("### Visualization Modules")
        baskets = [
            {"name": "Complex Mapping", "icon": "📊", "description": "Complex Mapping & Analytic Function Simulator"},
            {"name": "Matrixland", "icon": "🧩", "description": "Matrixland & Vector Playground"},
            {"name": "Eigen Exploratorium", "icon": "🔍", "description": "Eigen Exploratorium"},
            {"name": "Inner Product Lab", "icon": "⚙️", "description": "Inner Product & Orthonormalization Lab"}
        ]
        for basket in baskets:
            is_selected = st.session_state["selected_basket"] == basket["name"]
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<div style='font-size:18px;'>{basket['icon']}</div>", unsafe_allow_html=True)
            with col2:
                button_label = f"{basket['name']}"
                if st.button(
                    button_label, 
                    key=f"btn_{basket['name']}", 
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state["selected_basket"] = basket["name"]
                    st.rerun()
                if basket["name"] == st.session_state["selected_basket"]:
                    st.markdown(f"<p style='margin: 0 0 6px 0; font-size: 0.75rem; color: rgba(236, 239, 202, 0.8);'>{basket['description']}</p>", unsafe_allow_html=True)
        
        st.markdown("", unsafe_allow_html=True)
        st.markdown("### Display Settings")
        st.markdown("", unsafe_allow_html=True)
        if "default_image_size" not in st.session_state:
            st.session_state["default_image_size"] = "medium"
        st.session_state["default_image_size"] = st.select_slider(
            "Default Image Size",
            options=["small", "medium", "large"],
            value=st.session_state["default_image_size"]
        )
        if "visualization_size" not in st.session_state or st.session_state["visualization_size"] != st.session_state["default_image_size"]:
            st.session_state["visualization_size"] = st.session_state["default_image_size"]
        with st.expander("About Image Controls"):
            st.markdown("", unsafe_allow_html=True)
        
        st.markdown("", unsafe_allow_html=True)
        basket_title = ""
        basket_content = []
        if st.session_state["selected_basket"] == "Complex Mapping":
            basket_title = "Complex Mapping & Function Simulator"
            basket_content = ["Complex Visualizer", "Complex Integration"]
        elif st.session_state["selected_basket"] == "Matrixland":
            basket_title = "Matrixland & Vector Playground"
            basket_content = ["Vector Spaces", "Transformations"]
        elif st.session_state["selected_basket"] == "Eigen Exploratorium":
            basket_title = "Eigen Exploratorium"
            basket_content = ["Eigenvalues & Eigenvectors", "Matrix Equations"]
        elif st.session_state["selected_basket"] == "Inner Product Lab":
            basket_title = "Inner Product Lab"
            basket_content = ["Inner Products", "Gram-Schmidt Process"]
        
        st.markdown(f"", unsafe_allow_html=True)
        for item in basket_content:
            st.markdown(f"", unsafe_allow_html=True)
        with st.expander("About CVLA Lab"):
            st.markdown("", unsafe_allow_html=True)
    
    return st.session_state["selected_basket"] 

