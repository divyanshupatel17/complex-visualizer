import os
def check_file_structure():
    files_to_check = [
        "app.py",
        "requirements.txt",
        "README.md",
        "utils/sidebar.py",
        "baskets/basket1/complex_mapping.py",
        "models/complex_function_grapher.py",
        "models/harmonic_flow_predictor.py",
        "models/conformal_map_animator.py",
        "models/integral_contour_interpreter.py"
    ]
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("✅ All required files exist!")
    print("\nProject structure is ready for development and deployment.")
    print("To run the application, use: streamlit run app.py")
if __name__ == "__main__":
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the complex-visualizer directory.")
    else:
        check_file_structure() 

