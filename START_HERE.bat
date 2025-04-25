@echo off
echo ============ COMPLEX VISUALIZER ============
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed! Please install Python first.
    echo Go to https://www.python.org/downloads/ to download Python.
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)
echo Python found! Setting up your visualization environment...
pip install streamlit numpy scipy pandas plotly pillow matplotlib
echo.
echo Starting the Complex Visualizer...
echo The program will open in your web browser automatically.
echo.
echo If you need to stop the application, come back to this window
echo and press CTRL+C, then close your browser.
streamlit run app.py
pause 