@echo off
echo ============================================
echo    COMPLEX VISUALIZER - DEPLOYMENT TOOL
echo ============================================
echo.
echo This script will prepare your project for deployment.
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed! Please install Python first.
    echo Go to https://www.python.org/downloads/ to download Python.
    pause
    exit /b
)

REM Create a deployment folder
set DEPLOY_FOLDER=Complex-Visualizer-Deploy
echo Creating deployment folder: %DEPLOY_FOLDER%
if exist %DEPLOY_FOLDER% (
    echo Removing existing deployment folder...
    rmdir /s /q %DEPLOY_FOLDER%
)
mkdir %DEPLOY_FOLDER%

REM Copy project files to deployment folder
echo Copying project files...
xcopy *.py %DEPLOY_FOLDER%\ /Y
xcopy *.txt %DEPLOY_FOLDER%\ /Y

REM Create necessary directories
if exist baskets (
    echo Copying baskets folder...
    xcopy baskets %DEPLOY_FOLDER%\baskets\ /E /I /Y
)

if exist models (
    echo Copying models folder...
    xcopy models %DEPLOY_FOLDER%\models\ /E /I /Y
)

if exist utils (
    echo Copying utils folder...
    xcopy utils %DEPLOY_FOLDER%\utils\ /E /I /Y
)

if exist assets (
    echo Copying assets folder...
    xcopy assets %DEPLOY_FOLDER%\assets\ /E /I /Y
)

REM Create requirements.txt file
echo Creating requirements.txt...
echo streamlit>=1.26.0 > %DEPLOY_FOLDER%\requirements.txt
echo numpy>=1.24.0 >> %DEPLOY_FOLDER%\requirements.txt
echo scipy>=1.10.0 >> %DEPLOY_FOLDER%\requirements.txt
echo pandas>=1.5.0 >> %DEPLOY_FOLDER%\requirements.txt
echo plotly>=5.13.0 >> %DEPLOY_FOLDER%\requirements.txt
echo pillow>=9.4.0 >> %DEPLOY_FOLDER%\requirements.txt
echo matplotlib>=3.7.0 >> %DEPLOY_FOLDER%\requirements.txt
echo sympy>=1.11.0 >> %DEPLOY_FOLDER%\requirements.txt

REM Create START_HERE.bat file
echo Creating START_HERE.bat file...
echo @echo off > %DEPLOY_FOLDER%\START_HERE.bat
echo echo ============ COMPLEX VISUALIZER ============ >> %DEPLOY_FOLDER%\START_HERE.bat
echo python --version ^> nul 2^>^&1 >> %DEPLOY_FOLDER%\START_HERE.bat
echo if %%ERRORLEVEL%% neq 0 ( >> %DEPLOY_FOLDER%\START_HERE.bat
echo     echo Python is not installed! Please install Python first. >> %DEPLOY_FOLDER%\START_HERE.bat
echo     echo Go to https://www.python.org/downloads/ to download Python. >> %DEPLOY_FOLDER%\START_HERE.bat
echo     echo IMPORTANT: Make sure to check "Add Python to PATH" during installation. >> %DEPLOY_FOLDER%\START_HERE.bat
echo     pause >> %DEPLOY_FOLDER%\START_HERE.bat
echo     exit /b >> %DEPLOY_FOLDER%\START_HERE.bat
echo ) >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo Python found! Setting up your visualization environment... >> %DEPLOY_FOLDER%\START_HERE.bat
echo pip install -r requirements.txt >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo. >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo Starting the Complex Visualizer... >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo The program will open in your web browser automatically. >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo. >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo If you need to stop the application, come back to this window >> %DEPLOY_FOLDER%\START_HERE.bat
echo echo and press CTRL+C, then close your browser. >> %DEPLOY_FOLDER%\START_HERE.bat
echo streamlit run app.py >> %DEPLOY_FOLDER%\START_HERE.bat
echo pause >> %DEPLOY_FOLDER%\START_HERE.bat

REM Create a README.txt file
echo Creating README.txt...
echo =================================================== > %DEPLOY_FOLDER%\README.txt
echo COMPLEX VISUALIZER - README >> %DEPLOY_FOLDER%\README.txt
echo =================================================== >> %DEPLOY_FOLDER%\README.txt
echo. >> %DEPLOY_FOLDER%\README.txt
echo Thank you for downloading the Complex Visualizer! >> %DEPLOY_FOLDER%\README.txt
echo. >> %DEPLOY_FOLDER%\README.txt
echo TO START THE APPLICATION: >> %DEPLOY_FOLDER%\README.txt
echo 1. Make sure Python is installed on your computer >> %DEPLOY_FOLDER%\README.txt
echo    (Download from https://www.python.org/downloads/) >> %DEPLOY_FOLDER%\README.txt
echo 2. Double-click on the START_HERE.bat file >> %DEPLOY_FOLDER%\README.txt
echo. >> %DEPLOY_FOLDER%\README.txt
echo The application will automatically: >> %DEPLOY_FOLDER%\README.txt
echo - Check if Python is installed >> %DEPLOY_FOLDER%\README.txt
echo - Install required libraries >> %DEPLOY_FOLDER%\README.txt
echo - Start the visualization application in your web browser >> %DEPLOY_FOLDER%\README.txt
echo. >> %DEPLOY_FOLDER%\README.txt
echo FEATURES: >> %DEPLOY_FOLDER%\README.txt
echo - Complex function visualization with domain coloring >> %DEPLOY_FOLDER%\README.txt
echo - Matrix operations and transformations >> %DEPLOY_FOLDER%\README.txt
echo - Eigenvalue and eigenvector exploration >> %DEPLOY_FOLDER%\README.txt
echo - Inner product visualizations >> %DEPLOY_FOLDER%\README.txt
echo - And much more! >> %DEPLOY_FOLDER%\README.txt
echo. >> %DEPLOY_FOLDER%\README.txt
echo Enjoy exploring complex mathematics visually! >> %DEPLOY_FOLDER%\README.txt

REM Create a ZIP file of the deployment folder
echo Creating ZIP file for easy sharing...
powershell Compress-Archive -Path %DEPLOY_FOLDER% -DestinationPath %DEPLOY_FOLDER%.zip -Force

echo.
echo ============================================
echo DEPLOYMENT COMPLETE!
echo ============================================
echo.
echo Your deployment package has been created:
echo - Folder: %DEPLOY_FOLDER%
echo - ZIP file: %DEPLOY_FOLDER%.zip
echo.
echo You can now share the ZIP file with others.
echo Recipients just need to:
echo 1. Extract the ZIP file
echo 2. Double-click START_HERE.bat to run the application
echo.
echo Press any key to exit...
pause > nul 