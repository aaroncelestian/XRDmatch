@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM XRDmatch step-by-step installer for Windows
REM Checks for Anaconda/Miniconda, installs Miniconda if missing, creates env + Desktop shortcut.

set "ENV_NAME=xrdmatch"
set "PYTHON_VERSION=3.11"
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ==============================================
echo   XRDmatch — Step-by-step installer (Windows)
echo ==============================================
echo.
echo This will:
echo   1. Check for Anaconda / Miniconda (conda^)
echo   2. Install Miniconda if conda is missing
echo   3. Create conda env "%ENV_NAME%" (Python %PYTHON_VERSION%^)
echo   4. Install dependencies from requirements.txt
echo   5. Create a Desktop shortcut
echo.
echo Project folder: %SCRIPT_DIR%
echo.
pause

REM ---------- find conda ----------
echo.
echo Step 1 — Looking for Anaconda / conda
set "CONDA_EXE="

where conda >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%i in ('where conda') do (
    set "CONDA_EXE=%%i"
    goto :conda_found
  )
)

for %%P in (
  "%USERPROFILE%\miniconda3\Scripts\conda.exe"
  "%USERPROFILE%\anaconda3\Scripts\conda.exe"
  "%USERPROFILE%\Miniconda3\Scripts\conda.exe"
  "%USERPROFILE%\Anaconda3\Scripts\conda.exe"
  "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
  "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
  "C:\ProgramData\miniconda3\Scripts\conda.exe"
  "C:\ProgramData\anaconda3\Scripts\conda.exe"
) do (
  if exist %%~P (
    set "CONDA_EXE=%%~P"
    goto :conda_found
  )
)

echo   ! conda was not found.
echo   XRDmatch needs Anaconda or Miniconda.
echo   Full Anaconda: https://www.anaconda.com/download
echo.
set /p INSTALL_MC="  Install Miniconda now? [Y/n] "
if /I "%INSTALL_MC%"=="n" (
  echo Cannot continue without conda.
  echo Install Anaconda, open a new terminal, and re-run install.bat
  exit /b 1
)

echo.
echo Step 1b — Download Miniconda
set "INSTALLER=%TEMP%\Miniconda3-XRDmatch-install.exe"
set "MC_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%MC_URL%' -OutFile '%INSTALLER%'"
if errorlevel 1 (
  echo Download failed. Install manually from https://www.anaconda.com/download
  exit /b 1
)

echo Step 1c — Run Miniconda installer
start /wait "" "%INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniconda3
del /q "%INSTALLER%" 2>nul
set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
  echo Miniconda install finished but conda.exe was not found.
  exit /b 1
)

:conda_found
echo   OK Found conda: %CONDA_EXE%
for /f "delims=" %%b in ('"%CONDA_EXE%" info --base') do set "CONDA_BASE=%%b"
echo   OK conda base: %CONDA_BASE%

REM ---------- env ----------
echo.
echo Step 2 — Create conda environment "%ENV_NAME%"
"%CONDA_EXE%" env list | findstr /R /C:"^%ENV_NAME% " >nul
if not errorlevel 1 (
  echo   ! Environment "%ENV_NAME%" already exists.
  set /p RECREATE="  Recreate it from scratch? [y/N] "
  if /I "!RECREATE!"=="y" (
    "%CONDA_EXE%" env remove -y -n %ENV_NAME%
    "%CONDA_EXE%" create -y -n %ENV_NAME% python=%PYTHON_VERSION%
  ) else (
    echo   OK Keeping existing env
  )
) else (
  "%CONDA_EXE%" create -y -n %ENV_NAME% python=%PYTHON_VERSION%
)

REM ---------- deps ----------
echo.
echo Step 3 — Install Python packages
call "%CONDA_BASE%\condabin\conda.bat" activate %ENV_NAME%
if errorlevel 1 (
  echo Failed to activate %ENV_NAME%
  exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
  echo Dependency install failed.
  exit /b 1
)

REM ---------- verify ----------
echo.
echo Step 4 — Verify installation
python -c "import PyQt5,matplotlib,numpy,pandas,scipy,requests,bs4,lxml,gemmi,pymatgen; print('  OK imports')"
if errorlevel 1 (
  echo Some packages failed to import.
  exit /b 1
)

REM ---------- desktop shortcut ----------
echo.
echo Step 5 — Create Desktop shortcut
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\create_windows_desktop_shortcut.ps1" -ProjectRoot "%SCRIPT_DIR%." -EnvName %ENV_NAME%
if errorlevel 1 (
  echo Warning: could not create Desktop shortcut.
) else (
  echo   OK Desktop shortcut created
)

echo.
echo ==============================================
echo   Done — XRDmatch is ready
echo ==============================================
echo.
echo Double-click "XRDmatch" on your Desktop, or run:
echo   conda activate %ENV_NAME%
echo   python main.py
echo.
pause
endlocal
