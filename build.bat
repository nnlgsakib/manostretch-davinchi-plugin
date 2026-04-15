@echo off
setlocal

echo ============================================
echo   ManoStretch v4.0 - Build Script
echo ============================================
echo.

:: Add CMake and CUDA to PATH
set "PATH=C:\Program Files\CMake\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%"

:: Check CMake
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] CMake not found. Install from https://cmake.org
    pause
    exit /b 1
)

:: Check for Visual Studio Build Tools
set "GENERATOR=Visual Studio 17 2022"
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] cl.exe not in PATH, will rely on CMake to find VS...
)

:: Configure
echo [1/3] Configuring...
if not exist build (
    cmake -B build -G "%GENERATOR%" -A x64
    if %errorlevel% neq 0 (
        echo [ERROR] CMake configure failed.
        pause
        exit /b 1
    )
)

:: Build
echo.
echo [2/3] Building Release...
cmake --build build --config Release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

:: Show results
echo.
echo [3/3] Build complete!
echo.
echo   Plugin:    build\Release\ManoStretch.ofx
echo   Installer: build\Release\ManoStretch_Installer.exe
echo.

:: Check output exists
if exist "build\Release\ManoStretch.ofx" (
    for %%F in ("build\Release\ManoStretch.ofx") do echo   Plugin size: %%~zF bytes
)
if exist "build\Release\ManoStretch_Installer.exe" (
    for %%F in ("build\Release\ManoStretch_Installer.exe") do echo   Installer size: %%~zF bytes
)

echo.
echo ============================================
echo   To install: run ManoStretch_Installer.exe
echo   To clean:   build.bat clean
echo ============================================

if "%1"=="clean" (
    echo Cleaning build directory...
    rmdir /s /q build 2>nul
    echo Done.
)

pause
