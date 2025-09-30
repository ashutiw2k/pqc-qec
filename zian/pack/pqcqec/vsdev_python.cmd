@echo off
REM VS Dev + Conda wrapper for training_best0926_changing.py (and other Python runs)
REM 1. Load Visual Studio developer environment (cl.exe, link.exe, lib paths, Windows SDK)
IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
) ELSE (
  echo [WARN] VsDevCmd.bat not found at default BuildTools location.
)

REM 2. (Optional) Set a CUDA arch list to reduce build time if not already set
IF "%TORCH_CUDA_ARCH_LIST%"=="" (
  REM You can adjust the architecture below (example 8.6 for Ampere, 8.9 for Ada, 7.5 Turing, etc.)
  SET TORCH_CUDA_ARCH_LIST=8.6
)

REM 3. Activate conda environment if available (pc2)
IF EXIST "A:\miniconda3\Scripts\activate.bat" (
  call "A:\miniconda3\Scripts\activate.bat" pc2
) ELSE (
  echo [WARN] Conda activate script not found at A:\miniconda3\Scripts\activate.bat
)

REM 3.5 Configure CUDA Toolkit (prefer 12.6 if installed)
SET "_CUDA_BASE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
IF EXIST "%_CUDA_BASE%\v12.6\bin\nvcc.exe" (
  SET "CUDA_VER=12.6"
) ELSE IF EXIST "%_CUDA_BASE%\v12.5\bin\nvcc.exe" (
  SET "CUDA_VER=12.5"
) ELSE IF EXIST "%_CUDA_BASE%\v12.4\bin\nvcc.exe" (
  SET "CUDA_VER=12.4"
) ELSE (
  REM fallback: pick first dir that matches v12.*
  FOR /F "delims=" %%d IN ('dir /b /ad "%_CUDA_BASE%" ^| findstr /r /c:"^v12\.[0-9]"') DO (
    IF NOT DEFINED CUDA_VER SET "CUDA_VER=%%d"
  )
)
IF DEFINED CUDA_VER (
  IF EXIST "%_CUDA_BASE%\v%CUDA_VER%\bin\nvcc.exe" (
    SET "CUDA_HOME=%_CUDA_BASE%\v%CUDA_VER%"
    SET "CUDA_PATH=%CUDA_HOME%"
    REM Put desired CUDA bin first in PATH (do not try to surgically remove others here)
    SET "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
  ) ELSE (
    echo [WARN] Chosen CUDA_VER=%CUDA_VER% but nvcc.exe missing under %_CUDA_BASE%\v%CUDA_VER%\bin
  )
) ELSE (
  echo [WARN] Could not detect a CUDA v12.x installation under %_CUDA_BASE%
)

REM 4. Print brief diagnostics (first run only)
IF NOT DEFINED _VSDEV_WRAPPER_SHOWN (
  echo [INFO] vsdev_python.cmd initialized. Using:
  where cl >nul 2>nul && (for /f "delims=" %%i in ('where cl') do echo    cl: %%i) || echo    cl: NOT FOUND
  where nvcc >nul 2>nul && (for /f "delims=" %%i in ('where nvcc') do echo    nvcc: %%i) || echo    nvcc: NOT FOUND
  echo    TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
  IF DEFINED CUDA_HOME echo    CUDA_HOME=%CUDA_HOME%
  IF DEFINED CUDA_VER echo    CUDA_VER=%CUDA_VER%
  SET _VSDEV_WRAPPER_SHOWN=1
)

REM 5. Delegate to python with all original args
python %*
