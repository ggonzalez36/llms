@echo off
setlocal

set "CONDA_EXE=C:\Users\User\Documents\Playground\tools\miniforge3\Scripts\conda.exe"
set "SCRIPT=%~dp0chapter02_5_rag_run.py"

if not exist "%CONDA_EXE%" (
  echo [ERROR] No se encontro conda en:
  echo         %CONDA_EXE%
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo [ERROR] No se encontro el script:
  echo         %SCRIPT%
  exit /b 1
)

if "%~1"=="" (
  echo [INFO] Ejecutando Chapter 2.5 RAG con datos de ejemplo...
  "%CONDA_EXE%" run -n holllm python "%SCRIPT%" --cache-home "%~dp0.chroma_home_%USERNAME%"
  exit /b %ERRORLEVEL%
)

"%CONDA_EXE%" run -n holllm python "%SCRIPT%" %*
exit /b %ERRORLEVEL%
