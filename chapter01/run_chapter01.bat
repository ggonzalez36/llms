@echo off
setlocal

set "CONDA_EXE=C:\Users\User\Documents\Playground\tools\miniforge3\Scripts\conda.exe"
set "SCRIPT=%~dp0chapter01_run.py"

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
  echo [INFO] Ejecutando prueba rapida en CPU con un modelo pequeno...
  "%CONDA_EXE%" run -n holllm python "%SCRIPT%" --model sshleifer/tiny-gpt2 --device cpu --local-files-only --no-chat --sample --max-new-tokens 40 --repetition-penalty 1.2 --no-repeat-ngram-size 2 --prompt "Write one short funny sentence about learning language models."
  exit /b %ERRORLEVEL%
)

"%CONDA_EXE%" run -n holllm python "%SCRIPT%" %*
exit /b %ERRORLEVEL%
