# Chapter 1 local setup (Windows)

This machine is ready to run Chapter 1 from:

- `C:\Users\User\Documents\Playground\Hands-On-Large-Language-Models\chapter01`

## 1) Run Chapter 1 as a Python script (recommended)

Use the executable script:

- `C:\Users\User\Documents\Playground\Hands-On-Large-Language-Models\chapter01\chapter01_run.py`

Run a quick CPU test:

```powershell
C:\Users\User\Documents\Playground\tools\miniforge3\Scripts\conda.exe run -n holllm python `
  C:\Users\User\Documents\Playground\Hands-On-Large-Language-Models\chapter01\chapter01_run.py `
  --model sshleifer/tiny-gpt2 --device cpu --local-files-only --no-chat --sample --max-new-tokens 40 `
  --repetition-penalty 1.2 --no-repeat-ngram-size 2 `
  --prompt "Write one short funny sentence about learning language models."
```

Run with the original Chapter 1 model:

```powershell
C:\Users\User\Documents\Playground\tools\miniforge3\Scripts\conda.exe run -n holllm python `
  C:\Users\User\Documents\Playground\Hands-On-Large-Language-Models\chapter01\chapter01_run.py `
  --model microsoft/Phi-3-mini-4k-instruct --device cpu
```

If you prefer Visual Studio terminal in `cmd`, use:

```cmd
cd /d C:\Users\User\Documents\Playground\Hands-On-Large-Language-Models\chapter01
run_chapter01.bat
```

To pass custom arguments in `cmd`:

```cmd
run_chapter01.bat --model microsoft/Phi-3-mini-4k-instruct --device cpu --prompt "Create a funny joke about chickens."
```

## 2) (Optional) Start Jupyter in the correct environment

From PowerShell:

```powershell
cd C:\Users\User\Documents\Playground\Hands-On-Large-Language-Models
.\start-chapter01.ps1
```

If PowerShell blocks scripts in your session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\start-chapter01.ps1
```

## 3) Open the notebook

Open:

- `chapter01/Chapter 1 - Introduction to Language Models.ipynb`

## 4) Adjust one line for this machine

In the model loading cell, replace:

```python
device_map="cuda",
```

with:

```python
device_map="cpu",
```

Note:

- `microsoft/Phi-3-mini-4k-instruct` is large for CPU-only use and can be slow.
- If you want a faster local test first, use `sshleifer/tiny-gpt2`.

## 5) Quick local smoke test (already validated)

This environment has:

- `torch==2.11.0+cpu`
- `transformers==4.41.2`
- `accelerate==0.31.0`
- `jupyterlab==4.5.6`
- `notebook==7.5.5`
