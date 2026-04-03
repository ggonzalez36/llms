$ErrorActionPreference = "Stop"

$condaExe = "C:\Users\User\Documents\Playground\tools\miniforge3\Scripts\conda.exe"
$chapterDir = Join-Path $PSScriptRoot "chapter01"

if (-not (Test-Path -LiteralPath $condaExe)) {
    throw "No se encontro conda en: $condaExe"
}

if (-not (Test-Path -LiteralPath $chapterDir)) {
    throw "No se encontro la carpeta chapter01 en: $chapterDir"
}

# Launch Jupyter Lab in the chapter01 folder using the holllm environment.
& $condaExe run -n holllm jupyter lab --notebook-dir "$chapterDir" @args
exit $LASTEXITCODE
