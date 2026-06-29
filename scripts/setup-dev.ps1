param(
    [string]$VenvPath = ".venv",
    [switch]$WithApi,
    [switch]$WithMl
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

$python = Join-Path $VenvPath "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e ".[dev]"

if ($WithApi) {
    & $python -m pip install -e ".[api]"
}

if ($WithMl) {
    Write-Host "Installing generic ML extras. Install PyTorch/CUDA/backend-specific wheels separately when needed."
    & $python -m pip install -e ".[ml]"
}

& $python -m sidewalk_ai.diagnostics
