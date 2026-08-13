#Requires -Version 5.1
<#
.SYNOPSIS
    Runs the same gates as .github/workflows/checks.yml.
#>

$ErrorActionPreference = "Stop"

# $ErrorActionPreference does not apply to native executables in Windows
# PowerShell 5.1: a failing `python -m pytest` sets $LASTEXITCODE but does not
# stop the script. Without the explicit check below this helper reported
# success while the gates were red.
function Invoke-Check {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host "== $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "== $Name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Check "compileall"  { python -m compileall sidewalk_ai -q }
Invoke-Check "ruff"        { python -m ruff check sidewalk_ai }
Invoke-Check "black"       { python -m black --check sidewalk_ai }
Invoke-Check "pytest"      { python -m pytest }
Invoke-Check "diagnostics" { python -m sidewalk_ai.diagnostics }

Write-Host "All checks passed" -ForegroundColor Green
