<#
.SYNOPSIS
    Creates or repairs the project's virtual environment (.venv) and installs dependencies.

.DESCRIPTION
    Always builds .venv from a real Python installation, never from another project's
    virtual environment, and rebuilds it automatically when it is broken (for example,
    packages compiled for a different Python version).

.PARAMETER Python
    Python version to use with the py launcher, e.g. 3.13 or 3.14. Default: newest installed.

.PARAMETER Force
    Delete and rebuild .venv even if it looks healthy.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Force
    .\setup.ps1 -Python 3.13 -Force
#>
param(
    [string]$Python = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$Venv = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Requirements = Join-Path $ProjectDir "requirements.txt"
$ImportCheck = "import numpy, pydantic_core, langchain_core, langchain_openai, gradio, pypdf, docx2txt, dotenv"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    $ErrorActionPreference = "Continue"  # pip notices on stderr are not failures; exit code decides
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)." }
}

function Invoke-Probe {
    # Run a native command only to inspect it: its errors must not stop this script
    # (Windows PowerShell 5.1 turns redirected stderr into terminating errors under "Stop").
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = "Continue"
    $output = & $Exe @Arguments 2>$null
    return @{ Ok = ($LASTEXITCODE -eq 0); Output = @($output) }
}

function Get-BasePython {
    # Resolve a base interpreter: sys.base_prefix points outside any virtual environment.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        if ($Python) { $candidates += , @("py", "-$Python") } else { $candidates += , @("py", "-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @("python") }
    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $launcherArgs = @($candidate | Select-Object -Skip 1)
        $code = "import sys, os; print(os.path.join(sys.base_prefix, 'python.exe')); print('%d.%d' % sys.version_info[:2])"
        $probe = Invoke-Probe $exe ($launcherArgs + @("-c", $code))
        $output = $probe.Output
        if ($probe.Ok -and $output.Count -ge 2 -and (Test-Path $output[0])) {
            if ($Python -and $output[1] -ne $Python) { continue }  # 'python' on PATH is another version
            return @{ Exe = $output[0]; Version = $output[1] }
        }
    }
    if ($Python) { throw "Python $Python was not found. Install it from https://www.python.org/downloads/ or omit -Python." }
    throw "No Python installation found. Install Python 3.10+ from https://www.python.org/downloads/."
}

Write-Host "== Enterprise RAG Knowledge Assistant setup ==" -ForegroundColor Cyan
if ($env:VIRTUAL_ENV -and ((Resolve-Path $env:VIRTUAL_ENV -ErrorAction SilentlyContinue).Path -ne $Venv)) {
    Write-Host "Note: another virtual environment is active ($env:VIRTUAL_ENV). It will not be used." -ForegroundColor Yellow
}

$base = Get-BasePython
$major, $minor = $base.Version.Split(".") | ForEach-Object { [int]$_ }
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    throw "Python $($base.Version) is too old; this project needs Python 3.10 or newer."
}
Write-Host "Using Python $($base.Version) at $($base.Exe)"

# Decide whether the existing .venv can be kept.
$rebuild = [bool]$Force
if (-not $rebuild -and (Test-Path $Venv)) {
    if (-not (Test-Path $VenvPython)) {
        Write-Host ".venv is incomplete; rebuilding." -ForegroundColor Yellow
        $rebuild = $true
    } else {
        $probe = Invoke-Probe $VenvPython @("-c", "import sys; print('%d.%d' % sys.version_info[:2])")
        $venvVersion = ($probe.Output -join "").Trim()
        if (-not $probe.Ok -or $venvVersion -ne $base.Version) {
            Write-Host ".venv uses Python $venvVersion, expected $($base.Version); rebuilding." -ForegroundColor Yellow
            $rebuild = $true
        } else {
            if (-not (Invoke-Probe $VenvPython @("-c", $ImportCheck)).Ok) {
                Write-Host ".venv packages are missing or broken; rebuilding." -ForegroundColor Yellow
                $rebuild = $true
            }
        }
    }
}

if ($rebuild -and (Test-Path $Venv)) {
    if ($env:VIRTUAL_ENV -and ((Resolve-Path $env:VIRTUAL_ENV -ErrorAction SilentlyContinue).Path -eq $Venv)) {
        Write-Host "Tip: run 'deactivate' after setup finishes, then activate again." -ForegroundColor Yellow
    }
    try {
        Remove-Item -Recurse -Force $Venv
    } catch {
        throw "Could not delete .venv - close any running app/terminal using it (python app.py) and try again. $_"
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating .venv ..."
    Invoke-Checked $base.Exe @("-m", "venv", $Venv) "Creating the virtual environment"
}

Write-Host "Installing dependencies (this can take a few minutes the first time) ..."
Invoke-Checked $VenvPython @("-m", "pip", "install", "--quiet", "--upgrade", "pip") "Upgrading pip"
Invoke-Checked $VenvPython @("-m", "pip", "install", "--quiet", "-r", $Requirements) "Installing requirements"
Invoke-Checked $VenvPython @("-c", $ImportCheck) "Verifying the installed packages"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start the app with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    python app.py"
