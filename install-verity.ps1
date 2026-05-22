# install-verity.ps1
# One-click Verity setup for Windows.
#
# Right-click this file and choose "Run with PowerShell".
# Or run from a PowerShell prompt:  .\install-verity.ps1
#
# Default install location is %USERPROFILE%\Verity. Override with:
#   .\install-verity.ps1 -InstallPath C:\path\you\prefer
#
# To install from a fork, override the repo URL:
#   .\install-verity.ps1 -RepoUrl https://github.com/<you>/Verity.git
# The URL must be on github.com (https) by ValidatePattern below.

param(
    [string]$InstallPath = "$env:USERPROFILE\Verity",

    # 2026-05-12: ValidatePattern guards against -RepoUrl pointing at
    # an untrusted host. `npm install` and `npm run build` run
    # arbitrary post-install / build scripts from whatever code was
    # cloned, so blindly accepting any URL would let a caller turn
    # this installer into a remote-code-execution vector.
    [ValidatePattern('^https://github\.com/[^/\s]+/[^/\s]+\.git$')]
    [string]$RepoUrl     = "https://github.com/johnnyryan/Verity.git"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Write-Step([string]$Msg) {
    Write-Host "  $Msg"
}

Write-Host ""
Write-Host "Verity setup" -ForegroundColor Cyan
Write-Host "============" -ForegroundColor Cyan
Write-Host ""

# 1. Prerequisite checks.
Write-Host "Checking prerequisites..." -ForegroundColor Cyan
$missing = @()
if (-not (Test-Command node))   { $missing += "Node.js 18+ (https://nodejs.org)" }
if (-not (Test-Command git))    { $missing += "Git (https://git-scm.com)" }
if (-not (Test-Command ollama)) { $missing += "Ollama (https://ollama.com)" }

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "  - $m" }
    Write-Host ""
    Write-Host "Install the missing tools and run this script again."
    exit 1
}
Write-Step "[OK] Node.js, Git, Ollama present"

# LM Studio is a UI app, not a CLI. Check a couple of common install paths.
$lmStudioPaths = @(
    "$env:LOCALAPPDATA\LM-Studio\LM Studio.exe",
    "$env:LOCALAPPDATA\Programs\LM Studio\LM Studio.exe",
    "$env:USERPROFILE\AppData\Local\LM-Studio\LM Studio.exe"
)
$lmStudioFound = $lmStudioPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($lmStudioFound) {
    Write-Step "[OK] LM Studio found at $lmStudioFound"
} else {
    Write-Host ""
    Write-Host "[WARN] LM Studio not found in common paths." -ForegroundColor Yellow
    Write-Host "       Install LM Studio 0.3.x or newer from https://lmstudio.ai"
    Write-Host "       Continuing; you can install LM Studio after this script."
}

# 2. Clone or update the Verity repo.
Write-Host ""
Write-Host "Fetching Verity source..." -ForegroundColor Cyan
if (Test-Path $InstallPath) {
    Write-Step "[INFO] $InstallPath already exists; pulling latest"
    Push-Location $InstallPath
    try { git pull } catch { Write-Host "  [WARN] git pull failed; continuing with existing copy" }
    Pop-Location
} else {
    Write-Step "[INFO] Cloning into $InstallPath"
    git clone $RepoUrl $InstallPath
}

# 3. Install dependencies and build.
Write-Host ""
Write-Host "Installing dependencies (npm install)..." -ForegroundColor Cyan
Push-Location "$InstallPath\project"
npm install
Write-Host ""
Write-Host "Building (npm run build)..." -ForegroundColor Cyan
npm run build
Pop-Location

# 4. Pull the two critic models via Ollama.
Write-Host ""
Write-Host "Pulling critic models (this may take a few minutes)..." -ForegroundColor Cyan
ollama pull granite3.2:8b
ollama pull granite3.2:2b

# 5. Done. Print the MCP config and next steps.
Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Open LM Studio. Settings -> Model Context Protocol. Paste:"
Write-Host ""
Write-Host '       {' -ForegroundColor Yellow
Write-Host '         "mcpServers": {' -ForegroundColor Yellow
Write-Host '           "verity": {' -ForegroundColor Yellow
Write-Host '             "url": "http://localhost:8090/mcp",' -ForegroundColor Yellow
Write-Host '             "timeout": 240000,' -ForegroundColor Yellow
Write-Host '             "retries": 1' -ForegroundColor Yellow
Write-Host '           }' -ForegroundColor Yellow
Write-Host '         }' -ForegroundColor Yellow
Write-Host '       }' -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Load a chat model in LM Studio. That is your worker."
Write-Host ""
Write-Host "  3. Start Verity. From a PowerShell prompt:"
Write-Host "       $InstallPath\start-verity.ps1"
Write-Host "     (Right-click the file and 'Run with PowerShell' also works.)"
Write-Host ""
Write-Host "  4. After any answer in LM Studio, type '/verify'."
Write-Host ""
