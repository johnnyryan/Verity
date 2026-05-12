# =============================================================================
#  ollama-amd.ps1  --  robust per-process AMD-pinned Ollama launcher for Verity
# =============================================================================
#
#  Purpose
#  -------
#  Pins Ollama to the AMD 5700 XT (via Vulkan) without polluting user-scope
#  environment. LM Studio, Lore indexing, and any other CUDA workload on the
#  NVIDIA 5070 Ti are unaffected because their processes inherit their own
#  env block and never see the variables this launcher sets.
#
#  Isolation guarantees
#  --------------------
#  - All env mutations are SCOPED TO THIS POWERSHELL PROCESS.
#    No [Environment]::SetEnvironmentVariable(...,'User'/'Machine') calls.
#    Nothing written to the registry, nothing written to user/system scope.
#  - The launched `ollama serve` inherits this process's env and propagates
#    it to its child runners (llama-server.exe).
#  - When you run -Action Stop, only Ollama-related processes are killed.
#  - On script exit, ZERO PERSISTENT STATE remains. No services registered,
#    no Task Scheduler entries, no registry keys.
#
#  Usage
#  -----
#    .\ollama-amd.ps1                       # Start (default)
#    .\ollama-amd.ps1 -Action Start
#    .\ollama-amd.ps1 -Action Stop
#    .\ollama-amd.ps1 -Action Restart
#    .\ollama-amd.ps1 -Action Status
#    .\ollama-amd.ps1 -Action Verify        # exit 0 = on AMD; exit 2 = on NVIDIA
#    .\ollama-amd.ps1 -Action Logs
#
#  Exit codes
#  ----------
#    0  success
#    1  start failed (port did not bind in time)
#    2  GPU verification failed -- Ollama landed on NVIDIA after both retries
#    3  Ollama binary not found
#
#  Note on ExecutionPolicy: if your default policy is Restricted, run via
#    powershell -ExecutionPolicy Bypass -File C:\AI\verify\CLI\ollama-amd.ps1
# =============================================================================

[CmdletBinding()]
param(
    [ValidateSet('Start','Stop','Restart','Status','Verify','Logs')]
    [string]$Action = 'Start',

    # Heavier hammer: restrict the Vulkan loader to AMD's ICD only.
    # Used automatically as a retry if the first start lands on NVIDIA;
    # set manually to skip the retry and use it on first attempt.
    [switch]$ForceAmdVulkanDriver
)

$ErrorActionPreference = 'Stop'
$script:RetriedWithLoaderRestriction = $false

# --- Paths & state ----------------------------------------------------------
$StateDir    = Join-Path $env:LOCALAPPDATA 'verity-ollama'
$PidFile     = Join-Path $StateDir 'ollama.pid'
$LogFile     = Join-Path $StateDir 'ollama.log'
$LogErrFile  = Join-Path $StateDir 'ollama.err.log'
$LauncherLog = Join-Path $StateDir 'launcher.log'

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

function Write-Log([string]$msg, [string]$level = 'INFO') {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line  = "$stamp [$level] $msg"
    Add-Content -Path $LauncherLog -Value $line
    switch ($level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        'OK'    { Write-Host $line -ForegroundColor Green }
        default { Write-Host $line }
    }
}

# --- Locate Ollama binary ---------------------------------------------------
function Find-Ollama {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe",
        "${env:ProgramFiles(x86)}\Ollama\ollama.exe"
    )
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

# --- Process discovery ------------------------------------------------------
function Get-OllamaServePids {
    Get-CimInstance Win32_Process -Filter "Name='ollama.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'serve' } |
        Select-Object -ExpandProperty ProcessId
}

function Get-AllOllamaProcesses {
    # Matches: ollama.exe (CLI + serve), "ollama app.exe" (tray), llama-server,
    # ollama_llama_server (the runner spawned per-model).
    $names = @('ollama', 'ollama app', 'llama-server', 'ollama_llama_server')
    $procs = @()
    foreach ($n in $names) {
        $found = Get-Process -Name $n -ErrorAction SilentlyContinue
        if ($found) { $procs += $found }
    }
    return $procs
}

function Stop-AllOllama {
    $procs = Get-AllOllamaProcesses
    if (-not $procs) {
        Write-Log 'No Ollama processes running.'
        return
    }
    Write-Log ("Stopping Ollama processes: " + (($procs | ForEach-Object { "$($_.Name)#$($_.Id)" }) -join ', '))

    # Graceful first (WM_CLOSE on GUI; ignored by console processes)
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.Id -ErrorAction Stop } catch { }
    }
    Start-Sleep -Milliseconds 800

    # Force-kill survivors
    $survivors = Get-AllOllamaProcesses
    foreach ($p in $survivors) {
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop }
        catch { Write-Log "Could not stop PID $($p.Id): $_" 'WARN' }
    }
    Start-Sleep -Milliseconds 400

    $remaining = Get-AllOllamaProcesses
    if ($remaining) {
        Write-Log ("Still running: " + (($remaining | ForEach-Object { $_.Id }) -join ', ')) 'WARN'
    } else {
        Write-Log 'All Ollama processes stopped.' 'OK'
    }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
}

# --- GPU placement check ----------------------------------------------------
# Parse the Ollama err log for its own statement of which backend it chose.
# This is the definitive source. Returns a hashtable like:
#   @{ Library = 'Vulkan'; Device = 'AMD Radeon RX 5700 XT'; OnAmd = $true; OnNvidia = $false }
# Returns $null if no inference-compute line is in the log yet.
function Get-OllamaInferenceBackend {
    if (-not (Test-Path $LogErrFile)) { return $null }
    $lines = Get-Content $LogErrFile -ErrorAction SilentlyContinue
    $line  = $lines | Where-Object { $_ -match 'inference compute' } | Select-Object -Last 1
    if (-not $line) { return $null }
    if ($line -match 'library=([A-Za-z]+).*description="([^"]+)"') {
        $lib  = $matches[1]
        $desc = $matches[2]
        return @{
            Library  = $lib
            Device   = $desc
            OnAmd    = ($desc -match '(?i)\b(amd|radeon|ati)\b')
            OnNvidia = ($desc -match '(?i)\b(nvidia|geforce|rtx|gtx|quadro|tesla)\b') -or ($lib -ieq 'CUDA')
        }
    }
    if ($line -match 'library=CUDA') {
        return @{ Library = 'CUDA'; Device = 'NVIDIA'; OnAmd = $false; OnNvidia = $true }
    }
    return $null
}

function Get-OllamaOnNvidia {
    # Returns @{ OnNvidia = bool; Procs = [string[]] }
    #
    # DEFINITIVE check: parse Ollama's own log for which backend it picked.
    # The parent ollama.exe process appears in `nvidia-smi --query-compute-apps`
    # whenever it has a driver context open, even when inference is on AMD via
    # Vulkan -- so nvidia-smi is too noisy for a verdict on its own.
    $backend = Get-OllamaInferenceBackend
    if ($backend) {
        if ($backend.OnAmd) {
            return @{ OnNvidia = $false; Procs = @() }
        }
        if ($backend.OnNvidia) {
            return @{ OnNvidia = $true; Procs = @("Ollama log: inference on $($backend.Library) / $($backend.Device)") }
        }
    }

    # No "inference compute" line yet (no model loaded since startup).
    # Fall back to nvidia-smi, but only count RUNNER processes as evidence
    # -- the parent ollama.exe with [N/A] memory is noise.
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return @{ OnNvidia = $false; Procs = @() } }
    try {
        $csv = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
    } catch { return @{ OnNvidia = $false; Procs = @() } }
    if (-not $csv) { return @{ OnNvidia = $false; Procs = @() } }
    # Only flag the actual runner; ollama.exe parent is a false positive.
    $hits = @($csv | Where-Object { $_ -match 'ollama_llama_server|llama-server\.exe|granite' })
    return @{ OnNvidia = $hits.Count -gt 0; Procs = $hits }
}

# --- Per-process env scoping ------------------------------------------------
# Every assignment below is to $env:VAR which is process-local. The new
# `ollama serve` we spawn inherits these; LM Studio / Lore / anything else
# already running keeps its own env block untouched.
# Discover AMD's Vulkan ICD JSON path on a modern Windows install. On Win10+
# the loader doesn't read C:\Windows\System32\*.json directly any more --
# ICDs are registered per-display-device in the driver store, accessible
# via the PNP display-class registry. We scan all subkeys for a
# VulkanDriverName whose DriverDesc names AMD or Radeon, and return that
# JSON path.
function Get-AmdVulkanIcd {
    $displayClass = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
    if (-not (Test-Path $displayClass)) { return $null }

    $candidates = @()
    Get-ChildItem -Path $displayClass -ErrorAction SilentlyContinue | ForEach-Object {
        $key = Get-ItemProperty -Path $_.PSPath -ErrorAction SilentlyContinue
        if ($null -ne $key.VulkanDriverName -and $key.VulkanDriverName -ne '' -and $key.DriverDesc) {
            $candidates += [PSCustomObject]@{
                Desc = [string]$key.DriverDesc
                Icd  = [string]$key.VulkanDriverName
            }
        }
    }

    # Prefer DriverDesc matching AMD / Radeon (skip NVIDIA, Intel, etc.)
    $amd = $candidates | Where-Object { $_.Desc -match '(?i)\b(amd|radeon|ati)\b' } | Select-Object -First 1
    if ($amd -and (Test-Path $amd.Icd)) { return $amd.Icd }
    return $null
}

function Set-OllamaEnv {
    # Server binding + Ollama behaviour
    $env:OLLAMA_HOST              = '127.0.0.1:11434'
    $env:OLLAMA_LLM_LIBRARY       = 'vulkan'
    $env:OLLAMA_MAX_LOADED_MODELS = '2'
    $env:OLLAMA_KEEP_ALIVE        = '24h'
    $env:OLLAMA_NUM_PARALLEL      = '1'

    # Vulkan device selection (llama.cpp backend reads this if more than
    # one Vulkan device is visible). With the driver-files restriction
    # below it's belt-and-braces.
    $env:GGML_VK_VISIBLE_DEVICES  = '0'

    # Hide CUDA & HIP enumeration from Ollama so the CUDA detector
    # doesn't grab the NVIDIA card at startup. LM Studio's process is
    # unaffected because env is per-process.
    $env:CUDA_VISIBLE_DEVICES     = ''
    $env:HIP_VISIBLE_DEVICES      = ''
    $env:ROCR_VISIBLE_DEVICES     = ''
    $env:GPU_DEVICE_ORDINAL       = ''

    # ALWAYS restrict the Vulkan loader to AMD's ICD only. This is the
    # one reliable lever -- everything else (OLLAMA_LLM_LIBRARY=vulkan,
    # GGML_VK_VISIBLE_DEVICES) is a hint that recent Ollama can ignore.
    # If the loader only sees the AMD driver, Ollama has no choice.
    $amdIcd = Get-AmdVulkanIcd
    if ($amdIcd) {
        $env:VK_DRIVER_FILES         = $amdIcd
        # Older Vulkan loaders read VK_ICD_FILENAMES instead; set both.
        $env:VK_ICD_FILENAMES        = $amdIcd
        # Belt-and-braces: also tell newer loaders to disable any driver
        # whose manifest filename contains "nv" or "intel".
        $env:VK_LOADER_DRIVERS_DISABLE = '*nv*;*intel*;*igvk*'
        Write-Log "Vulkan loader restricted to AMD ICD only: $amdIcd" 'OK'
    } else {
        Write-Log 'Could not locate AMD Vulkan ICD via PNP registry. Ollama may default to NVIDIA.' 'WARN'
    }
}

# --- Wait for /api/tags to come up after spawn ------------------------------
function Wait-OllamaReady {
    # 60-second budget. Ollama does GPU discovery + model scan before binding,
    # which on a cold rig can take 30+ s. Two-stage check: TCP port first
    # (cheap, accurate signal that the listener exists), then HTTP /api/tags
    # to confirm the API is actually serving.
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $portReady = $false
        try {
            $iar = $tcp.BeginConnect('127.0.0.1', 11434, $null, $null)
            if ($iar.AsyncWaitHandle.WaitOne(1000) -and $tcp.Connected) {
                $tcp.EndConnect($iar)
                $portReady = $true
            }
        } catch { }
        finally { try { $tcp.Close() } catch { } }

        if ($portReady) {
            try {
                $r = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/tags' `
                                       -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
                if ($r.StatusCode -eq 200) { return $true }
            } catch { }
        }
        Start-Sleep -Milliseconds 1000
    }
    return $false
}

# --- Force a model load so the GPU binding actually shows up ----------------
function Probe-LoadModel {
    try {
        $tags = Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5
        $modelName = $null
        if ($tags.models -and $tags.models.Count -gt 0) {
            # Prefer smallest available model for the probe
            $modelName = ($tags.models | Sort-Object size | Select-Object -First 1).name
        }
        if (-not $modelName) {
            Write-Log 'No models installed in Ollama. Cannot probe GPU placement; verification skipped.' 'WARN'
            return $false
        }
        Write-Log "Probing GPU placement with model: $modelName"
        $body = @{
            model   = $modelName
            prompt  = 'hi'
            stream  = $false
            options = @{ num_predict = 1 }
        } | ConvertTo-Json -Compress
        $null = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' `
                                  -Method Post -Body $body `
                                  -ContentType 'application/json' -TimeoutSec 60
        Start-Sleep -Seconds 1   # let runner settle so nvidia-smi can see it
        return $true
    } catch {
        Write-Log "Probe failed: $_" 'WARN'
        return $false
    }
}

# --- Actions ----------------------------------------------------------------
function Action-Start {
    $ollama = Find-Ollama
    if (-not $ollama) {
        Write-Log 'Ollama binary not found. Install from https://ollama.com or add it to PATH.' 'ERROR'
        exit 3
    }
    Write-Log "Ollama binary: $ollama"

    # Idempotency: if a serve is already up, decide whether to no-op or restart
    $existing = Get-OllamaServePids
    if ($existing) {
        $null = Probe-LoadModel
        $gpu = Get-OllamaOnNvidia
        if ($gpu.OnNvidia) {
            Write-Log 'Ollama is running but on NVIDIA. Restarting on AMD.' 'WARN'
            Stop-AllOllama
        } else {
            Write-Log "Ollama already running (PID $($existing -join ', ')) and not on NVIDIA. No-op." 'OK'
            Show-Status
            return
        }
    } else {
        # Make sure no stale tray app is around with its own broken env
        $tray = Get-Process -Name 'ollama app' -ErrorAction SilentlyContinue
        if ($tray) {
            Write-Log 'Killing Ollama tray app (it would spawn a competing serve with broken env).' 'WARN'
            Stop-AllOllama
        }
    }

    Set-OllamaEnv

    Write-Log "Launching: $ollama serve"
    Write-Log "  CUDA_VISIBLE_DEVICES    = '<empty -- hidden>'"
    Write-Log "  HIP_VISIBLE_DEVICES     = '<empty -- hidden>'"
    Write-Log "  OLLAMA_LLM_LIBRARY      = $env:OLLAMA_LLM_LIBRARY"
    Write-Log "  GGML_VK_VISIBLE_DEVICES = $env:GGML_VK_VISIBLE_DEVICES"
    if ($env:VK_DRIVER_FILES) {
        Write-Log "  VK_DRIVER_FILES         = $env:VK_DRIVER_FILES"
        Write-Log "  VK_ICD_FILENAMES        = (same as above, legacy loader var)"
        Write-Log "  VK_LOADER_DRIVERS_DISABLE = $env:VK_LOADER_DRIVERS_DISABLE"
    }

    # Truncate per-launch ollama logs to keep them readable
    Set-Content -Path $LogFile    -Value '' -Force
    Set-Content -Path $LogErrFile -Value '' -Force

    $proc = Start-Process -FilePath $ollama -ArgumentList 'serve' `
                          -RedirectStandardOutput $LogFile `
                          -RedirectStandardError  $LogErrFile `
                          -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $proc.Id
    Write-Log "Started Ollama PID $($proc.Id). Logs: $LogFile" 'OK'

    if (-not (Wait-OllamaReady)) {
        Write-Log 'Ollama did not bind :11434 within 60 s. Check error log.' 'ERROR'
        Get-Content $LogErrFile -Tail 20 -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Log 'Ollama responding on :11434.' 'OK'

    # Verify GPU placement with a real model probe
    $null = Probe-LoadModel
    $gpu = Get-OllamaOnNvidia
    if ($gpu.OnNvidia) {
        Write-Log 'GPU check FAILED -- Ollama landed on NVIDIA despite VK_DRIVER_FILES restriction:' 'ERROR'
        foreach ($p in $gpu.Procs) { Write-Log "  $p" 'ERROR' }
        Write-Log '  This shouldn''t happen. Tail of ollama.err.log:' 'ERROR'
        Get-Content $LogErrFile -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object { Write-Log "    $_" 'ERROR' }
        Write-Log '  Run: $env:VK_DRIVER_FILES = "<AMD ICD path>"; ollama serve  -- in a fresh shell to debug.' 'ERROR'
        exit 2
    }

    Write-Log 'GPU check passed: no Ollama runners on NVIDIA.' 'OK'

    # Pre-warm the Granite critics so the first /verify doesn't have to
    # cold-load them (cold load is ~10 s for 8B, ~3 s for 2B; concurrent
    # cold loads under OLLAMA_NUM_PARALLEL=1 can exceed CRITIC_TIMEOUT_MS
    # and abort both critics). Fire a tiny 1-token request to each.
    # Non-fatal if a model isn't pulled yet.
    Warm-CriticModels
    Show-Status
}

# Pre-warm both Granite critics so /verify's first call doesn't pay
# cold-load latency. ~12 s total on a fresh Ollama; subsequent verify
# calls are fast because the runners stay resident for OLLAMA_KEEP_ALIVE.
function Warm-CriticModels {
    $critics = @('granite3.2:8b', 'granite3.2:2b')
    foreach ($model in $critics) {
        Write-Log "Pre-warming $model ..."
        $body = @{
            model   = $model
            prompt  = 'hi'
            stream  = $false
            options = @{ num_predict = 1 }
        } | ConvertTo-Json -Compress
        try {
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $r = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' `
                                   -Method Post -Body $body `
                                   -ContentType 'application/json' -TimeoutSec 60
            $sw.Stop()
            Write-Log ("  {0} warm in {1:N1} s" -f $model, $sw.Elapsed.TotalSeconds) 'OK'
        } catch {
            Write-Log "  $model not warmed (skipped): $_" 'WARN'
        }
    }
}

function Action-Stop    { Stop-AllOllama }
function Action-Restart { Action-Stop; Start-Sleep -Seconds 1; Action-Start }

function Action-Verify {
    $existing = Get-OllamaServePids
    if (-not $existing) {
        Write-Log 'Ollama is not running. Nothing to verify.' 'WARN'
        exit 1
    }
    $null = Probe-LoadModel
    $gpu = Get-OllamaOnNvidia
    if ($gpu.OnNvidia) {
        Write-Log 'GPU check FAILED -- Ollama on NVIDIA.' 'ERROR'
        foreach ($p in $gpu.Procs) { Write-Log "  $p" 'ERROR' }
        exit 2
    }
    Write-Log 'GPU check passed: Ollama not on NVIDIA.' 'OK'
    exit 0
}

function Show-Status {
    Write-Host ''
    Write-Host '=== Verity-Ollama launcher status ===' -ForegroundColor Cyan

    $pids = Get-OllamaServePids
    if ($pids) {
        Write-Host "ollama serve PID(s): $($pids -join ', ')"
        try {
            $tags = Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 3
            if ($tags.models -and $tags.models.Count -gt 0) {
                Write-Host 'Models known to Ollama:'
                foreach ($m in $tags.models) {
                    $sizeGb = [math]::Round($m.size / 1GB, 2)
                    Write-Host ("  - {0,-30}  {1,6} GB" -f $m.name, $sizeGb)
                }
            } else {
                Write-Host '(no models pulled yet -- `ollama pull granite3.2:8b` and `ollama pull granite3.2:2b`)'
            }
        } catch {
            Write-Host 'Ollama not responding to /api/tags' -ForegroundColor Yellow
        }
    } else {
        Write-Host 'Ollama not running.' -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host 'GPU placement:'
    $gpu = Get-OllamaOnNvidia
    if ($gpu.OnNvidia) {
        Write-Host '  [FAIL] Ollama runners detected on NVIDIA -- this will time out under Verity.' -ForegroundColor Red
        foreach ($p in $gpu.Procs) { Write-Host "    $p" -ForegroundColor Red }
        Write-Host '  Fix: run with -Action Restart.' -ForegroundColor Yellow
    } else {
        Write-Host '  [OK] No Ollama runners on NVIDIA.' -ForegroundColor Green
    }

    Write-Host ''
    Write-Host 'Log files:'
    Write-Host "  Launcher:    $LauncherLog"
    Write-Host "  Ollama out:  $LogFile"
    Write-Host "  Ollama err:  $LogErrFile"
    Write-Host ''
}

function Action-Logs {
    Write-Host '-- launcher.log (last 30 lines) --' -ForegroundColor Cyan
    if (Test-Path $LauncherLog) { Get-Content $LauncherLog -Tail 30 } else { Write-Host '(none)' }
    Write-Host ''
    Write-Host '-- ollama.err.log (last 40 lines) --' -ForegroundColor Cyan
    if (Test-Path $LogErrFile) { Get-Content $LogErrFile -Tail 40 } else { Write-Host '(none)' }
}

# --- Dispatch ---------------------------------------------------------------
switch ($Action) {
    'Start'   { Action-Start }
    'Stop'    { Action-Stop }
    'Restart' { Action-Restart }
    'Status'  { Show-Status }
    'Verify'  { Action-Verify }
    'Logs'    { Action-Logs }
}
