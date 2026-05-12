# =============================================================================
#  start-verity.ps1  --  one-click bring-up for Verity
# =============================================================================
#
#  Boots the whole Verity stack:
#    1. Ensures Ollama is running on the AMD GPU (delegates to ollama-amd.ps1)
#    2. Ensures the Verity MCP server is running on :8090
#    3. Health-checks both and reports status
#
#  All state is process-local. Nothing is written to user/system env,
#  nothing registered as a service, no Task Scheduler entries.
#
#  Usage
#  -----
#    .\start-verity.ps1                       # Start everything (default)
#    .\start-verity.ps1 -Action Start
#    .\start-verity.ps1 -Action Stop          # Stop Verity then Ollama
#    .\start-verity.ps1 -Action Restart
#    .\start-verity.ps1 -Action Status
#    .\start-verity.ps1 -InstallShortcut      # Drop a desktop icon (one-time)
#
#  Exit codes
#  ----------
#    0  success
#    1  Verity MCP server did not become healthy
#    2  GPU verification failed (Ollama landed on NVIDIA after retries)
#    3  Ollama binary not found
#    4  node.exe not on PATH
#    5  Verity build not found -- run `npm run build` first
# =============================================================================

[CmdletBinding()]
param(
    [ValidateSet('Start','Stop','Restart','Status')]
    [string]$Action = 'Start',
    [switch]$InstallShortcut,
    [switch]$InstallStopShortcut
)

$ErrorActionPreference = 'Stop'

# ---─ Paths ---------------------------------------------------------------------------------------------------─
$Root              = Split-Path -Parent $PSCommandPath
$OllamaLauncher    = Join-Path $Root 'CLI\ollama-amd.ps1'
$VerityDist        = Join-Path $Root 'dist\index.js'
$StateDir          = Join-Path $env:LOCALAPPDATA 'verity'
$VerityPidFile     = Join-Path $StateDir 'verity.pid'
$VerityLog         = Join-Path $StateDir 'verity.log'
$VerityErrLog      = Join-Path $StateDir 'verity.err.log'
$LauncherLog       = Join-Path $StateDir 'launcher.log'

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

# ---─ Logging ------------------------------------------------------------------------------------------------─
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

# ---─ Process discovery ---------------------------------------------------------------------------------
function Find-Node {
    $cmd = Get-Command node -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:ProgramFiles\nodejs\node.exe",
        "${env:ProgramFiles(x86)}\nodejs\node.exe"
    )
    foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}

function Get-VerityPid {
    # The real-world CommandLine for the launched Verity can be any of:
    #   "C:\Program Files\nodejs\node.exe" "C:\AI\verify\dist\index.js"   <- absolute
    #   "C:\Program Files\nodejs\node.exe" dist/index.js                  <- relative (Start-Process with -WorkingDirectory)
    # The old regex required "verify\dist\index.js" anywhere in CommandLine
    # and missed the relative form. Bug: launcher then thought no Verity
    # was running and spawned a new one which died on EADDRINUSE while the
    # stale process kept serving. New approach: match any node.exe whose
    # CommandLine references dist/index.js or dist\\index.js AND whose
    # owner is listening on the Verity port (the only definitive signal
    # that this is the Verity server we care about).
    $serverPort = 8090
    try {
        $conn = Get-NetTCPConnection -LocalPort $serverPort -State Listen -ErrorAction Stop
    } catch { return $null }
    foreach ($c in $conn) {
        $procId = $c.OwningProcess
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        if ($proc.Name -ne 'node.exe') { continue }
        if ($proc.CommandLine -match 'dist[\\/]+index\.js') { return $procId }
    }
    return $null
}

function Test-VerityHealth {
    try {
        $r = Invoke-RestMethod -Uri 'http://localhost:8090/health' -TimeoutSec 3 -ErrorAction Stop
        return ($r.status -eq 'ok')
    } catch { return $false }
}

# --- Transactional Qwen-config management -----------------------------------
# Verity needs Qwen loaded with a tighter context window + q8_0 KV cache
# to avoid the VRAM-pressure crash documented in 2026-05-11. But other
# users (Lore indexing) want Qwen at its native long-context config.
#
# Rule: on Verity Start we SNAPSHOT whatever Qwen's per-model load
# defaults currently are, then write Verity's preferred values. On Verity
# Stop we RESTORE the snapshot and delete the backup. Lore (or anyone
# else) gets back exactly the config it had before Verity loaded -- no
# collateral damage.
#
# Snapshot files are kept alongside the live config files with a
# `.verity-backup` suffix. If a snapshot already exists when Start runs,
# we don't overwrite it -- the existing backup IS the original pre-Verity
# state. Idempotent re-Start is safe.

$QwenConfigDir = Join-Path $env:USERPROFILE '.lmstudio\.internal\user-concrete-model-default-config\qwen\qwen3.5-9b@lmstudio-community\Qwen3.5-9B-GGUF'
$QwenConfigFiles = @(
    'Qwen3.5-9B-Q8_0.gguf.json',
    'Qwen3.5-9B-Q4_K_M.gguf.json'
)

# What Verity wants while it's loaded: 64k context (matches the design doc
# target of "up to 64k"), q8_0 KV cache (halves cache footprint vs f16),
# model stays resident. If Lore needs 128k, this is still less than half
# the previous 256k default but enough for the typical Verity workload.
$VerityPreferredQwenConfig = @'
{
  "preset": "",
  "operation": {
    "fields": []
  },
  "load": {
    "fields": [
      {
        "key": "llm.load.contextLength",
        "value": 65536
      },
      {
        "key": "llm.load.llama.keepModelInMemory",
        "value": true
      },
      {
        "key": "llm.load.llama.kCacheQuantizationType",
        "value": {
          "checked": true,
          "value": "q8_0"
        }
      },
      {
        "key": "llm.load.llama.vCacheQuantizationType",
        "value": {
          "checked": true,
          "value": "q8_0"
        }
      }
    ]
  }
}
'@

function Apply-VerityQwenConfig {
    if (-not (Test-Path $QwenConfigDir)) {
        Write-Log "Qwen config dir not found: $QwenConfigDir -- skipping config snapshot." 'WARN'
        return
    }
    foreach ($fname in $QwenConfigFiles) {
        $live   = Join-Path $QwenConfigDir $fname
        $backup = "$live.verity-backup"
        if (-not (Test-Path $live)) {
            Write-Log "Qwen config file missing: $live -- skipping." 'WARN'
            continue
        }
        if (Test-Path $backup) {
            Write-Log "Backup already present at $backup -- keeping it (will not re-snapshot)."
        } else {
            Copy-Item -Path $live -Destination $backup -Force
            Write-Log "Snapshotted pre-Verity Qwen config: $fname -> .verity-backup"
        }
        # UTF-8 with no BOM, as LM Studio's JSON parser expects.
        [System.IO.File]::WriteAllText($live, $VerityPreferredQwenConfig)
        Write-Log "Applied Verity Qwen config (64k ctx, q8_0 KV): $fname" 'OK'
    }
    Write-Log "Note: in-memory Qwen instance keeps the OLD config until next reload."
    Write-Log "  Eject Qwen from LM Studio (or let it idle-evict) to pick up Verity's settings."
}

function Get-VerityBackupExists {
    # True if at least one .verity-backup file is present (i.e. Verity
    # currently considers itself "loaded" w.r.t. Qwen config).
    if (-not (Test-Path $QwenConfigDir)) { return $false }
    foreach ($fname in $QwenConfigFiles) {
        if (Test-Path (Join-Path $QwenConfigDir "$fname.verity-backup")) {
            return $true
        }
    }
    return $false
}

function Revert-VerityQwenConfig {
    if (-not (Test-Path $QwenConfigDir)) {
        Write-Log "Qwen config dir not found: $QwenConfigDir -- nothing to revert." 'INFO'
        return
    }
    $any = $false
    foreach ($fname in $QwenConfigFiles) {
        $live   = Join-Path $QwenConfigDir $fname
        $backup = "$live.verity-backup"
        if (-not (Test-Path $backup)) {
            continue
        }
        Copy-Item -Path $backup -Destination $live -Force
        Remove-Item -Path $backup -Force
        Write-Log "Restored pre-Verity Qwen config: $fname" 'OK'
        $any = $true
    }
    if (-not $any) {
        Write-Log "No Verity backup files present -- Qwen config was not modified by Verity. Nothing to revert."
    } else {
        Write-Log "Note: in-memory Qwen instance keeps Verity's config until next reload."
        Write-Log "  Eject Qwen from LM Studio (or let it idle-evict) for Lore-compatible settings."
    }
}

# ---─ Verity MCP server ---------------------------------------------------------------------------------
function Start-Verity {
    $existing = Get-VerityPid
    if ($existing -and (Test-VerityHealth)) {
        Write-Log "Verity already running (PID $existing) and healthy." 'OK'
        return
    }
    if ($existing) {
        Write-Log "Stale Verity PID $existing not responding on /health. Killing." 'WARN'
        try { Stop-Process -Id $existing -Force -ErrorAction Stop } catch { }
        Start-Sleep -Seconds 1
    }

    $node = Find-Node
    if (-not $node) {
        Write-Log 'node.exe not found on PATH. Install Node.js 18+ first.' 'ERROR'
        exit 4
    }
    if (-not (Test-Path $VerityDist)) {
        Write-Log "Verity build not found at $VerityDist. Run: npm run build (from $Root)" 'ERROR'
        exit 5
    }

    Write-Log "Starting Verity MCP server: $node $VerityDist"
    Set-Content -Path $VerityLog    -Value '' -Force
    Set-Content -Path $VerityErrLog -Value '' -Force

    $proc = Start-Process -FilePath $node `
                          -ArgumentList @($VerityDist) `
                          -WorkingDirectory $Root `
                          -RedirectStandardOutput $VerityLog `
                          -RedirectStandardError  $VerityErrLog `
                          -WindowStyle Hidden -PassThru
    Set-Content -Path $VerityPidFile -Value $proc.Id
    Write-Log "Verity launched (PID $($proc.Id)). Waiting for /health…"

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        if (Test-VerityHealth) {
            Write-Log 'Verity is healthy on :8090.' 'OK'
            return
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Log 'Verity did not become healthy within 30 s. Tail of err log:' 'ERROR'
    if (Test-Path $VerityErrLog) { Get-Content $VerityErrLog -Tail 20 }
    exit 1
}

function Stop-Verity {
    # NB: $pid is a PowerShell automatic variable (current process ID) and
    # is read-only. Use $vPid for the Verity server's PID throughout.
    $vPid = Get-VerityPid
    if (-not $vPid) {
        Write-Log 'Verity MCP server: not running.'
        return
    }
    # Capture RAM footprint before killing so we can report what was freed.
    $ramMb = 0
    try {
        $proc = Get-Process -Id $vPid -ErrorAction Stop
        $ramMb = [math]::Round($proc.WorkingSet64 / 1MB)
    } catch { }
    Write-Log "Stopping Verity MCP server (PID $vPid, ~$ramMb MB CPU RAM, includes DeBERTa NLI model + tiktoken)."
    try { Stop-Process -Id $vPid -ErrorAction Stop } catch { }
    Start-Sleep -Milliseconds 400
    if (Get-VerityPid) {
        try { Stop-Process -Id $vPid -Force -ErrorAction Stop } catch { }
    }
    if (Test-Path $VerityPidFile) { Remove-Item $VerityPidFile -Force -ErrorAction SilentlyContinue }
    Write-Log "Verity MCP server stopped. ~$ramMb MB CPU RAM released." 'OK'
}

# ---─ Composite status ---------------------------------------------------------------------------------─
function Show-Status {
    Write-Host ''
    Write-Host '======== Verity stack status ========' -ForegroundColor Cyan
    Write-Host ''

    # Ollama
    Write-Host '--- Ollama (AMD GPU) ---'
    if (Test-Path $OllamaLauncher) {
        & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $OllamaLauncher -Action Status
    } else {
        Write-Host "  Launcher missing: $OllamaLauncher" -ForegroundColor Red
    }

    # Verity
    Write-Host '--- Verity MCP server ---'
    $vPid = Get-VerityPid
    if ($vPid) {
        Write-Host "  PID: $vPid"
        if (Test-VerityHealth) {
            Write-Host '  [OK] /health responding on http://localhost:8090' -ForegroundColor Green
        } else {
            Write-Host '  [FAIL] /health not responding' -ForegroundColor Red
        }
    } else {
        Write-Host '  Not running.' -ForegroundColor Yellow
    }
    Write-Host ''
}

# ---─ Desktop shortcut installer ------------------------------------------------------------------─
function Install-DesktopShortcut {
    $desktop = [Environment]::GetFolderPath('Desktop')
    if (-not $desktop) {
        Write-Log 'Could not resolve Desktop folder.' 'ERROR'
        exit 1
    }
    $shortcutPath = Join-Path $desktop 'Verity.lnk'

    $wsh = New-Object -ComObject WScript.Shell
    $shortcut = $wsh.CreateShortcut($shortcutPath)
    $shortcut.TargetPath        = 'powershell.exe'
    $shortcut.Arguments         = "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    $shortcut.WorkingDirectory  = $Root
    $shortcut.IconLocation      = 'powershell.exe,0'
    $shortcut.Description       = 'Start Verity: pins Ollama to AMD then starts the MCP server on :8090. Double-click to bring the stack up.'
    $shortcut.WindowStyle       = 1   # Normal window
    $shortcut.Save()

    Write-Log "Desktop shortcut created: $shortcutPath" 'OK'
    Write-Host ''
    Write-Host 'Double-click the new "Verity" icon on your desktop to bring the stack up.'
    Write-Host 'The PowerShell window will stay open so you can see the status; close it when ready.'
    Write-Host ''
}

function Install-StopDesktopShortcut {
    $desktop = [Environment]::GetFolderPath('Desktop')
    if (-not $desktop) {
        Write-Log 'Could not resolve Desktop folder.' 'ERROR'
        exit 1
    }
    $shortcutPath = Join-Path $desktop 'Verity Stop.lnk'

    $wsh = New-Object -ComObject WScript.Shell
    $shortcut = $wsh.CreateShortcut($shortcutPath)
    $shortcut.TargetPath        = 'powershell.exe'
    $shortcut.Arguments         = "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Action Stop"
    $shortcut.WorkingDirectory  = $Root
    # shell32.dll,131 is the classic red 'stop / error' icon, present on
    # every Windows since XP. User can right-click > Properties > Change
    # Icon to swap if it doesn't render the way they want.
    $shortcut.IconLocation      = 'shell32.dll,131'
    $shortcut.Description       = 'Stop Verity: unloads all Granite models from AMD VRAM, DeBERTa NLI from CPU RAM, and kills the MCP server. LM Studio + CUDA workloads on NVIDIA are unaffected.'
    $shortcut.WindowStyle       = 1   # Normal window
    $shortcut.Save()

    Write-Log "Desktop shortcut created: $shortcutPath" 'OK'
    Write-Host ''
    Write-Host 'Double-click the new "Verity Stop" icon on your desktop to bring the stack down.'
    Write-Host 'The PowerShell window will stay open so you can see what was released.'
    Write-Host ''
}

# Post-stop verification: confirm nothing Verity-related anywhere
function Confirm-EverythingOff {
    Write-Host ''
    Write-Host 'Verifying no Verity processes remain...'
    $remaining = @()

    # 1. CPU side: any node.exe still serving Verity?
    if (Get-VerityPid) {
        $remaining += 'node.exe (Verity MCP server) still running'
    }

    # 2. AMD side: any ollama / llama-server / granite runner still alive?
    $ollamaProcs = Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('ollama','ollama app','llama-server','ollama_llama_server') }
    if ($ollamaProcs) {
        foreach ($p in $ollamaProcs) {
            $remaining += "$($p.Name).exe (PID $($p.Id)) still resident on AMD GPU"
        }
    }

    # 3. NVIDIA side: nvidia-smi compute-app probe. Should be empty for
    #    anything Verity-related (worker on NVIDIA is LM Studio's, not ours).
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $csv = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
            $nvHits = @($csv | Where-Object { $_ -match 'ollama|llama-server|llama_server|granite' })
            if ($nvHits) {
                foreach ($h in $nvHits) {
                    $remaining += "NVIDIA: $h  -- Ollama leaked onto NVIDIA (should not happen)"
                }
            }
        } catch { }
    }

    if ($remaining.Count -eq 0) {
        Write-Host ''
        Write-Host '  [OK] Verity MCP server stopped  -- DeBERTa NLI + tiktoken released from CPU RAM.' -ForegroundColor Green
        Write-Host '  [OK] Ollama serve stopped       -- Granite 3.2 8B + 2B released from AMD VRAM.' -ForegroundColor Green
        Write-Host '  [OK] No Verity-related processes on NVIDIA (LM Studio + Lore unaffected).' -ForegroundColor Green
        Write-Host ''
        Write-Host '  Everything Verity loaded has been released. Nothing on either GPU or CPU remains.' -ForegroundColor Green
    } else {
        Write-Host ''
        Write-Host '  [PARTIAL] Some Verity-related processes are still alive:' -ForegroundColor Yellow
        foreach ($r in $remaining) { Write-Host "    - $r" -ForegroundColor Yellow }
        Write-Host '  Retry: .\start-verity.ps1 -Action Stop' -ForegroundColor Yellow
    }
    Write-Host ''
}

# ---─ Dispatch ---------------------------------------------------------------------------------------------─
if ($InstallShortcut) {
    Install-DesktopShortcut
    exit 0
}
if ($InstallStopShortcut) {
    Install-StopDesktopShortcut
    exit 0
}

switch ($Action) {
    'Start' {
        Write-Log '--- Verity stack: Start ---'
        if (-not (Test-Path $OllamaLauncher)) {
            Write-Log "Ollama launcher missing: $OllamaLauncher" 'ERROR'
            exit 3
        }
        & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $OllamaLauncher -Action Start
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Ollama launcher exited with code $LASTEXITCODE -- aborting Verity start." 'ERROR'
            exit $LASTEXITCODE
        }
        # DISABLED 2026-05-11: Apply-VerityQwenConfig wrote a load config
        # with empty `operation.fields`, which made LM Studio's prediction
        # path fail schema validation on `llm.prediction.maxPredictedTokens`
        # (must be >= 1) on the next chat completion against Qwen. Until
        # we can derive a sensible operation.fields default, leave the
        # JSON config untouched. Apply-VerityQwenConfig/Revert-VerityQwenConfig
        # are still defined below for future re-enable.
        # Apply-VerityQwenConfig
        Start-Verity
        Show-Status
    }
    'Stop' {
        Write-Log '--- Verity stack: Stop ---'
        Stop-Verity
        if (Test-Path $OllamaLauncher) {
            & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $OllamaLauncher -Action Stop
        }
        # DISABLED 2026-05-11 alongside Apply-VerityQwenConfig above --
        # there's nothing to revert because Start didn't apply anything.
        # Revert-VerityQwenConfig
        # Wait briefly for runner processes to finish releasing VRAM,
        # then confirm nothing Verity-related remains on either GPU or CPU.
        Start-Sleep -Seconds 1
        Confirm-EverythingOff
    }
    'Restart' {
        Write-Log '--- Verity stack: Restart ---'
        Stop-Verity
        if (Test-Path $OllamaLauncher) {
            & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $OllamaLauncher -Action Restart
            if ($LASTEXITCODE -ne 0) {
                Write-Log "Ollama launcher restart exited $LASTEXITCODE -- aborting." 'ERROR'
                exit $LASTEXITCODE
            }
        }
        # DISABLED 2026-05-11 -- Apply/Revert pair is no-op for now.
        Start-Verity
        Show-Status
    }
    'Status' {
        Show-Status
    }
}
