# ============================================================================
#  verify.ps1 - direct-call workaround for the Multi-Agent Verity
# ============================================================================
#
#  Bypasses LM Studio's chat UI (which currently fails to pass MCP tools to
#  the model). Calls the verity's MCP endpoint directly and prints the
#  verdict.
#
#  Usage:
#    .\verify.ps1 -Question "..." -Answer "..."
#    .\verify.ps1 -Question "..." -Answer "..." -Mode deep
#    .\verify.ps1 -Question "..." -Answer "..." -Mode deeper -UseNli $false
#    .\verify.ps1 -Question "..." -Answer "..." -TaskType reasoning
#    .\verify.ps1 -Question "..." -Answer "..." -PriorContext "..." -ContextMode with_context
#    .\verify.ps1 -FromClipboard    # pulls last Qwen answer from clipboard, prompts for question
#    .\verify.ps1 -Raw              # print the full JSON, no pretty formatting
#
#  Exit codes:
#    0 - pass
#    1 - warn
#    2 - fail
#    3 - error (critics unavailable, pipeline failure, etc.)
# ============================================================================

[CmdletBinding()]
param(
    [string]$Question,
    [string]$Answer,
    [ValidateSet('standard','deep','deeper')]
    [string]$Mode = 'standard',
    [ValidateSet('code','prose','reasoning','research','auto')]
    [string]$TaskType = 'auto',
    [ValidateSet('minimal','with_context','full')]
    [string]$ContextMode = 'minimal',
    [string]$PriorContext = '',
    [bool]$UseNli = $true,
    [switch]$FromClipboard,
    [switch]$Raw,
    [string]$Url = 'http://localhost:8090'
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Argument fallback: pull answer from clipboard if -FromClipboard is set
# ---------------------------------------------------------------------------
if ($FromClipboard) {
    if ([string]::IsNullOrWhiteSpace($Answer)) {
        $Answer = Get-Clipboard -Raw
        if ([string]::IsNullOrWhiteSpace($Answer)) {
            Write-Host "Clipboard is empty. Copy an answer first." -ForegroundColor Red
            exit 3
        }
        Write-Host "Using clipboard as answer ($(($Answer.Length)) chars)." -ForegroundColor DarkGray
    }
    if ([string]::IsNullOrWhiteSpace($Question)) {
        $Question = Read-Host "What question did the answer respond to?"
    }
}

if ([string]::IsNullOrWhiteSpace($Question) -or [string]::IsNullOrWhiteSpace($Answer)) {
    Write-Host "Error: -Question and -Answer are required (or use -FromClipboard)." -ForegroundColor Red
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\verify.ps1 -Question 'What is 2+2?' -Answer '5'"
    Write-Host "  .\verify.ps1 -FromClipboard"
    exit 3
}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
try {
    $health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 3
    if ($health.status -ne 'ok') {
        Write-Host "Verity unhealthy: $($health | ConvertTo-Json -Compress)" -ForegroundColor Red
        exit 3
    }
} catch {
    Write-Host "Verity not reachable at $Url. Is it running?" -ForegroundColor Red
    Write-Host "  Start it with: node C:\AI\verify\dist\index.js" -ForegroundColor DarkGray
    Write-Host "  Or:            & 'C:\AI\experiments\_manual-start.ps1'" -ForegroundColor DarkGray
    exit 3
}

# ---------------------------------------------------------------------------
# MCP session: initialize -> notifications/initialized -> tools/call
# ---------------------------------------------------------------------------
$headers = @{
    'Content-Type' = 'application/json'
    'Accept'       = 'application/json, text/event-stream'
}

# Step 1: initialize - server assigns the session id via response header
$initBody = @{
    jsonrpc = '2.0'
    id      = 1
    method  = 'initialize'
    params  = @{
        protocolVersion = '2024-11-05'
        capabilities    = @{}
        clientInfo      = @{ name = 'verify-ps1'; version = '1.0' }
    }
} | ConvertTo-Json -Depth 10

try {
    $initResp = Invoke-WebRequest -Uri "$Url/mcp" -Method Post -Headers $headers `
                                  -Body $initBody -TimeoutSec 10 -UseBasicParsing
} catch {
    Write-Host "MCP initialize failed: $_" -ForegroundColor Red
    exit 3
}

# Session id is returned in a response header; iterate to handle both
# hashtable and list-style Headers collections across PowerShell versions
$sessionId = $null
foreach ($key in $initResp.Headers.Keys) {
    if ($key -ieq 'mcp-session-id') {
        $val = $initResp.Headers[$key]
        if ($val -is [string]) { $sessionId = $val } else { $sessionId = $val[0] }
        break
    }
}
if ([string]::IsNullOrWhiteSpace($sessionId)) {
    Write-Host "MCP server did not return a session id. Cannot continue." -ForegroundColor Red
    exit 3
}

# Build a new hashtable rather than cloning (Hashtable.Clone isn't reliable)
$sessionHeaders = @{
    'Content-Type'   = 'application/json'
    'Accept'         = 'application/json, text/event-stream'
    'mcp-session-id' = $sessionId
}

# Step 2: notifications/initialized
$initdBody = @{ jsonrpc = '2.0'; method = 'notifications/initialized' } | ConvertTo-Json
Invoke-WebRequest -Uri "$Url/mcp" -Method Post -Headers $sessionHeaders -Body $initdBody -TimeoutSec 10 -UseBasicParsing | Out-Null

# Step 3: tools/call
$callArgs = @{
    question     = $Question
    answer       = $Answer
    mode         = $Mode
    task_type    = $TaskType
    context_mode = $ContextMode
    use_nli      = $UseNli
}
if ($ContextMode -eq 'with_context' -and -not [string]::IsNullOrWhiteSpace($PriorContext)) {
    $callArgs['prior_context'] = $PriorContext
}

$callBody = @{
    jsonrpc = '2.0'
    id      = 2
    method  = 'tools/call'
    params  = @{
        name      = 'verify_previous_answer'
        arguments = $callArgs
    }
} | ConvertTo-Json -Depth 10

# Longer timeout for deep/deeper modes
$timeout = switch ($Mode) {
    'standard' { 90 }
    'deep'     { 180 }
    'deeper'   { 240 }
}

Write-Host ""
Write-Host "Verifying... (mode=$Mode) this can take $timeout seconds" -ForegroundColor DarkGray

$start = Get-Date
try {
    $callResp = Invoke-WebRequest -Uri "$Url/mcp" -Method Post -Headers $sessionHeaders `
                                  -Body $callBody -TimeoutSec $timeout
} catch {
    Write-Host "Verity call failed: $_" -ForegroundColor Red
    exit 3
}
$elapsed = [int]((Get-Date) - $start).TotalSeconds

# ---------------------------------------------------------------------------
# Parse SSE response (data: <json>)
#
# A single MCP tools/call may emit multiple `data:` events: server-side
# heartbeats, progress notifications, and finally the JSON-RPC response we
# care about (the one whose `id` matches our request id of 2). Earlier
# revisions of this script grabbed only the FIRST data: line, which would
# silently drop the real response if the server emitted an interim event.
# ---------------------------------------------------------------------------
$raw = $callResp.Content
$dataLines = @($raw -split "`n" | Where-Object { $_ -match '^data:\s*\{' })
if ($dataLines.Count -eq 0) {
    Write-Host "Could not parse MCP response." -ForegroundColor Red
    Write-Host $raw
    exit 3
}

$mcpResponse = $null
foreach ($line in $dataLines) {
    $candidate = $line -replace '^data:\s*', ''
    try {
        $obj = $candidate | ConvertFrom-Json
    } catch {
        continue
    }
    # The JSON-RPC response we sent has id=2; prefer matching that. Fall
    # back to any object that has a `result` or `error` field (i.e. is
    # itself a response, not a notification).
    if ($obj.id -eq 2 -or $null -ne $obj.result -or $null -ne $obj.error) {
        $mcpResponse = $obj
        if ($obj.id -eq 2) { break }
    }
}
if ($null -eq $mcpResponse) {
    Write-Host "Could not parse MCP response (no matching JSON-RPC reply)." -ForegroundColor Red
    Write-Host $raw
    exit 3
}

if ($mcpResponse.error) {
    Write-Host "MCP error: $($mcpResponse.error.message)" -ForegroundColor Red
    exit 3
}

# Tool result is a text content block containing the verity JSON
$verdictText = $mcpResponse.result.content[0].text
$verdict = $verdictText | ConvertFrom-Json

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if ($Raw) {
    Write-Output $verdictText
    switch ($verdict.consensus) {
        'pass'  { exit 0 }
        'warn'  { exit 1 }
        'fail'  { exit 2 }
        default { exit 3 }
    }
}

# Pretty formatting
$consensusColor = switch ($verdict.consensus) {
    'pass'  { 'Green' }
    'warn'  { 'Yellow' }
    'fail'  { 'Red' }
    default { 'DarkGray' }
}

$consensusSymbol = switch ($verdict.consensus) {
    'pass'  { '[PASS]' }
    'warn'  { '[WARN]' }
    'fail'  { '[FAIL]' }
    default { '[ERR ]' }
}

Write-Host ""
Write-Host "================================================================"
Write-Host -NoNewline "  "
Write-Host -NoNewline $consensusSymbol -ForegroundColor $consensusColor
Write-Host "  $($verdict.summary)"
Write-Host "================================================================"
Write-Host ""

Write-Host "Critics:" -ForegroundColor Cyan
foreach ($prop in $verdict.critics.PSObject.Properties) {
    $c = $prop.Value
    if ($c.unavailable) {
        Write-Host ("  {0,-20} UNAVAILABLE  ({1}ms)  {2}" -f $c.display_name, $c.latency_ms, $c.error) -ForegroundColor DarkGray
    } else {
        $cColor = switch ($c.verdict) {
            'pass' { 'Green' }
            'warn' { 'Yellow' }
            'fail' { 'Red' }
            default { 'DarkGray' }
        }
        Write-Host -NoNewline ("  {0,-20} " -f $c.display_name)
        Write-Host -NoNewline ("{0,-5}" -f $c.verdict) -ForegroundColor $cColor
        Write-Host (" severity={0}  ({1}ms)" -f $c.severity, $c.latency_ms)
        foreach ($concern in $c.concerns) {
            Write-Host "      - $concern" -ForegroundColor DarkGray
        }
        foreach ($fix in $c.suggested_fixes) {
            Write-Host "      -> $fix" -ForegroundColor DarkCyan
        }
    }
}

# NLI
if ($verdict.nli_check) {
    Write-Host ""
    Write-Host "NLI check:" -ForegroundColor Cyan
    if ($verdict.nli_check.ran) {
        Write-Host ("  claims checked: {0}  contradictions: {1}  unsupported: {2}" -f `
            $verdict.nli_check.claims_checked,
            $verdict.nli_check.contradictions.Count,
            $verdict.nli_check.unsupported.Count)
        foreach ($c in $verdict.nli_check.contradictions) {
            Write-Host ("    CONTRADICTED (conf {0:P0}): {1}" -f $c.confidence, $c.claim) -ForegroundColor Red
        }
        foreach ($u in $verdict.nli_check.unsupported) {
            Write-Host "    UNSUPPORTED: $($u.claim)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  skipped: $($verdict.nli_check.notes)" -ForegroundColor DarkGray
    }
}

# Deep-mode signals
if ($verdict.consistency_check) {
    Write-Host ""
    Write-Host "Consistency check (deep):" -ForegroundColor Cyan
    if ($verdict.consistency_check.ran) {
        Write-Host ("  samples={0} claims={1} divergence={2}" -f `
            $verdict.consistency_check.samples_generated,
            $verdict.consistency_check.claims_checked,
            $verdict.consistency_check.divergence_score)
    } else {
        Write-Host "  skipped: $($verdict.consistency_check.notes)" -ForegroundColor DarkGray
    }
}
if ($verdict.perplexity) {
    Write-Host ""
    Write-Host "Perplexity (deep):" -ForegroundColor Cyan
    if ($verdict.perplexity.ran) {
        Write-Host ("  method={0} tokens={1} perplexity={2} low-conf spans={3}" -f `
            $verdict.perplexity.method,
            $verdict.perplexity.tokens_scored,
            $verdict.perplexity.perplexity,
            $verdict.perplexity.low_confidence_spans.Count)
    } else {
        Write-Host "  skipped: $($verdict.perplexity.notes)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host ("Wall-clock: {0}s  (verity reported {1}ms)" -f $elapsed, $verdict.latency_ms) -ForegroundColor DarkGray

switch ($verdict.consensus) {
    'pass'  { exit 0 }
    'warn'  { exit 1 }
    'fail'  { exit 2 }
    default { exit 3 }
}
