param(
    [string]$SourceDir = "",
    [string]$Remote = "gdrive:AI-Agent-Memory",
    [switch]$Restore
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

python -m pip install -e $repoRoot | Out-Host

$stateHome = Join-Path $env:USERPROFILE ".v1-memory"
New-Item -ItemType Directory -Force -Path $stateHome | Out-Null
$cfgTarget = Join-Path $stateHome "context_finder_config.json"
if (-not (Test-Path $cfgTarget)) {
    Copy-Item (Join-Path $repoRoot "examples\context_finder_config.json") $cfgTarget
}

if ($Restore) {
    if (-not $SourceDir) {
        foreach ($cand in @("G:\My Drive\AI-Agent-Memory", "H:\My Drive\AI-Agent-Memory", "$env:USERPROFILE\Google Drive\My Drive\AI-Agent-Memory")) {
            if (Test-Path $cand) { $SourceDir = $cand; break }
        }
    }
    if ($SourceDir -and (Test-Path $SourceDir)) {
        $map = @{
            "providers\antigravity" = (Join-Path $env:USERPROFILE ".gemini")
            "providers\claude"      = (Join-Path $env:USERPROFILE ".claude")
            "providers\codex"       = (Join-Path $env:USERPROFILE ".codex")
            "providers\v1-memory"   = $stateHome
        }
        foreach ($kv in $map.GetEnumerator()) {
            $src = Join-Path $SourceDir $kv.Key
            if (Test-Path $src) {
                robocopy $src $kv.Value /E /XO /R:2 /W:1 /NFL /NDL /NJH /NJS | Out-Null
            }
        }
        Write-Host "Restored backups from $SourceDir" -ForegroundColor Green
    } else {
        $rclone = (Get-Command rclone -ErrorAction SilentlyContinue).Source
        if (-not $rclone) { throw "No mounted Google Drive folder or runnable rclone found." }
        & $rclone copy "$Remote/providers/antigravity" (Join-Path $env:USERPROFILE ".gemini") --update
        & $rclone copy "$Remote/providers/claude" (Join-Path $env:USERPROFILE ".claude") --update
        & $rclone copy "$Remote/providers/codex" (Join-Path $env:USERPROFILE ".codex") --update
        & $rclone copy "$Remote/providers/v1-memory" $stateHome --update
    }
}

$syncExe = (Get-Command v1-sync -ErrorAction SilentlyContinue).Source
if ($syncExe) {
    $action = New-ScheduledTaskAction -Execute $syncExe
    $trigger = New-ScheduledTaskTrigger -Daily -At 9:00PM
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
    Register-ScheduledTask -TaskName "V1AgentMemoryDailySync" -Action $action -Trigger $trigger -Settings $settings -Description "Daily incremental WAL-safe backup for AI CLI conversations and skills." -Force | Out-Null
    Write-Host "Registered Windows Scheduled Task 'V1AgentMemoryDailySync' (Daily 9:00 PM + catch-up)." -ForegroundColor Green
}

v1-context --reindex | Out-Null
Write-Host "Bootstrap complete! Run v1-status to verify backup state." -ForegroundColor Cyan
