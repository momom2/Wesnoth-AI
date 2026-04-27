# Launch self-play with a trained transformer checkpoint.
#
# Pairs with cluster/pull_checkpoint.ps1: pull a fresh model back from
# the cluster, then run it locally to see how it actually plays. Wraps
#   python main.py --policy transformer --resume <ckpt> --games <N>
# with sensible defaults for the user's typical workflow.
#
# Defaults:
#   - With no flags: auto-picks the most recently modified
#     supervised*.pt in training/checkpoints/. That's almost always
#     the freshly-pulled one.
#   - -Checkpoint <path>: explicit path (relative or absolute).
#   - -Games N: parallelism. Default 4 (matches NUM_PARALLEL_GAMES
#     in constants.py and what main.py would pick on its own).
#   - -DryRun: print the resolved command, don't actually launch.
#
# Notes:
#   - This launches Wesnoth GUI windows on Windows -- there's no
#     headless option that worked for us locally. Expect 4 (or
#     -Games N) Wesnoth windows to pop up.
#   - Press Ctrl+C in the launching shell to stop. Any in-flight
#     trajectories are saved as part of the trainer's normal
#     checkpoint flow.
#   - Self-play TRAINS on top of the resumed checkpoint by default
#     (the transformer policy has trainable=True). Pass
#     -SkipTraining if you only want pure rollouts for evaluation
#     (currently no flag exists for that on the Python side, so
#     this is informational only).

param(
    [string]$Checkpoint = '',
    [int]$Games = 4,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ($Checkpoint -eq '') {
    # Auto-pick the freshest supervised*.pt by mtime.
    $candidates = @(
        Get-ChildItem -Path 'training/checkpoints/supervised*.pt' `
            -File -ErrorAction SilentlyContinue
    ) | Sort-Object LastWriteTime -Descending
    if ($candidates.Count -eq 0) {
        Write-Error @"
[selfplay] no supervised*.pt found in training/checkpoints/.

Pull one from the cluster first:
  powershell -ExecutionPolicy Bypass -File cluster/pull_checkpoint.ps1
"@
        exit 1
    }
    $Checkpoint = $candidates[0].FullName
    $age = (Get-Date) - $candidates[0].LastWriteTime
    Write-Host ("[selfplay] auto-picked latest: {0} ({1:N1}h old)" -f $Checkpoint, $age.TotalHours)
} else {
    if (-not (Test-Path $Checkpoint)) {
        Write-Error "[selfplay] checkpoint not found: $Checkpoint"
        exit 1
    }
    $Checkpoint = (Resolve-Path $Checkpoint).Path
    Write-Host "[selfplay] checkpoint: $Checkpoint"
}

Write-Host "[selfplay] games: $Games"
Write-Host "[selfplay] policy: transformer"
Write-Host ""

$cmd = @(
    'python', 'main.py',
    '--policy', 'transformer',
    '--resume', $Checkpoint,
    '--games', $Games
)

if ($DryRun) {
    Write-Host "[selfplay] dry run -- would launch:"
    Write-Host "  $($cmd -join ' ')"
    exit 0
}

Write-Host "[selfplay] launching Wesnoth (Ctrl+C in this shell to stop)..."
Write-Host ""

# Hand off to python.exe directly. Output streams to this terminal.
& $cmd[0] @($cmd[1..($cmd.Count - 1)])
exit $LASTEXITCODE
