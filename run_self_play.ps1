# Launch self-play with a trained transformer checkpoint.
#
# Two modes -- pick one:
#
#   TRAINING mode (default): N parallel Wesnoth processes, fast turbo
#   (10x), no animations, the transformer keeps training on the
#   trajectories it generates. Pair this with -Games 4 (the default)
#   for sensible local throughput.
#
#   DISPLAY mode (-Display): ONE Wesnoth window, 2x turbo,
#   animations on so a human can follow what's happening. Training is
#   disabled -- the loaded checkpoint is treated as read-only. Useful
#   for sanity-checking a fresh checkpoint pull, demoing the model,
#   or just watching it play.
#
# Pairs with cluster/pull_checkpoint.ps1: pull a fresh model back from
# the cluster, then either train on top of it OR watch one game.
#
# Defaults:
#   - With no -Checkpoint: auto-picks the most recently modified
#     supervised*.pt in training/checkpoints/. That's almost always
#     the freshly-pulled one.
#   - -Games N (training mode only): parallelism. Default 4. Ignored
#     in -Display mode (which forces 1).
#   - -DryRun: print the resolved command, don't actually launch.
#
# Notes:
#   - This launches Wesnoth GUI windows on Windows -- there's no
#     headless option that worked for us locally. In training mode,
#     expect -Games windows. In display mode, just one.
#   - Press Ctrl+C in the launching shell to stop. The GUI's Cancel
#     button uses taskkill /T to bring down the whole tree if you
#     launched via cluster/gui.pyw.

param(
    [string]$Checkpoint = '',
    [int]$Games = 4,
    [switch]$Display,
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

if ($Display) {
    Write-Host "[selfplay] mode  : DISPLAY (1 game, 2x turbo, animations on, no training)"
    $effectiveGames = 1
} else {
    Write-Host "[selfplay] mode  : TRAINING ($Games parallel games, fast turbo, training on)"
    $effectiveGames = $Games
}
Write-Host "[selfplay] policy: transformer"
Write-Host ""

$cmd = @(
    'python', 'main.py',
    '--policy', 'transformer',
    '--resume', $Checkpoint,
    '--games', $effectiveGames
)
if ($Display) {
    $cmd += '--display'
}

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
