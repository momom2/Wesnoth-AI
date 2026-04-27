# Pull a supervised-training checkpoint back from the ENSTA Mesogip cluster.
#
# The cluster's job writes two kinds of files into training/checkpoints/:
#
#   supervised.pt              -- "rolling latest". Updated every
#                                 --ckpt-every steps (default 500).
#                                 Always present, may be mid-epoch.
#   supervised_epochN.pt       -- immutable per-epoch snapshot.
#                                 Written when epoch N finishes.
#
# Defaults:
#   - With no flags: pulls the LATEST per-epoch snapshot (highest N).
#     Best for in-situ evaluation against a stable, reproducible
#     checkpoint.
#   - -Rolling pulls supervised.pt (the live latest, possibly
#     mid-epoch).
#   - -Epoch N pulls supervised_epochN.pt specifically.
#   - -List doesn't pull anything; just shows what's on the cluster
#     (sizes + mtimes).
#
# Connection cost: 1 ssh round trip (List, default, or Epoch). Through
# the relais -> istanbul proxy chain that's two password prompts.

param(
    [switch]$Rolling,
    [int]$Epoch = -1,
    [switch]$List,
    [string]$RemoteHost = 'mesogip_outside',
    [string]$RemotePath = '~/wesnoth-ai',
    [string]$LocalPath  = 'training/checkpoints'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$remoteCkptDir = "$RemotePath/training/checkpoints"

if ($List) {
    Write-Host "[pull] checkpoints on ${RemoteHost}:${remoteCkptDir}:"
    & ssh $RemoteHost "ls -lh $remoteCkptDir/supervised*.pt 2>/dev/null"
    exit $LASTEXITCODE
}

# Decide which file to pull.
if ($Epoch -ge 0) {
    $remoteFile = "supervised_epoch$Epoch.pt"
    $resolveCmd = "ls $remoteCkptDir/$remoteFile 2>/dev/null"
} elseif ($Rolling) {
    $remoteFile = 'supervised.pt'
    $resolveCmd = "ls $remoteCkptDir/$remoteFile 2>/dev/null"
} else {
    # Pick the highest-numbered supervised_epochN.pt. Sort with -V so
    # epoch10 sorts after epoch9. tail -n1 gives the highest.
    $resolveCmd = "ls $remoteCkptDir/supervised_epoch*.pt 2>/dev/null | sort -V | tail -n1"
    $remoteFile = $null   # we'll learn it from the resolve step
}

# Resolve the path on the cluster (one ssh trip).
Write-Host "[pull] resolving target on ${RemoteHost} ..."
$remoteFull = (& ssh $RemoteHost $resolveCmd) | Select-Object -First 1
if ($LASTEXITCODE -ne 0) {
    throw "[pull] ssh resolve failed (exit $LASTEXITCODE)"
}
if (-not $remoteFull) {
    if ($Epoch -ge 0) {
        throw "[pull] no $remoteFile on the cluster. Try -List to see what's available."
    } elseif (-not $Rolling) {
        throw "[pull] no per-epoch snapshots yet on the cluster. The first epoch may not have finished. Try -Rolling to grab the live supervised.pt instead, or -List to see."
    } else {
        throw "[pull] supervised.pt not found on the cluster (training may not have started)."
    }
}
$remoteFull = $remoteFull.Trim()
$remoteFile = Split-Path -Leaf $remoteFull

# Make sure the local target dir exists.
New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
$localFull = Join-Path $LocalPath $remoteFile

# scp it back. Single connection.
Write-Host "[pull] scp ${RemoteHost}:${remoteFull} -> $localFull"
& scp "${RemoteHost}:${remoteFull}" $localFull
if ($LASTEXITCODE -ne 0) {
    throw "[pull] scp failed (exit $LASTEXITCODE)"
}

$size = (Get-Item $localFull).Length
Write-Host ("[pull] done: {0} ({1:N1} MB)" -f $localFull, ($size / 1MB))
Write-Host ""
Write-Host "[pull] try in self-play with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File run_self_play.ps1 -Checkpoint `"$localFull`""
Write-Host "  (or just run_self_play.ps1 with no args -- it auto-picks the freshest *.pt)"
