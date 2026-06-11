# Pull a training checkpoint back from the ENSTA Mesogip cluster.
#
# The cluster writes these into training/checkpoints/:
#
#   sim_selfplay.pt            -- "rolling self-play target". Updated
#                                 every --save-every iters by
#                                 sim_self_play.py (default 5).
#                                 Always present once self-play has
#                                 run. THE DEFAULT PULL TARGET.
#   sim_selfplay_archive_*.pt  -- periodic snapshots maintained by
#                                 cluster/cleanup.py (tier-retained,
#                                 typically ~1-10 kept). Useful as
#                                 the "reference" arg for eval_sim.
#   supervised.pt              -- supervised "rolling latest". Updated
#                                 mid-epoch by the supervised trainer.
#                                 Present only if supervised is the
#                                 active workflow.
#   supervised_epochN.pt       -- immutable per-epoch supervised
#                                 snapshot. Most relevant epoch is
#                                 typically already on the cluster as
#                                 the warm-start anchor for self-play.
#
# Defaults (flipped 2026-05-13 to match self-play-primary workflow):
#   - With no flags: pulls sim_selfplay.pt -- the active self-play
#     target. Same file that `Sync + Continue (self-play)` pushes
#     BACK, so the default Pull is the natural "see what cluster
#     trained today" action.
#   - -Supervised pulls the highest supervised_epoch*.pt (the old
#     pre-2026-05-13 default; still useful as the supervised
#     baseline anchor).
#   - -Rolling pulls supervised.pt (the mid-epoch supervised
#     latest -- rarely what you want once self-play is the
#     primary workflow).
#   - -Epoch N pulls supervised_epochN.pt specifically.
#   - -List doesn't pull anything; lists what's on the cluster
#     (size + mtime), now including sim_selfplay.pt and
#     sim_selfplay_archive_*.pt alongside the supervised files.
#
# Connection cost: 1 ssh round trip per pull. Through the relais ->
# istanbul proxy chain that's two password prompts; piped through
# the GUI's askpass plumbing so it's silent in the GUI workflow.

param(
    [switch]$Supervised,
    [switch]$Rolling,
    [switch]$Archive,
    [string]$ArchiveStamp = '',
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
    # List both supervised AND sim_selfplay variants. The shell glob
    # silently emits nothing for a class that's absent, so this
    # works whether the cluster has only supervised, only self-play,
    # or both. `2>/dev/null` suppresses the "no such file" stderr.
    & ssh $RemoteHost "ls -lh $remoteCkptDir/sim_selfplay*.pt $remoteCkptDir/supervised*.pt 2>/dev/null"
    exit $LASTEXITCODE
}

# Decide which file to pull. Precedence matches PowerShell parameter
# semantics: explicit flags win over defaults; `-Epoch N` is the most
# specific and ties to a numbered file, so it goes first.
if ($Epoch -ge 0) {
    $remoteFile = "supervised_epoch$Epoch.pt"
    $resolveCmd = "ls $remoteCkptDir/$remoteFile 2>/dev/null"
} elseif ($Rolling) {
    # Legacy supervised-rolling mode. Kept for backward compat with
    # workflows that pre-date self-play. Almost certainly NOT what
    # the operator wants today; we leave the flag working but the
    # default has moved on.
    $remoteFile = 'supervised.pt'
    $resolveCmd = "ls $remoteCkptDir/$remoteFile 2>/dev/null"
} elseif ($Supervised) {
    # Old default behavior, opt-in now. Pick the highest-numbered
    # supervised_epochN.pt. Sort with -V so epoch10 > epoch9.
    $resolveCmd = "ls $remoteCkptDir/supervised_epoch*.pt 2>/dev/null | sort -V | tail -n1"
    $remoteFile = $null   # we'll learn it from the resolve step
} elseif ($Archive) {
    # Pull a tier-retained snapshot from the cluster
    # (`sim_selfplay_archive_<YYYYMMDD-HHMMSS>.pt`). Lets the
    # operator evaluate against a specific past state, e.g. for
    # use as `--reference` in eval_sim. With no -ArchiveStamp,
    # pulls the FRESHEST archive (highest mtime); with a stamp,
    # pulls the exact named one.
    if ($ArchiveStamp) {
        $remoteFile = "sim_selfplay_archive_${ArchiveStamp}.pt"
        $resolveCmd = "ls $remoteCkptDir/$remoteFile 2>/dev/null"
    } else {
        $resolveCmd = "ls -t $remoteCkptDir/sim_selfplay_archive_*.pt 2>/dev/null | head -n1"
        $remoteFile = $null   # learn it from resolve
    }
} else {
    # NEW DEFAULT (2026-05-13): pull sim_selfplay.pt. This is the
    # file the cluster writes after every save_every iters and the
    # one we want to inspect/eval/push-back. If the cluster hasn't
    # run any self-play yet, the resolve will surface a clear
    # error and the operator can fall back to -Supervised.
    $remoteFile = 'sim_selfplay.pt'
    $resolveCmd = "ls $remoteCkptDir/$remoteFile 2>/dev/null"
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
    } elseif ($Rolling) {
        throw "[pull] supervised.pt not found on the cluster (supervised training may not have run)."
    } elseif ($Supervised) {
        throw "[pull] no per-epoch supervised snapshots on the cluster. The first epoch may not have finished. Try -List to see what's available."
    } elseif ($Archive) {
        if ($ArchiveStamp) {
            throw "[pull] sim_selfplay_archive_${ArchiveStamp}.pt not found on the cluster. Try -List to see which timestamps are available."
        } else {
            throw "[pull] no sim_selfplay_archive_*.pt on the cluster (cleanup.py hasn't created any snapshots yet). Try -List or pull the rolling self-play instead."
        }
    } else {
        # Default mode (self-play) failure. Most likely cause:
        # cluster hasn't trained self-play yet. Point at the
        # supervised fallback explicitly.
        throw "[pull] sim_selfplay.pt not found on the cluster (self-play training may not have started yet). Try -Supervised to pull the supervised warm-start anchor instead, or -List to see what's available."
    }
}
$remoteFull = $remoteFull.Trim()
$remoteFile = Split-Path -Leaf $remoteFull

# Make sure the local target dir exists.
New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
$localFull = Join-Path $LocalPath $remoteFile

# Atomic-ish pull: scp into a sibling .tmp file, THEN Move-Item over
# the final path. Move-Item on the same NTFS volume uses
# MoveFileEx(MOVEFILE_REPLACE_EXISTING) which is atomic at the
# filesystem level -- a self-play process reading the same path
# concurrently either sees the OLD file in full or the NEW file in
# full, never a torn half-written buffer. Without this, scp writes
# directly to the target name and a concurrent reader can hit a
# truncated file (PyTorch torch.load throws "PytorchStreamReader
# failed reading zip archive" or similar). One-line fix; trivial
# insurance.
$tmpFull = "$localFull.tmp"
# Clean up any orphan .tmp from a prior aborted pull.
if (Test-Path $tmpFull) { Remove-Item -LiteralPath $tmpFull -Force }

Write-Host "[pull] scp ${RemoteHost}:${remoteFull} -> $tmpFull"
& scp "${RemoteHost}:${remoteFull}" $tmpFull
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $tmpFull) { Remove-Item -LiteralPath $tmpFull -Force }
    throw "[pull] scp failed (exit $LASTEXITCODE)"
}

# Atomic replace. -Force overrides the existing $localFull.
Move-Item -LiteralPath $tmpFull -Destination $localFull -Force

$size = (Get-Item $localFull).Length
Write-Host ("[pull] done: {0} ({1:N1} MB)" -f $localFull, ($size / 1MB))
Write-Host ""
Write-Host "[pull] try locally with:"
Write-Host "  python tools/sim_self_play.py --checkpoint-in `"$localFull`""
Write-Host "  python tools/sim_demo_game.py    # one-game demo -> Wesnoth-loadable .bz2"
Write-Host "  (or use the GUI's 'Train (self-play)' / 'Display 1 game' buttons --"
Write-Host "   both auto-pick the freshest *.pt under training\checkpoints\)"
