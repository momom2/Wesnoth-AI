# Sync code-only changes to the ENSTA Mesogip cluster.
#
# Use this when you've edited Python or cluster scripts locally and
# want to push them to the cluster without rebuilding the whole
# wai_cluster_bundle.tar.gz (which re-ships the 390MB corpus). Sends:
#
#   - Every *.py file at the project root.
#   - Every *.py file in tools/.
#   - cluster/{job.sbatch,setup.sh,run.sh,RUNBOOK.md,build_bundle.ps1,sync.ps1}.
#
# Does NOT touch on the cluster:
#   - replays_dataset/, replays_raw/  (corpus -- built once, ship via bundle)
#   - training/checkpoints/            (the in-flight model -- never overwrite)
#   - wesnoth_src/                     (read-only, scrape-time only)
#   - add-ons/wesnoth_ai/              (Lua side, runs on your Windows box)
#   - .venv/                           (built by setup.sh on the cluster)
#
# Usage (from the project root, doesn't matter -- script does its own cd):
#   powershell -ExecutionPolicy Bypass -File cluster\sync.ps1
#   powershell -ExecutionPolicy Bypass -File cluster\sync.ps1 -Restart
#   powershell -ExecutionPolicy Bypass -File cluster\sync.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File cluster\sync.ps1 -RemoteHost foo
#
# Flags:
#   -DryRun       List what would be sent, don't connect to the cluster.
#   -Restart      After extracting, scancel the running job so the chained
#                 follow-up sbatch picks up the new code immediately. Costs
#                 at most ~500 steps (last periodic checkpoint). Without
#                 this, the new code only takes effect at the next natural
#                 walltime hit (up to ~4h).
#   -RemoteHost   ssh alias to use. Default: mesogip_outside.
#   -RemotePath   Project root on the cluster. Default: ~/wesnoth-ai.
#
# Connection cost: ONE ssh round trip. With the proxy chain through
# relais.ensta.fr -> istanbul.ensta.fr that means two password prompts
# per run, whether or not -Restart is passed (the restart commands run
# inside the same ssh session that extracts). A local temp file holds
# the tarball briefly; PowerShell 5.1's binary pipe is unreliable so
# we route the byte stream through cmd.exe.

param(
    [switch]$Restart,
    [switch]$DryRun,
    [string]$RemoteHost = 'mesogip_outside',
    [string]$RemotePath = '~/wesnoth-ai'
)

$ErrorActionPreference = 'Stop'

# cd to the project root regardless of how the script was invoked.
Set-Location (Split-Path -Parent $PSScriptRoot)

# Pick a tar that understands Windows-drive paths. BSD tar (ships with
# Windows 10+ at C:\Windows\System32\tar.exe) does; GNU tar (Git Bash's
# /usr/bin/tar.exe) does NOT -- it tries to resolve `C:\...` as a remote
# host. If the script is launched from a shell whose PATH puts Git Bash
# first (e.g. running powershell from inside Git Bash), the wrong tar
# wins. Prefer System32 explicitly.
$tarBin = 'tar'
$winTar = Join-Path $env:WINDIR 'System32\tar.exe'
if (Test-Path $winTar) {
    $tarBin = $winTar
}

# Build the manifest. Positive list -- we never auto-send anything we
# didn't enumerate, so no risk of clobbering corpus or checkpoints.
$rootPy   = @(Get-ChildItem -Path '*.py' -File | ForEach-Object { $_.Name })
$toolsPy  = @(Get-ChildItem -Path 'tools' -Filter '*.py' -File |
              ForEach-Object { "tools/$($_.Name)" })

$clusterCandidates = @(
    'cluster/job.sbatch',
    'cluster/setup.sh',
    'cluster/run.sh',
    'cluster/RUNBOOK.md',
    'cluster/build_bundle.ps1',
    'cluster/sync.ps1',
    'cluster/pull_checkpoint.ps1',
    'cluster/gui.pyw',
    'cluster/gui.bat'
)
$clusterFiles = @($clusterCandidates | Where-Object { Test-Path $_ })

# unit_stats.json is part of the static corpus shipped via build_bundle,
# but if it's been re-scraped (Wesnoth version bump) it needs to ride
# along. Cheap to send (~250KB) so always include if present.
$extras = @()
if (Test-Path 'unit_stats.json') { $extras += 'unit_stats.json' }

$paths = $rootPy + $toolsPy + $clusterFiles + $extras

Write-Host "[sync] manifest ($($paths.Count) files):"
$paths | ForEach-Object { Write-Host "  $_" }

if ($DryRun) {
    Write-Host ""
    Write-Host "[sync] dry run -- not connecting to ${RemoteHost}."
    exit 0
}

# Build the remote command. Always extracts; with -Restart, also bounces
# the running job. `set -e` so any failure aborts the whole chain.
$remoteCmd = "set -e; cd $RemotePath && tar -xf - && echo '[sync] extracted ok'"

if ($Restart) {
    # `|| true` because run.sh stop returns non-zero if there's no
    # tracked job to scancel -- that's not an error here.
    $remoteCmd += " && (bash cluster/run.sh stop || true)"
    $remoteCmd += " && sleep 2"
    $remoteCmd += " && bash cluster/run.sh status"
}

Write-Host ""

# Strategy: pack to a local temp file, then route the bytes to ssh via
# cmd.exe's native pipe. Direct PS pipe (`tar.exe | ssh.exe`) does NOT
# preserve binary in PowerShell 5.1 -- it inserts a text-conversion
# layer between native processes that corrupts the tarball ("tar:
# Skipping to next header" on the remote). cmd.exe pipes ARE byte-clean
# for native commands, so we hand the pipe step off to it.

# 1. Pack locally.
$tmpTar = Join-Path $env:TEMP "wai_sync_$([guid]::NewGuid().Guid).tar"
Write-Host "[sync] packing $tmpTar ..."
$tarArgs = @('-cf', $tmpTar) + $paths
& $tarBin @tarArgs
if ($LASTEXITCODE -ne 0) {
    Remove-Item $tmpTar -ErrorAction SilentlyContinue
    throw "tar pack failed (code $LASTEXITCODE)"
}
$size = (Get-Item $tmpTar).Length
Write-Host ("[sync] packed: {0:N1} KB" -f ($size / 1KB))

# 2. Stream bytes through cmd.exe pipe -> ssh -> remote tar -xf.
Write-Host "[sync] streaming to ${RemoteHost}:${RemotePath} ..."
# Build the cmd-shell command. Inside the PS double-quoted string,
# embed literal double-quotes with backtick-escape (`"). cmd.exe sees:
#   type "C:\...\wai_sync_xxx.tar" | ssh HOST "cd PATH && tar -xf - && ..."
$cmdLine = "type `"$tmpTar`" | ssh $RemoteHost `"$remoteCmd`""
& cmd /c $cmdLine
$rc = $LASTEXITCODE

# 3. Local cleanup regardless of remote outcome.
Remove-Item $tmpTar -ErrorAction SilentlyContinue

if ($rc -ne 0) {
    throw "[sync] FAILED (exit code $rc) -- files may be partially synced. Verify with: ssh $RemoteHost 'cd $RemotePath && ls -la encoder.py tools/supervised_train.py'"
}

Write-Host ""
Write-Host "[sync] done."
if ($Restart) {
    Write-Host "[sync] cluster job restarted -- watch progress with: ssh $RemoteHost 'bash cluster/run.sh tail'"
} else {
    Write-Host "[sync] new code will activate at the next chain hop. Pass -Restart to bounce the job now."
}
