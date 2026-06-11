# Build a deployable tarball for the ENSTA Mesogip cluster.
#
# Default mode (self-play): tiny bundle, ~10 MB.
#   Includes code (root *.py + tools/), unit_stats.json, the
#   wesnoth_src/data/{multiplayer,core/macros,core/terrain*}
#   slice that scenario_pool / scenario_events / terrain_resolver
#   read at runtime, and the cluster scripts. NO replay corpus.
#   This is what self-play training actually needs.
#
#   `tools/sim_self_play.py` samples scenarios from the ladder map
#   pool (`tools/scenario_pool.py`) and rolls out the simulator
#   in-process; the replay corpus is for supervised warmstart only.
#
# Supervised mode (-IncludeReplays): adds `replays_dataset/`
#   (~120 MB) for the supervised warmstart pre-training run that
#   produces the checkpoint self-play resumes from.
#
# Skips in both modes: __pycache__, *.pyc, local checkpoints (the
# cluster trains fresh), local benchmark/log/scratch dirs, every
# replay-corpus VARIANT (we keep dozens lying around for diff_replay
# experiments — none of them belong on the cluster), and the bulky
# wesnoth_src subdirs that aren't read at sim time.
#
# Output: wai_cluster_bundle.tar.gz at the repo root.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File cluster\build_bundle.ps1
#   powershell -ExecutionPolicy Bypass -File cluster\build_bundle.ps1 -IncludeReplays

param(
    [switch]$IncludeReplays
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$bundle = "wai_cluster_bundle.tar.gz"
$tmp    = "$bundle.tmp"
if (Test-Path $tmp)    { Remove-Item $tmp }
# Note: don't delete the existing $bundle yet — we keep the previous
# (complete) bundle visible in case someone scp's mid-build. We'll
# rename atomically at the end.

# Use Windows 10+ bsdtar from System32 explicitly. If a Git Bash
# /usr/bin/tar.exe is earlier on PATH, GNU tar runs instead and
# (a) misinterprets `C:\...` as a remote host and (b) errors with
# "file changed as we read it" when our own pyc / log writes touch
# the tree mid-archive. bsdtar at System32 has neither problem.
$tarBin = 'tar'
$winTar = Join-Path $env:WINDIR 'System32\tar.exe'
if (Test-Path $winTar) {
    $tarBin = $winTar
}

Write-Host "[bundle] mode: $(if ($IncludeReplays) {'supervised (with replays_dataset)'} else {'self-play (no replay corpus)'})"
Write-Host "[bundle] writing $tmp ..."

$excludes = @(
    "--exclude=*.pyc",
    "--exclude=__pycache__",
    "--exclude=tmp_scratch",
    "--exclude=logs",
    "--exclude=.venv",
    "--exclude=.git",
    "--exclude=.claude",
    "--exclude=training/checkpoints",
    "--exclude=training/logs",
    "--exclude=training/replays",
    "--exclude=benchmarks",
    "--exclude=docs",
    # Bundle-self exclusion. bsdtar tries to add its own output
    # stream when the tarball lives inside the tree it's archiving.
    "--exclude=wai_cluster_bundle.tar.gz",
    "--exclude=wai_cluster_bundle.tar.gz.tmp",
    # Replay-corpus exclusions. We accumulate dozens of named
    # variants over time (re-extract experiments, audit splits,
    # quarantine folders); ALWAYS exclude every variant. If the
    # operator wanted them they'd ask for supervised mode, which
    # opts the canonical `replays_dataset` back in below.
    "--exclude=replays_dataset.old",
    "--exclude=replays_dataset_full",
    "--exclude=replays_dataset_full2",
    "--exclude=replays_dataset_full3",
    "--exclude=replays_dataset_full4",
    "--exclude=replays_dataset_full5",
    "--exclude=replays_dataset_full6",
    "--exclude=replays_dataset_reextract",
    "--exclude=replays_dataset_reextract2",
    "--exclude=replays_dataset_reextract3",
    "--exclude=replays_dataset_reextract4",
    "--exclude=replays_dataset_reextract5",
    "--exclude=replays_dataset_review_rng",
    "--exclude=replays_dataset_set_aside",
    "--exclude=replays_dataset_500",
    "--exclude=replays_dataset_500_v2",
    "--exclude=replays_raw",
    "--exclude=replays_raw_500",
    "--exclude=replays_raw_500_v2",
    "--exclude=replays_raw_500_v3",
    "--exclude=replays_raw_review_gold",
    "--exclude=replays_raw_review_oos",
    "--exclude=replays_raw_review_rng",
    "--exclude=replays_raw_set_aside"
    # wesnoth_src: keep ONLY the bits the sim reads at runtime
    # (scenario events, terrain resolver, faction definitions,
    # macros, schema). Everything else is build artifacts /
    # campaign content / translations / images / sounds.
    "--exclude=wesnoth_src/.git",
    "--exclude=wesnoth_src/attic",
    "--exclude=wesnoth_src/copyrights.csv",
    "--exclude=wesnoth_src/data/campaigns",
    "--exclude=wesnoth_src/data/ai",
    "--exclude=wesnoth_src/data/test",
    "--exclude=wesnoth_src/data/tools",
    "--exclude=wesnoth_src/data/core/units",
    "--exclude=wesnoth_src/data/core/images",
    "--exclude=wesnoth_src/data/core/music",
    "--exclude=wesnoth_src/data/core/sounds",
    "--exclude=wesnoth_src/data/core/terrain-graphics",
    "--exclude=wesnoth_src/data/core/encyclopedia",
    "--exclude=wesnoth_src/data/core/help.cfg",
    "--exclude=wesnoth_src/data/core/about*",
    "--exclude=wesnoth_src/data/gui",
    "--exclude=wesnoth_src/data/themes",
    "--exclude=wesnoth_src/po",
    "--exclude=wesnoth_src/src",
    "--exclude=wesnoth_src/utils",
    "--exclude=wesnoth_src/packaging",
    "--exclude=wesnoth_src/projectfiles",
    "--exclude=wesnoth_src/cmake",
    "--exclude=wesnoth_src/changelogs",
    "--exclude=wesnoth_src/manual",
    "--exclude=wesnoth_src/m4",
    "--exclude=wesnoth_src/doc",
    "--exclude=wesnoth_src/fonts",
    "--exclude=wesnoth_src/icons",
    "--exclude=wesnoth_src/images",
    "--exclude=wesnoth_src/sounds",
    "--exclude=wesnoth_src/saves"
)

# Self-play default: also exclude the canonical replays_dataset.
# Supervised mode opts it back in by simply NOT excluding it.
if (-not $IncludeReplays) {
    $excludes += @("--exclude=replays_dataset")
} else {
    if (-not (Test-Path 'replays_dataset')) {
        throw "[bundle] -IncludeReplays passed but replays_dataset/ doesn't exist."
    }
}

$args = $excludes + @("-czf", $tmp, ".")

& $tarBin @args
if ($LASTEXITCODE -ne 0) { throw "tar failed with code $LASTEXITCODE" }

# Atomic rename: a partial $tmp can't be mistaken for a complete bundle.
if (Test-Path $bundle) { Remove-Item $bundle }
Move-Item $tmp $bundle

$size = (Get-Item $bundle).Length / 1MB
Write-Host ("[bundle] done. {0} = {1:N1} MB" -f $bundle, $size)
Write-Host "[bundle] next: scp $bundle mesogip_outside:~/"
