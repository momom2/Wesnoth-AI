# Build a deployable tarball for the ENSTA Mesogip cluster.
#
# Bundles:
#   - Code: every Python file at the repo root, plus tools/.
#   - Data: unit_stats.json, the replays_dataset/ corpus, and the slice of
#     wesnoth_src/ that scenario_events.py + scenarios.py read at runtime.
#   - Cluster scripts: cluster/setup.sh, cluster/run.sh.
#
# Skips: __pycache__, *.pyc, the local checkpoints (the cluster trains
# fresh), the local-only smoke-test artifacts in tmp_scratch/, the bulky
# wesnoth_src/data/core/units/ tree (we already pre-baked the per-unit
# stats into unit_stats.json so the cluster doesn't need it).
#
# Output: wai_cluster_bundle.tar.gz at the repo root.
# Expected size: ~390 MB (dominated by replays_dataset).

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$bundle = "wai_cluster_bundle.tar.gz"
$tmp    = "$bundle.tmp"
if (Test-Path $tmp)    { Remove-Item $tmp }
# Note: don't delete the existing $bundle yet — we keep the previous
# (complete) bundle visible in case someone scp's mid-build. We'll
# rename atomically at the end.

# Use system tar (Windows 10+ ships bsdtar). The --exclude flags trim
# fat we don't need on the cluster.
Write-Host "[bundle] writing $tmp ..."
$args = @(
    "-czf", $tmp,
    "--exclude=*.pyc",
    "--exclude=__pycache__",
    "--exclude=tmp_scratch",
    "--exclude=logs",
    "--exclude=.venv",
    "--exclude=training/checkpoints",
    "--exclude=replays_dataset.old",
    "--exclude=replays_raw",
    "--exclude=wai_cluster_bundle.tar.gz",
    # wesnoth_src: keep ONLY the bits scenario_events / scenarios touch.
    "--exclude=wesnoth_src/.git",
    "--exclude=wesnoth_src/data/campaigns",
    "--exclude=wesnoth_src/data/ai",
    "--exclude=wesnoth_src/data/core/units",
    "--exclude=wesnoth_src/data/core/images",
    "--exclude=wesnoth_src/data/core/music",
    "--exclude=wesnoth_src/data/core/sounds",
    "--exclude=wesnoth_src/po",
    "--exclude=wesnoth_src/src",
    "--exclude=wesnoth_src/utils",
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
    "--exclude=wesnoth_src/saves",
    "."
)
& tar @args
if ($LASTEXITCODE -ne 0) { throw "tar failed with code $LASTEXITCODE" }

# Atomic rename: a partial $tmp can't be mistaken for a complete bundle.
if (Test-Path $bundle) { Remove-Item $bundle }
Move-Item $tmp $bundle

$size = (Get-Item $bundle).Length / 1MB
Write-Host ("[bundle] done. {0} = {1:N1} MB" -f $bundle, $size)
Write-Host "[bundle] next: scp $bundle mesogip_outside:~/"
