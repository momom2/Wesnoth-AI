param([Parameter(Mandatory=$true)][string]$Head, [string]$EndBy = '2026-11-13 18:00')
# Prereg departure 43: the lean line, rebased after round 3 (departure 33), from round 19's replay
# (departure 42's retry rule). rounds_local/STOP must name round 3 (a REDEPLOY stop), and round 19's
# play, seed_base.txt and held.txt must be set by hand first (departure 43).
$py = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai\.venv\Scripts\python.exe'
$gpu = 'C:\Users\momom2\venvs\wesnoth_rocm\Scripts\python.exe'
$runs = 'C:\Users\momom2\wesnoth_runs'
$args = "-u scripts\chain_local.py --stage-c $runs\stage_c_local --stage1 $runs\selfplay_t1_local " +
        "--p-out $runs\p_local --rounds-out $runs\rounds_local --e 0.02 --rounds 39 " +
        "--checkpoint training\checkpoints\own2.pt " +
        "--checkpoint-sha256 aea027b2696a9fbce0ebcdc0a8483c9df7d35957934b81b0c62093a02cdd9129 " +
        "--expect-head $Head --gpu-python $gpu --tools-python $py --final-out $runs\final_match_lean " +
        "--heads-only-pass --rounds-n-train 5000 --rounds-end-by `"$EndBy`" --skip-p --after-stop 3 " +
        "--rounds-held 1250 --rounds-average 8 --rounds-lean --final-after-failure"   # departures 43, 44
$cmd = "cmd.exe /c `"`"$py`" $args >> $runs\chain_lean.console.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-run' }
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
$cmd
