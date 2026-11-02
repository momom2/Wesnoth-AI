param([string]$Head = '8eb24002')
$py = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai\.venv\Scripts\python.exe'
$gpu = 'C:\Users\momom2\venvs\wesnoth_rocm\Scripts\python.exe'
$runs = 'C:\Users\momom2\wesnoth_runs'
# --skip-p: prereg departure 31 (the rounds start with --no-p-gate).
$args = "-u scripts\chain_local.py --stage-c $runs\stage_c_local --stage1 $runs\selfplay_t1_local " +
        "--p-out $runs\p_local --rounds-out $runs\rounds_local --e 0.02 --rounds 9 " +
        "--checkpoint training\checkpoints\own2.pt " +
        "--checkpoint-sha256 aea027b2696a9fbce0ebcdc0a8483c9df7d35957934b81b0c62093a02cdd9129 " +
        "--expect-head $Head --gpu-python $gpu --tools-python $py --final-out $runs\final_match " +
        "--heads-only-pass --rounds-n-train 5000 --rounds-end-by `"2026-11-07 10:00`" " +
        "--tail-out $runs\tail_match --continue-rounds 39 --continue-end-by `"2026-11-08 12:00`" " +
        "--continue-final-out $runs\final_match_cont --skip-p"
$cmd = "cmd.exe /c `"`"$py`" $args > $runs\chain.console.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-run' }
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
$cmd
