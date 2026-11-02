$py = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai\.venv\Scripts\python.exe'
$gpu = 'C:\Users\momom2\venvs\wesnoth_rocm\Scripts\python.exe'
$runs = 'C:\Users\momom2\wesnoth_runs'
# Prereg departure 30: round 1 alone, with the chain's own rounds arguments.
New-Item -ItemType Directory -Force "$runs\rounds_local" | Out-Null
$args = "-u scripts\rounds_local.py --stage1 $runs\selfplay_t1_local --stage-c $runs\stage_c_local " +
        "--checkpoint training\checkpoints\own2.pt --tools-python $py --keep-awake " +
        "--out $runs\rounds_local --rounds 1 --p-out $runs\p_local " +
        "--heads-only-pass --end-by `"2026-11-07 10:00`" --n-train 5000"
$cmd = "cmd.exe /c `"set HIP_VISIBLE_DEVICES=1&& `"$gpu`" $args >> $runs\rounds_local\console.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-run' }
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
$cmd
