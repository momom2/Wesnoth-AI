param([Parameter(Mandatory=$true)][string]$Head, [Parameter(Mandatory=$true)][int]$K0,
      [Parameter(Mandatory=$true)][ValidateSet('heads_only', 'reduced_adamw')][string]$Arm,
      [string]$EndBy = '2026-11-08 02:00', [int]$Held = 0,
      [string]$FinalOut = 'final_match_after_stop', [int]$Average = 0)
# Prereg departure 33: the rounds go on past round K0's design stop, rebased on it.
# -FinalOut (departure 35): a later stop's re-arm names a fresh final-match directory.
$py = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai\.venv\Scripts\python.exe'
$gpu = 'C:\Users\momom2\venvs\wesnoth_rocm\Scripts\python.exe'
$runs = 'C:\Users\momom2\wesnoth_runs'
$trunk = ''
if ($Arm -eq 'reduced_adamw') { $trunk = ' --rounds-arm reduced_adamw --rounds-pass-window 10000000' }
if ($Held -gt 0) { $trunk += " --rounds-held $Held" }      # prereg departure 34
if ($Average -gt 0) { $trunk += " --rounds-average $Average" }   # prereg departure 37
$args = "-u scripts\chain_local.py --stage-c $runs\stage_c_local --stage1 $runs\selfplay_t1_local " +
        "--p-out $runs\p_local --rounds-out $runs\rounds_local --e 0.02 --rounds 39 " +
        "--checkpoint training\checkpoints\own2.pt " +
        "--checkpoint-sha256 aea027b2696a9fbce0ebcdc0a8483c9df7d35957934b81b0c62093a02cdd9129 " +
        "--expect-head $Head --gpu-python $gpu --tools-python $py --final-out $runs\$FinalOut " +
        "--heads-only-pass --rounds-n-train 5000 --rounds-end-by `"$EndBy`" --skip-p --after-stop $K0" + $trunk
$cmd = "cmd.exe /c `"`"$py`" $args >> $runs\chain_after_stop.console.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = 'C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-run' }
"rc=$($r.ReturnValue) pid=$($r.ProcessId)"
$cmd
