# Samples the inference server's CPU seconds every 5 s until $Stop exists.
# One line per sample: wall clock, pid, CPU seconds, seconds since the
# process started, and the share of one core over its life so far.
param([string]$Out, [string]$Stop)
while (-not (Test-Path $Stop)) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*eval_inference_server*' }
    foreach ($p in $procs) {
        try {
            $g = Get-Process -Id $p.ProcessId -ErrorAction Stop
            $cpu = $g.TotalProcessorTime.TotalSeconds
            $age = ((Get-Date) - $g.StartTime).TotalSeconds
            $share = if ($age -gt 0) { $cpu / $age } else { 0 }
            "{0} {1} {2:F1} {3:F1} {4:F3}" -f (Get-Date -Format HH:mm:ss), $p.ProcessId, $cpu, $age, $share |
                Out-File -FilePath $Out -Append -Encoding utf8
        } catch {}
    }
    Start-Sleep -Seconds 5
}
