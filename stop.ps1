$ErrorActionPreference = 'Stop'
$voiceRoot = $PSScriptRoot
$voiceStateFile = Join-Path $voiceRoot 'runtime\server.json'
if (-not (Test-Path -LiteralPath $voiceStateFile)) {
    Write-Host 'The app is not running.'
    exit 0
}
$voiceState = Get-Content -LiteralPath $voiceStateFile -Raw | ConvertFrom-Json
$voiceInfo = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $voiceState.pid) -ErrorAction SilentlyContinue
$voicePython = Join-Path $voiceRoot '.venv\Scripts\python.exe'
$voiceApp = Join-Path $voiceRoot 'app\GUI.py'
if ($voiceInfo -and ($voiceInfo.ExecutablePath -ieq $voicePython) -and $voiceInfo.CommandLine -and ($voiceInfo.CommandLine.IndexOf($voiceApp, [StringComparison]::OrdinalIgnoreCase) -ge 0)) {
    # Windows virtualenv's python.exe launches a child interpreter.
    $voiceWorkers = Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $voiceState.pid) | Where-Object { $_.Name -ieq 'python.exe' -and $_.CommandLine -and $_.CommandLine.IndexOf($voiceApp, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
    foreach ($voiceWorker in $voiceWorkers) {
        $voiceRecord = Join-Path $voiceRoot 'app\record.py'
        $voiceRecorders = Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $voiceWorker.ProcessId) | Where-Object { $_.Name -ieq 'python.exe' -and $_.CommandLine -and $_.CommandLine.IndexOf($voiceRecord, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
        foreach ($voiceRecorder in $voiceRecorders) {
            $voiceRecorderWorkers = Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $voiceRecorder.ProcessId) | Where-Object { $_.Name -ieq 'python.exe' -and $_.CommandLine -and $_.CommandLine.IndexOf($voiceRecord, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
            foreach ($voiceRecorderWorker in $voiceRecorderWorkers) { Stop-Process -Id $voiceRecorderWorker.ProcessId -ErrorAction SilentlyContinue }
            Stop-Process -Id $voiceRecorder.ProcessId -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $voiceWorker.ProcessId -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $voiceState.pid -ErrorAction SilentlyContinue
    Write-Host 'The speech app has stopped.'
} elseif ($voiceInfo) {
    throw 'The saved process belongs to another application; it was left running.'
} else {
    Write-Host 'The app is already stopped.'
}
