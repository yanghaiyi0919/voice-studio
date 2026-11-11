param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$voiceRoot = $PSScriptRoot
$voicePython = Join-Path $voiceRoot '.venv\Scripts\python.exe'
$voiceApp = Join-Path $voiceRoot 'app\GUI.py'
$voiceRuntime = Join-Path $voiceRoot 'runtime'
$voiceStateFile = Join-Path $voiceRuntime 'server.json'
$voiceOutLog = Join-Path $voiceRoot 'logs\server.log'
$voiceErrLog = Join-Path $voiceRoot 'logs\server.err.log'
foreach ($voiceDirectory in @('logs', 'runtime', 'private')) {
    New-Item -ItemType Directory -Path (Join-Path $voiceRoot $voiceDirectory) -Force | Out-Null
}

function Test-VoiceReady([string]$VoiceUrl) {
    try {
        $voiceResponse = Invoke-WebRequest -Uri ($VoiceUrl + '/config') -UseBasicParsing -TimeoutSec 2
        return ($voiceResponse.StatusCode -eq 200)
    } catch { return $false }
}

function Test-VoiceProcess([int]$VoiceProcessId) {
    $voiceInfo = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $VoiceProcessId) -ErrorAction SilentlyContinue
    if (-not $voiceInfo) { return $false }
    return (($voiceInfo.ExecutablePath -ieq $voicePython) -and ($voiceInfo.CommandLine.IndexOf($voiceApp, [StringComparison]::OrdinalIgnoreCase) -ge 0))
}

if (-not (Test-Path -LiteralPath $voicePython)) { throw 'Python environment is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $voiceRuntime 'models.json'))) { throw 'Models are not ready. Run scripts\prepare_models.py first.' }
$voiceRunning = $false
if (Test-Path -LiteralPath $voiceStateFile) {
    $voiceState = Get-Content -LiteralPath $voiceStateFile -Raw | ConvertFrom-Json
    if (Test-VoiceProcess $voiceState.pid) {
        $voiceRunning = $true
        $voiceUrl = $voiceState.url
        $voiceServerId = $voiceState.pid
    }
}

if (-not $voiceRunning) {
    $voicePort = $null
    foreach ($voiceCandidate in 7860..7879) {
        $voiceProbe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $voiceCandidate)
        try {
            $voiceProbe.Start()
            $voicePort = $voiceCandidate
            break
        } catch {
        } finally { $voiceProbe.Stop() }
    }
    if ($null -eq $voicePort) { throw 'No free local port was found between 7860 and 7879.' }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:GRADIO_ANALYTICS_ENABLED = 'False'
    $env:VOICE_PORT = [string]$voicePort
    $env:MODELSCOPE_CACHE = Join-Path $voiceRoot '.cache\modelscope'
    $env:HF_HOME = Join-Path $voiceRoot '.cache\huggingface'
    $env:TORCH_HOME = Join-Path $voiceRoot '.cache\torch'
    $env:GRADIO_TEMP_DIR = Join-Path $voiceRoot 'runtime\gradio'
    $env:PATH = (Join-Path $voiceRoot 'tools') + [System.IO.Path]::PathSeparator + $env:PATH
    $voiceProcess = Start-Process -FilePath $voicePython -ArgumentList @('-u', ('"' + $voiceApp + '"')) -WorkingDirectory (Join-Path $voiceRoot 'app') -WindowStyle Hidden -RedirectStandardOutput $voiceOutLog -RedirectStandardError $voiceErrLog -PassThru
    $voiceServerId = $voiceProcess.Id
    $voiceUrl = 'http://127.0.0.1:' + $voicePort
    @{ pid = $voiceServerId; port = $voicePort; url = $voiceUrl; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $voiceStateFile -Encoding UTF8
}

Write-Host 'Loading the speech models. The first start can take a little longer...'
$voiceDeadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $voiceDeadline) {
    if (Test-VoiceReady $voiceUrl) {
        Write-Host ('Ready: ' + $voiceUrl)
        if (-not $NoBrowser) { Start-Process $voiceUrl }
        exit 0
    }
    if (-not (Test-VoiceProcess $voiceServerId)) {
        Write-Host 'The app stopped during startup. Recent error details:'
        if (Test-Path -LiteralPath $voiceErrLog) { Get-Content -LiteralPath $voiceErrLog -Tail 35 -Encoding UTF8 }
        throw 'Startup failed. See logs\server.err.log.'
    }
    Start-Sleep -Seconds 2
}
throw ('The server is still loading. Check ' + $voiceOutLog + ' and ' + $voiceErrLog)
