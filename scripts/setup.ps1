param(
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [string]$IndexUrl = 'https://pypi.org/simple'
)
$ErrorActionPreference = 'Stop'
$voiceRoot = Split-Path -Path $PSScriptRoot -Parent
$voicePython = Join-Path $voiceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $voicePython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv (Join-Path $voiceRoot '.venv')
    } else {
        & python -m venv (Join-Path $voiceRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
}
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path $voiceRoot '.cache\pip'
$env:TEMP = Join-Path $voiceRoot 'runtime'
$env:TMP = $env:TEMP
foreach ($voiceDirectory in @('runtime', 'logs', 'private', '.cache')) {
    New-Item -ItemType Directory -Path (Join-Path $voiceRoot $voiceDirectory) -Force | Out-Null
}
$voiceTorchIndex = 'https://download.pytorch.org/whl/cpu'
if ($Device -eq 'cuda') { $voiceTorchIndex = 'https://download.pytorch.org/whl/cu124' }
& $voicePython -m pip install 'torch==2.5.1' 'torchaudio==2.5.1' --index-url $voiceTorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $voicePython -m pip install -r (Join-Path $voiceRoot 'requirements.txt') --index-url $IndexUrl
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $voicePython (Join-Path $PSScriptRoot 'prepare_models.py')
if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
Write-Host 'Setup complete. Double-click the start launcher in the project folder.'
