# Isolated Marian dependencies; the original project Torch/CUDA and NumPy stay in use.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$homeLine = Get-Content -LiteralPath (Join-Path $projectRoot '.venv\pyvenv.cfg') | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
$basePython = Join-Path (($homeLine -split '=', 2)[1].Trim()) 'python.exe'
$projectPackages = Join-Path $projectRoot '.venv\Lib\site-packages'
$pretrainedPackages = Join-Path $projectRoot '.analysis-deps\pretrained'
if (-not (Test-Path -LiteralPath $pretrainedPackages -PathType Container)) { throw 'Install the pinned pretrained dependencies first' }
$previousPythonPath = $env:PYTHONPATH
$previousPythonIoEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = $projectRoot + [IO.Path]::PathSeparator + $pretrainedPackages + [IO.Path]::PathSeparator + $projectPackages
    $env:PYTHONIOENCODING = 'utf-8'
    if ($MyInvocation.ExpectingInput) { $input | & $basePython -S @args }
    else { & $basePython -S @args }
    $projectExitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
}
exit $projectExitCode
