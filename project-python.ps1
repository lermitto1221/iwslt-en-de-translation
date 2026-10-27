# Use the venv's original Python and installed packages without its Windows
# redirector. -S excludes unrelated packages from the base installation.
# Example: .\project-python.ps1 -m mt --json
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvConfig = Join-Path $projectRoot '.venv\pyvenv.cfg'
$homeLine = Get-Content -LiteralPath $venvConfig | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
if (-not $homeLine) { throw 'Missing home entry in .venv/pyvenv.cfg' }
$basePythonDirectory = ($homeLine -split '=', 2)[1].Trim()
$basePython = Join-Path $basePythonDirectory 'python.exe'
$projectPackages = Join-Path $projectRoot '.venv\Lib\site-packages'
if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) { throw "Python not found: $basePython" }
if (-not (Test-Path -LiteralPath $projectPackages -PathType Container)) { throw 'Project dependencies are missing' }
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $projectRoot + [IO.Path]::PathSeparator + $projectPackages
    if ($MyInvocation.ExpectingInput) {
        $input | & $basePython -S @args
    }
    else {
        & $basePython -S @args
    }
    $projectExitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
exit $projectExitCode
