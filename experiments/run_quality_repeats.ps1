# Run only the two frozen follow-up experiments, one GPU task at a time.
# Re-entering this script verifies/reuses completed work and resumes last.pt.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$roundDirectory = Join-Path $projectRoot 'artifacts\quality_repeat_2026_10_20'
$lockPath = Join-Path $roundDirectory 'execution.lock'
if (-not (Test-Path -LiteralPath $roundDirectory -PathType Container)) {
    throw 'Prepare and freeze both seed-specific plans before execution.'
}
foreach ($seed in @(43, 44)) {
    $planPath = Join-Path $projectRoot "artifacts\quality_seed$seed\plan.json"
    if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) { throw "Missing frozen plan: $planPath" }
}
$runLock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$originalLocation = Get-Location
$previousUnbuffered = $env:PYTHONUNBUFFERED
try {
    Set-Location -LiteralPath $projectRoot
    $env:PYTHONUNBUFFERED = '1'
    foreach ($seed in @(43, 44)) {
        Write-Output "Verifying/running quality seed $seed; see per-seed progress and run_seed$seed.log."
        $hardwarePath = Join-Path $roundDirectory "hardware_seed$seed.csv"
        if (-not (Test-Path -LiteralPath $hardwarePath)) {
            & nvidia-smi --query-gpu=timestamp,name,driver_version,clocks.gr,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv | Out-File -LiteralPath $hardwarePath -Encoding utf8
        }
        & (Join-Path $projectRoot 'project-python.ps1') -m mt.run_quality --seed $seed *>> (Join-Path $roundDirectory "run_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "Seed $seed failed (exit $LASTEXITCODE); inspect its log before resuming." }
        Write-Output "Quality seed $seed completed and verified."
    }
}
finally {
    $env:PYTHONUNBUFFERED = $previousUnbuffered
    Set-Location -LiteralPath $originalLocation
    $runLock.Dispose()
}
