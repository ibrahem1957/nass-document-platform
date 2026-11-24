# Runs the three component test suites from their own folders (each suite uses its own import root).
# Usage (repository root, root virtual environment active):  .\scripts\run_tests.ps1
# No test contacts Qari, RunPod, or any network service. Expected baseline: see docs/TESTING.md.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
$results = @()
foreach ($suite in @(
    @{ Name = "DocPrep"; Dir = "docprep"; Args = @("-m", "pytest", "tests/docprep", "-q", "-p", "no:cacheprovider", "-rs") },
    @{ Name = "Studio";  Dir = "studio";  Args = @("-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs") },
    @{ Name = "Nass";    Dir = "nass";    Args = @("-m", "pytest", "tests/lab", "-q", "-p", "no:cacheprovider", "-rfs") }
)) {
    Write-Host "=== $($suite.Name) ===" -ForegroundColor Cyan
    Push-Location (Join-Path $root $suite.Dir)
    & python -X utf8 @($suite.Args)
    $results += "$($suite.Name): exit $LASTEXITCODE"
    Pop-Location
}
Write-Host ""
$results | ForEach-Object { Write-Host $_ }
Write-Host "Nass is expected to exit 1 with the 19 pre-existing failures listed in docs/TESTING.md."
