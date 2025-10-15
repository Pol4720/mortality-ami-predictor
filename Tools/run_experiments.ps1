param(
    [string]$Data = $env:DATASET_PATH
)

if (-not $Data) { throw "Please set DATASET_PATH or pass -Data path" }

$env:EXPERIMENT_TRACKER = $env:EXPERIMENT_TRACKER -as [string]
$env:TRACKING_URI = $env:TRACKING_URI -as [string]

python -m src.train --data $Data --task mortality --quick
python -m src.train --data $Data --task arrhythmia --quick
try { python -m src.train --data $Data --task regression --quick } catch { Write-Host "Regression skipped" }

python -m src.evaluate --data $Data
