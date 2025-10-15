#!/usr/bin/env bash
set -euo pipefail

# Usage: Ensure DATASET_PATH env var is set
: "${DATASET_PATH:?Must set DATASET_PATH to dataset file path}"

export EXPERIMENT_TRACKER=${EXPERIMENT_TRACKER:-mlflow}
export TRACKING_URI=${TRACKING_URI:-}

python -m src.train --data "$DATASET_PATH" --task mortality --quick
python -m src.train --data "$DATASET_PATH" --task arrhythmia --quick
# Optional regression target if available
python -m src.train --data "$DATASET_PATH" --task regression --quick || echo "Regression task skipped (no regression target)"

python -m src.evaluate --data "$DATASET_PATH"
