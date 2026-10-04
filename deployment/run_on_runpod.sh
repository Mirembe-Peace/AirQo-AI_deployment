#!/usr/bin/env bash
# Runs the full AirQo PM2.5 pipeline headless on a RunPod pod and keeps the logs.
# Usage (from the repo root, inside the pod's terminal):  bash deployment/run_on_runpod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

{
  echo "=== RunPod environment ==="
  date -u
  echo "Pod ID: ${RUNPOD_POD_ID:-unknown}"
  nvidia-smi || echo "nvidia-smi not available"
  python --version
} | tee outputs/logs/runpod_environment.txt

pip install -q -r deployment/requirements.txt 2>&1 | tail -n 5 | tee outputs/logs/runpod_pip_install.log

START=$(date +%s)
jupyter nbconvert --to notebook --execute AirQo_PM25_Pipeline.ipynb \
  --output AirQo_PM25_Pipeline_executed_runpod.ipynb \
  --ExecutePreprocessor.timeout=3600 2>&1 | tee outputs/logs/runpod_nbconvert.log
END=$(date +%s)

echo "Total wall-clock runtime: $((END - START)) s" | tee -a outputs/logs/runpod_environment.txt
echo "Run summary:"; cat outputs/logs/run_summary_runpod.json || true
echo "Pipeline log:"; tail -n 40 outputs/logs/pipeline_run.log
