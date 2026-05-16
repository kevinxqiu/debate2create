#!/bin/bash
# Run LASeR baseline across all environments and seeds.
# Usage: bash baselines/run_laser_all.sh
#
# Prerequisites:
#   - OPENAI_API_KEY must be set for OpenAI-backed runs
#   - dependencies must be installed in the active Python environment

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
LASER="${REPO_ROOT}/baselines/laser_baseline.py"

ENVS="swimmer ant half_cheetah hopper walker2d"
SEEDS="0 1 2"

for env in $ENVS; do
  for seed in $SEEDS; do
    log_dir="${REPO_ROOT}/outputs/baselines/laser/${env}/seed_${seed}"
    mkdir -p "$log_dir"
    echo "========================================="
    echo "Running LASeR: env=$env seed=$seed"
    echo "========================================="
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
      $PYTHON "$LASER" --env "$env" --seed "$seed" --output-dir "$log_dir" \
      2>&1 | tee "${log_dir}/run.log"
    echo ""
  done
done

echo "All LASeR runs complete."
