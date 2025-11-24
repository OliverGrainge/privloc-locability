#!/bin/bash

# Usage: ./submit_python.sh script.py [args...]
# Example: ./submit_python.sh train.py --epochs 100 --lr 0.001

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script> [arguments...]"
    exit 1
fi

PYTHON_SCRIPT="$1"
shift  # Remove the script name from arguments
ARGS="$@"

# Set your default resources here (based on your typical setup)
NODES=1
NTASKS=1
GPUS=2
CPUS=12
TIME="1:00:00"

PARTITION="quad_h200"

# Generate job name from script name with timestamp and optional config info
BASE_NAME=$(basename "$PYTHON_SCRIPT" .py)

# Try to extract meaningful info from arguments
CONFIG_INFO=""
for arg in $ARGS; do
    if [[ "$arg" == *.yaml ]] || [[ "$arg" == *.yml ]] || [[ "$arg" == *.json ]]; then
        # Extract config filename without path and extension
        CONFIG_NAME=$(basename "$arg" | sed 's/\.[^.]*$//')
        CONFIG_INFO="_${CONFIG_NAME}"
        break
    fi
done

# Add timestamp for uniqueness
TIMESTAMP=$(date +%m%d_%H%M)
JOB_NAME="${BASE_NAME}${CONFIG_INFO}_${TIMESTAMP}"

# Ensure logs directory exists
mkdir -p logs

# Submit the job
sbatch \
    --job-name="$JOB_NAME" \
    --nodes="$NODES" \
    --ntasks="$NTASKS" \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS" \
    --time="$TIME" \
    --partition="$PARTITION" \
    --output="logs/${JOB_NAME}_%j.out" \
    --error="logs/${JOB_NAME}_%j.err" \
    --wrap="python $PYTHON_SCRIPT $ARGS"

echo "Submitted job: $JOB_NAME"
echo "Resources: $GPUS GPU(s), $CPUS CPUs, $TIME time limit, partition: $PARTITION"
echo "Command: python $PYTHON_SCRIPT $ARGS"
echo "Logs: logs/${JOB_NAME}_<job_id>.out, logs/${JOB_NAME}_<job_id>.err"