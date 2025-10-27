#!/bin/bash

# Set environment variables for model caching
export HF_HOME="${DATASET_DIR:-/iridisfs/geosets/oeg1n18/predictiondatasets}/models_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

echo "Model cache directory: $HF_HOME"

# Create cache directory if it doesn't exist
mkdir -p "$HF_HOME"

# Run the model download script
python models/download/download_all_models.py
