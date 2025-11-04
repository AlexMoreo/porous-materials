#!/bin/bash

# Exit immediately on error
set -e

ENV_NAME="porous"
PYTHON_VERSION="3.12"

echo "=== Conda environment setup ==="

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "Error: Conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment if it doesn’t already exist
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Updating dependencies..."
else
    echo "Creating new conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
fi

# Activate the environment
echo "Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

echo "=== Setup complete! ==="
echo "To activate the environment later, run:"
echo "conda activate $ENV_NAME"
