#!/bin/bash

# setup.sh — create and prepare a Python virtual environment
# Usage: bash setup.sh

set -e  # exit on any error

ENV_DIR="env"

# Optional: clean setup (ask before deleting existing environment)
if [ -d "$ENV_DIR" ]; then
    read -p "The environment '$ENV_DIR' already exists. Recreate it? [y/N] " yn
    case $yn in
        [Yy]* ) rm -rf "$ENV_DIR";;
        * ) echo "Using existing environment."; exit 0;;
    esac
fi

echo "Creating virtual environment in: $ENV_DIR ..."
python3 -m venv "$ENV_DIR"

echo "Activating virtual environment..."
source "$ENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready!"
echo
echo "To activate it later, run:"
echo "source $ENV_DIR/bin/activate"
echo
echo "Then you can run your script, for example:"
echo "python your_script.py"
