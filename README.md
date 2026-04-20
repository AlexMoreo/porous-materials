# Porous Materials Regressors

This repository contains the reference implementation of the neural regressors used to predict adsorption isotherm curves for porous materials.

The code is organized around a small public-facing workflow:

- `src/training.py`: train the released regressors when paired training data are available
- `src/inference.py`: run inference from a saved model directory
- `src/show_prediction.py`: visualize predicted curves stored on disk

Some datasets and experiment artifacts in this repository are research assets rather than part of the public API. The core implementation intended for reuse is the code in `src/`. 

## What The Model Produces

Given an input adsorption curve, the inference pipeline outputs:

- `Gout.csv`: predicted adsorption curve for the target gas
- `Vout.csv`: predicted volume-based representation

The default inference pipeline combines four trained regressors:

- `RegressorA_pca`
- `RegressorA_raw`
- `RegressorB_pca`
- `RegressorB_raw`

## Installation

The codebase is lightweight and can be run with a standard Python environment. One simple CPU-only setup is:

```bash
git clone https://github.com/AlexMoreo/porous-materials.git
cd porous-materials
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib joblib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you prefer Conda, the equivalent setup is:

```bash
conda create -n porous python=3.12
conda activate porous
pip install numpy pandas scikit-learn matplotlib joblib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Input Format

The scripts expect CSV files with specific column names.

For inference input, the file must contain:

- `Sample`: sample identifier
- `Total volume`: normalization factor used internally by the model
- `adsor1`, `adsor2`, ..., `adsorN`: input adsorption curve values

For training, the paired gas files are expected to contain:

- `Sample`
- `Total volume`
- `feature1`, `feature2`, ..., `featureN`
- `adsor1`, `adsor2`, ..., `adsorN`

Missing values in training targets are represented internally with `-1`.

## Inference

Assume:

- pretrained models are stored in `./neuralregressor`
- the input file is `./data/test.csv`
- predictions should be written to `./predictions`

Run:

```bash
python3 src/inference.py --saved ./neuralregressor --input ./data/test.csv --out ./predictions
```

This creates:

- `./predictions/Gout.csv`
- `./predictions/Vout.csv`

To visualize the predicted curves:

```bash
python3 src/show_prediction.py ./predictions
```

To save the plots instead of opening them interactively:

```bash
python3 src/show_prediction.py ./predictions --plotdir ./plots
```

## Training

Training requires paired input/output gas datasets, which may not be publicly distributed with this repository.

When these files are available, training can be launched with:

```bash
python3 src/training.py \
  --gas-input ./path/to/input_gas.csv \
  --gas-output ./path/to/output_gas.csv \
  --save ./neuralregressor
```

This stores:

- model checkpoints
- dimensionality-reduction adapters
- a small parameter file used by inference
- an uncertainty model for flagging out-of-distribution inputs

## Repository Scope

This repository is being released primarily to make the regressor implementation explicit and inspectable. It should be read as a reference implementation of the modeling pipeline rather than as a polished benchmark suite.
