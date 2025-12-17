# Porous materials

## install

Please, make sure you have conda (or miniconda) installed. Then follow these instructions for enabling inference (cpu-only):

```bash
git clone git@github.com:AlexMoreo/porous-materials.git
cd porous-materials
conda create -n porous python=3.12
conda activate porous
pip install numpy pandas scikit-learn matplotlib
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

To train a model, use:

```bash
python src/training.py --gas-input ./data/training_set_N2.csv --gas-output ./data/training_set_H2.csv --save ./model/neural_regressor

```

Then, make sure you have the pre-trained model files in your system, e.g., in ./neuralregressor. Assume the data for which you want to issue predictions is in ./data/test.csv. Then run:

```bash
python src/inference.py --saved ./model/neural_regressor --input ./data/test.csv --out predictions
```

The directory "predictions" should contain two files, Gout.csv and Vout.csv containing the predictions for the output gas and the volume-based representations, respectively.

Once the predictions have been generated, they can be plotted using:

```bash
python src/show_prediction.py --plotdir ./plots ./predictions --cumulate
```






