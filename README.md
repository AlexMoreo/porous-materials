# Porous materials

## install

Please, make sure you have conda (or miniconda) installed. Then follow these instructions

```bash
git clone git@github.com:AlexMoreo/porous-materials.git
cd porous-materials
chmod +x setup.sh
./setup.sh
```

Then, make sure you have the pre-trained model files in your system, e.g., in ./neuralregressor. Assume the data for which you want to issue predictions is in ./data/test.csv. Then run:

```bash
python src/inference.py --saved neuralregressor/ --input ./data/test.csv --out predictions
```

The directory "predictions" should contain two files, Gout.csv and Vout.csv containing the predictions for the output gas and the volume-based representations, respectively.





