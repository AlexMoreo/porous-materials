import numpy as np
import pandas as pd
from pathlib import Path


def load_data(path, normalize_out=False):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    input_axis = [i for i in range(1, 121)]
    covariate_labels = [f'l_{i}' for i in input_axis]

    first_row = df.iloc[0]
    covariates = first_row[covariate_labels].values

    output_axis = df['P'].values
    out_values = df['adsorption'].values

    scale=1
    if normalize_out:
        scale = out_values[-1]
        out_values = out_values/scale

    return input_axis, covariates, output_axis, out_values, scale


def load_all_data(folder, normalize_out=False):
    files = get_file_paths(folder)
    Xs, Ys = [], []
    for file in files:
        input_axis, covariates, output_axis, out_values, scale = load_data(file, normalize_out=normalize_out)
        Xs.append(covariates)
        Ys.append(out_values)
        # print(input_axis)
    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)
    filenames = [Path(file).name for file in files]
    return Xs, Ys, input_axis, output_axis, scale, filenames


def get_file_paths(folder):
    dir = Path(folder)
    return [f for f in dir.iterdir() if f.is_file()]

