from importlib.metadata import Deprecated

import numpy as np
import pandas as pd
from pathlib import Path


def load_data(path, normalize_out=False):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    input_axis = np.asarray([i for i in range(1, 121)])
    covariate_labels = [f'l_{i}' for i in input_axis]

    first_row = df.iloc[0]
    covariates = first_row[covariate_labels].values

    output_axis = df['P'].values
    out_values = df['adsorption'].values

    # scale = 1
    scale = covariates[-1]
    if normalize_out:
        raise Deprecated('this has changed; the scale value is not the last one but the total volume absorved '
                         'which corresponds to the last covariate')
        scale = out_values[-1]
        out_values = out_values/scale

    return input_axis, covariates, output_axis, out_values, scale


def load_all_data(folder, normalize_out=False):
    files = get_file_paths(folder)
    Xs, Ys, scales = [], [], []
    input_axis_check, output_axis_check = None, None
    for file in files:
        input_axis, covariates, output_axis, out_values, scale = load_data(file, normalize_out=normalize_out)

        if input_axis_check is None:
            input_axis_check = input_axis
        else:
            assert (input_axis == input_axis_check).all(), f"file {file}'s input axis does not coincide with the rest"
        if output_axis_check is None:
            output_axis_check = output_axis
        else:
            if Path(file).name == 'l60_d0.8_vt10_mu2.08_sigma0.43.csv' and len(out_values) == 21:
                output_axis = output_axis_check
                fill_value = out_values[-1]
                extra_values = len(output_axis_check)-len(out_values)
                out_values = np.concatenate([out_values, np.full(fill_value=fill_value, shape=(extra_values,))])
                print('temp fix in file', file, 'added extra dimensions')
            assert (output_axis == output_axis_check).all(), f"file {file}'s output axis does not coincide with the rest"

        Xs.append(covariates)
        Ys.append(out_values)
        scales.append(scale)

        # print(file, len(output_axis), len(out_values))
    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)
    scales = np.asarray(scales)
    filenames = [Path(file).name for file in files]
    return Xs, Ys, input_axis, output_axis, scales, filenames


def get_file_paths(folder):
    dir = Path(folder)
    return [f for f in dir.iterdir() if f.is_file()]

