import numpy as np
import pandas as pd
from pathlib import Path


def load_data(path, cumulate_x=False, normalize=False):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # extract relevant features
    feature_cols = df.filter(regex=r'^feature\d+$').columns
    adsor_cols = df.filter(regex=r'^adsor\d+$').columns

    X = df[feature_cols]
    Y = df[adsor_cols]
    total_vol = df['Total volume']

    n, Xcol = X.shape
    X = X.loc[:, (X != 0).any()]
    XcolNonZero = X.shape[1]
    Ycol = Y.shape[1]
    print(f'loaded file {path}: found {n} instances,'
          f' input has {Xcol} features (retaining {XcolNonZero} non-zero), '
          f'output has {Ycol} dimensions')

    X = X.values
    Y = Y.values
    total_vol = total_vol.values

    if cumulate_x:
        X = np.cumsum(X, axis=1)

    if normalize:
        X /= total_vol[:, np.newaxis]
        Y /= total_vol[:, np.newaxis]

    # print
    # if cumulate_x:
    #     Xcum = X[:,-1]
    # else:
    #     Xcum = X.sum(axis=1)
    # for xcum, total, adsor in zip(Xcum, total_vol, Y[:,-1]):
    #     print(xcum, total, adsor)

    return X, Y



def load_data__depr(path, normalize_out=False):
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
        raise ValueError('this has changed; the scale value is not the last one but the total volume absorved '
                         'which corresponds to the last covariate')
        scale = out_values[-1]
        out_values = out_values/scale

    return input_axis, covariates, output_axis, out_values, scale


def load_all_data__depr(folder, normalize_out=False):
    files = get_file_paths(folder)
    Xs, Ys, scales = [], [], []
    input_axis_check, output_axis_check = None, None
    for file in files:
        input_axis, covariates, output_axis, out_values, scale = load_data__depr(file, normalize_out=normalize_out)

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



if __name__ == '__main__':
    # path = '../data/training/dataset_for_hydrogen.csv'
    path = '../data/training/dataset_for_nitrogen.csv'
    X, Y = load_data(path, cumulate_x=True, normalize=False)