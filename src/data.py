import numpy as np
import pandas as pd


def load_both_data(path_input_gas, path_output_gas, cumulate_vol=False, normalize=False, return_index=False, exclude_id=None):
    idxin, Xin, Yin = load_data(path_input_gas, cumulate_x=cumulate_vol, normalize=normalize, return_index=True, exclude_id=exclude_id)
    idxout, Xout, Yout  = load_data(path_output_gas, cumulate_x=cumulate_vol, normalize=normalize, return_index=True, exclude_id=exclude_id)
    assert all(idxin==idxout), f'index mismatch in files {path_input_gas} and {path_output_gas}'
    return (idxin, Xin, Yin, Yout) if return_index else (Xin, Yin, Yout)


def load_data(path, cumulate_x=False, normalize=False, return_index=False, exclude_id=None):
    df = pd.read_csv(path)

    if exclude_id is not None:
        df = df[~df['Sample'].isin(exclude_id)]

    df.columns = df.columns.str.strip()

    # extract relevant features
    feature_cols = df.filter(regex=r'^feature\d+$').columns
    adsor_cols = df.filter(regex=r'^adsor\d+$').columns
    idx = df['Sample'].values

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

    if return_index:
        return idx, X, Y
    else:
        return X, Y


def load_test_data(path, normalize=False, return_index=False):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # extract relevant features
    adsor_cols = df.filter(regex=r'^adsor\d+$').columns
    idx = df['Sample'].values

    Y = df[adsor_cols]
    total_vol = df['Total volume']

    n, Ycol = Y.shape
    print(f'loaded file {path}: found {n} instances of {Ycol} dimensions')

    Y = Y.values
    total_vol = total_vol.values

    if normalize:
        Y /= total_vol[:, np.newaxis]

    if return_index:
        return idx, Y, total_vol
    else:
        return Y, total_vol


if __name__ == '__main__':
    path_h2 = '../data/training/dataset_for_hydrogen.csv'
    path_n2 = '../data/training/dataset_for_nitrogen.csv'
    X, Y = load_data(path_h2, cumulate_x=True, normalize=False, exclude_id=['model41', 'model45'])
    # Xin, Yin, Yout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)
    print('[done]')