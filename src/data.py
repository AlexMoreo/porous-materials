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

    return X, Y


if __name__ == '__main__':
    # path = '../data/training/dataset_for_hydrogen.csv'
    path = '../data/training/dataset_for_nitrogen.csv'
    X, Y = load_data(path, cumulate_x=True, normalize=False)