import os.path
import pickle
import warnings
from collections import defaultdict

import numpy as np
from PIL.ImageOps import scale
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from data import *
from regression import StackRegressor, NeuralRegressor, PrecomputedBaseline
from nn_modules import FFModel, MonotonicNN, LSTMModel, TransformerRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from result_table.src.table import Table
from utils import *
from pathlib import Path

gas='nitrogen'
# gas='hydrogen'

# problem = 'direct'
problem = 'inverse'
assert problem in ['direct', 'inverse'], 'wrong problem'

components=8

path = f'../data/training/dataset_for_{gas}.csv'
X, y = load_data(path, cumulate_x=True, normalize=True)

if problem == 'inverse':
    X, y = y, X

standardize_output = True

loo = LeaveOneOut()
for i, (train, test) in enumerate(loo.split(X, y)):
    test_name = f'model{i+1}'
    print(f'{test_name}:')

    ytr = y[train]
    yte = y[test]

    yte_ = yte.copy()
    if standardize_output:
        zscorer = StandardScaler()
        ytr = zscorer.fit_transform(ytr)
        yte_ = zscorer.transform(yte_)

    pca = PCA(n_components=components)
    ytr_ = pca.fit_transform(ytr)
    yte_ = pca.transform(yte_)

    yte_ = pca.inverse_transform(yte_)
    if standardize_output:
        yte_ = zscorer.inverse_transform(yte_)

    suffix="-zscore" if standardize_output else ""
    plot_result(yte[0], yte_[0], f'../reconstruction/{gas}/PCA{components}{suffix}/{str(test_name)}.png', err_fun=mse)
