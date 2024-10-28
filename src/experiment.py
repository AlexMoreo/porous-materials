import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import LeaveOneOut
from data import *
from methods import StackRegressor, LSTMregressor
from utils import *
from pathlib import Path

X, y, in_axis, out_axis, scale, filenames = load_all_data('../data/training_set', normalize_out=True)

standardize = False
no_accum_y = False

def remove_accum(y):
    y = np.copy(y)
    y[:, 1:] -= y[:,:-1]
    return y


loo = LeaveOneOut()
errors = []
for i, (train, test) in enumerate(loo.split(X, y)):
    Xtr, ytr = X[train], y[train]
    Xte, yte = X[test], y[test]

    if standardize:
        zscorer = StandardScaler()
        Xtr = zscorer.fit_transform(Xtr)
        Xte = zscorer.transform(Xte)

    if no_accum_y:
        ytr = remove_accum(ytr)
        yte = remove_accum(yte)

    # method, reg = 'SVR', MultiOutputRegressor(LinearSVR())
    # method, reg = 'StackSVR', StackRegressor()
    method, reg = 'lstm', LSTMregressor(hidden_size=256, num_layers=3)
    reg.fit(Xtr, ytr)

    yte_pred = reg.predict(Xte)

    if no_accum_y:
        yte = np.cumsum(yte, axis=1)
        yte_pred = np.cumsum(yte_pred, axis=1)

    yte_pred*=scale
    yte*=scale

    plot_result(out_axis, yte[0], yte_pred[0], f'../results/plots/{method}/{filenames[test[0]]}.png', err_fun=mse)

    errors.append(mse(yte, yte_pred))

print(f'{method}\tMSE={np.mean(errors):.3f}+-{np.std(errors):.3f}')