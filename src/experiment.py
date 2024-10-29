import os.path
import pickle
from collections import defaultdict
from ossaudiodev import error

import numpy as np
from PIL.ImageOps import scale
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import LeaveOneOut
from torch.backends.mkl import verbose

from data import *
from methods import StackRegressor, NeuralRegressor, TheirBaseline, LSTMRegressor, FFModel
from sklearn.ensemble import RandomForestRegressor

from result_table.src.table import Table
from utils import *
from pathlib import Path


X, y, in_axis, out_axis, scales, filenames = load_all_data('../data/training_set', normalize_out=False)

standardize_input = False
no_accum_y = False
ignore_last_column_X = True


if ignore_last_column_X:
    X = X[:, :-1]
    in_axis = in_axis[:-1]


def remove_accum(y):
    y = np.copy(y)
    y[:, 1:] -= y[:,:-1]
    return y


def methods():
    input_size = X.shape[1]
    output_size = y.shape[1]
    yield 'their', TheirBaseline(filenames[test[0]], scale=test_scale)
    # yield 'SVR', MultiOutputRegressor(LinearSVR())
    # yield 'StackSVR', StackRegressor()
    #yield 'lstm-32-1', LSTMregressor(hidden_size=32, num_layers=1)
    yield 'lstm-256-4', LSTMRegressor(input_size, output_size, hidden_size=256, num_layers=4, bidirectional=True)
    yield 'ff-128-256', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256]))
    # method, reg = 'lstm-256-4', LSTMregressor(hidden_size=256, num_layers=4)
    yield 'RF', RandomForestRegressor()

table = Table('mse')
table_path = '../results/tables/mse.pdf'
table.format.show_std = False
table.format.mean_prec = 1
table.format.stat_test = None
ansia = True

errors = defaultdict(lambda :[])
test_names = []

loo = LeaveOneOut()
for i, (train, test) in enumerate(loo.split(X, y)):
    Xtr, ytr = X[train], y[train]
    Xte, yte = X[test], y[test]
    test_name = filenames[test[0]]
    test_name = test_name.replace('_', ', ')
    test_name = test_name.replace('mu', '$\mu$')
    test_name = test_name.replace('sigma', '$\sigma$')
    test_name = test_name.replace('.csv', '')

    if standardize_input:
        zscorer = StandardScaler()
        Xtr = zscorer.fit_transform(Xtr)
        Xte = zscorer.transform(Xte)

    if no_accum_y:
        ytr = remove_accum(ytr)
        yte = remove_accum(yte)

    test_scale = scales[test[0]]

    for method, reg in methods():
        show_table = False
        method_results_path = f'../results/errors/{method}.pkl'
        if os.path.exists(method_results_path):
            method_errors = pickle.load(open(method_results_path, 'rb'))
        else:
            os.makedirs(Path(method_results_path).parent, exist_ok=True)
            method_errors = {}

        if test_name not in method_errors:
            reg.fit(Xtr, ytr)
            yte_pred = reg.predict(Xte)

            if no_accum_y:
                yte = np.cumsum(yte, axis=1)
                yte_pred = np.cumsum(yte_pred, axis=1)

            yte_pred *= test_scale
            yte *= test_scale

            plot_result(out_axis, yte[0], yte_pred[0], f'../results/plots/{method}/{filenames[test[0]]}.png', err_fun=mse)

            error_mean = mse(yte, yte_pred)
            method_errors[test_name] = error_mean
            if ansia:
                show_table = True
        else:
            error_mean = method_errors[test_name]

        errors[method].append(error_mean)

        pickle.dump(method_errors, open(method_results_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        table.add(benchmark=test_name, method=method, v=error_mean)
        if show_table:
            table.latexPDF(table_path, resizebox=True)

    filenames.append(test_name)
    print(test_name)

table.latexPDF(table_path, benchmark_order=sorted(table.benchmarks), verbose=False, resizebox=False)
for method, errors_means in errors.items():
    print(f'{method}\tMSE={np.mean(errors_means):.10f}+-{np.std(errors_means):.10f}')