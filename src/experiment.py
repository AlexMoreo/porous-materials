import os.path
import pickle
from collections import defaultdict

import numpy as np
from PIL.ImageOps import scale
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import LeaveOneOut

from data import *
from regression import StackRegressor, NeuralRegressor, PrecomputedBaseline
from nn_modules import FFModel, MonotonicNN, LSTMModel, TransformerRegressor
from sklearn.ensemble import RandomForestRegressor

from result_table.src.table import Table
from utils import *
from pathlib import Path

gas='nitrogen'
# gas='hydrogen'

path = f'../data/training/dataset_for_{gas}.csv'
y, X = load_data(path, cumulate_x=True, normalize=True)

def methods():
    input_size = X.shape[1]
    output_size = y.shape[1]
    # yield 'SVR', MultiOutputRegressor(LinearSVR())
    # yield 'StackSVR', StackRegressor()
    # yield 'lstm-256-4', NeuralRegressor(model=LSTMModel(input_size, output_size))
    # yield 'ff-128-256', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256]))
    # yield 'ff-128-256-128-r02', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256,128]), reg_strength=0.1)
    # yield 'ff-256-256-256-128-128-r01', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[256,256,256,128,128]), reg_strength=0.1)
    # yield 'ff-128-128-r01', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,128], smooth_length=3), reg_strength=0.1)
    # yield 'ff-128-128-r01-mono', NeuralRegressor(MonotonicNN(input_size, output_size, hidden_sizes=[128, 128], smooth_length=0), reg_strength=0.01, lr=0.001)
    # yield 'ff-128-256-128-r01', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256,128], smooth_length=3), reg_strength=0.1)
    yield 'transformer-fino', NeuralRegressor(TransformerRegressor(input_dim=1, output_dim=output_size, seq_len=input_size), clip=True, reg_strength=0, lr=0.001)

 # method, reg = 'lstm-256-4', LSTMregressor(hidden_size=256, num_layers=4)
    #yield 'RF', RandomForestRegressor()

table = Table('mse')
table_path = f'../results/tables/mse_{gas}.pdf'
table.format.show_std = False
table.format.mean_prec = 1
table.format.stat_test = None
ansia = True

errors = defaultdict(lambda :[])
test_names = []

standardize_input = False

loo = LeaveOneOut()
for i, (train, test) in enumerate(loo.split(X, y)):
    Xtr, ytr = X[train], y[train]
    Xte, yte = X[test], y[test]
    test_name = f'model{i+1}'

    if standardize_input:
        zscorer = StandardScaler()
        Xtr = zscorer.fit_transform(Xtr)
        Xte = zscorer.transform(Xte)

    for method, reg in methods():
        # reg.cuda()
        show_table = False
        method_results_path = f'../results/errors/{method}_{gas}.pkl'
        if os.path.exists(method_results_path):
            method_errors = pickle.load(open(method_results_path, 'rb'))
        else:
            os.makedirs(Path(method_results_path).parent, exist_ok=True)
            method_errors = {}

        if test_name not in method_errors:
            reg.fit(Xtr, ytr)
            yte_pred = reg.predict(Xte)

            plot_result(yte[0], yte_pred[0], f'../results/plots/{method}/{str(test_name)}.png', err_fun=mse)

            error_mean = mse(yte, yte_pred)
            method_errors[test_name] = error_mean
            pickle.dump(method_errors, open(method_results_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            if ansia:
                show_table = True
        else:
            error_mean = method_errors[test_name]

        errors[method].append(error_mean)

        table.add(benchmark=test_name, method=method, v=error_mean)
        if show_table:
            table.latexPDF(table_path, resizebox=True)

    print(test_name)

method_replace={'their': 'RFbaseline'}
table.latexPDF(table_path, benchmark_order=sorted(table.benchmarks), verbose=False, resizebox=False, method_replace=method_replace)
for method, errors_means in errors.items():
    print(f'{method}\tMSE={np.mean(errors_means):.10f}+-{np.std(errors_means):.10f}')
