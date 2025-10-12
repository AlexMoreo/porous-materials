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

force = False
tests = 'all'

# problem = 'direct'
problem = 'inverse'
assert problem in ['direct', 'inverse'], 'wrong problem'

reduce_input = True
components=12


path = f'../data/training/dataset_for_{gas}.csv'
X, y = load_data(path, cumulate_x=True, normalize=True)

if problem == 'inverse':
    X, y = y, X

lr=0.005

def methods():
    input_size = X.shape[1]
    output_size = y.shape[1]
#     yield 'SVR', MultiOutputRegressor(LinearSVR(max_iter=3000))
#     yield 'SVR-optim', GridSearchCV(
#         MultiOutputRegressor(SVR(max_iter=3000)),
#         param_grid={
#             'estimator__C': np.logspace(-4, 4, 9),
#             'estimator__gamma': ['scale', 'auto']
#         },
#         refit=True,
#         verbose=True,
#         n_jobs=-1
# )
#     yield 'GrBoostR-optim', GridSearchCV(
#         MultiOutputRegressor(GradientBoostingRegressor()),
#         param_grid={
#             'estimator__learning_rate': [0.1, 0.01],
#         },
#         refit=True,
#         verbose=True,
#         n_jobs=-1
#     )
    # yield 'StackSVR', StackRegressor()
    # yield 'StackRF-tat', StackRegressor(RandomForestRegressor(), mode='tat')
    # yield 'StackRF-5fcv', StackRegressor(RandomForestRegressor(), mode='kfcv', k=5)
    # yield 'StackRF', StackRegressor(RandomForestRegressor())
    # yield 'lstm-256-4', NeuralRegressor(model=LSTMModel(input_size, output_size))
    # yield 'ff-128-256-128', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256,128]), clip=True)
    # yield 'ff-128-256-128-noNorm', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128, 256, 128]), clip=False)
    # yield 'ff-128-256-128-mono', NeuralRegressor(MonotonicNN(FFModel(input_size, output_size, hidden_sizes=[128, 256, 128])), clip=True)
    # yield 'ff-128-256-128-mono-smooth', NeuralRegressor(MonotonicNN(FFModel(input_size, output_size, hidden_sizes=[128, 256, 128])), clip=True, reg_strength=0.01)
    # yield 'ff-64-64-mono', NeuralRegressor(MonotonicNN(FFModel(input_size, output_size, hidden_sizes=[64, 64])), clip=True)
    #yield f'ff-16-PCA{components}', NeuralRegressor(FFModel(input_size, output_size=components, hidden_sizes=[16]), clip=False, reg_strength=0, lr=lr)
    #yield f'ff-32-PCA{components}', NeuralRegressor(FFModel(input_size, output_size=components, hidden_sizes=[32]), clip=False, reg_strength=0, lr=lr)
    #yield f'ff-32-64-32-PCA{components}', NeuralRegressor(FFModel(input_size, output_size=components, hidden_sizes=[32,64,32]), clip=False, reg_strength=0, lr=lr)
    yield f'ff-64-128-128-64-PCA{components}', NeuralRegressor(FFModel(input_size, output_size=components, hidden_sizes=[64, 128, 128, 64]), clip=False, reg_strength=0, lr=lr)
    # yield 'ff-128-256-512-512-256-128-mono', NeuralRegressor(MonotonicNN(FFModel(input_size, output_size, hidden_sizes=[128,256,512,512,256,128])), clip=True)
    # yield 'ff-128-256-128-r02', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256,128]), reg_strength=0.1)
    # yield 'ff-256-256-256-128-128-r01', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[256,256,256,128,128]), reg_strength=0.1)
    # yield 'ff-128-128-r01', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,128], smooth_length=3), reg_strength=0.1)
    # yield 'ff-128-128-r01-mono', NeuralRegressor(MonotonicNN(input_size, output_size, hidden_sizes=[128, 128], smooth_length=0), reg_strength=0.01, lr=0.001)
    # yield 'ff-128-256-128-r01', NeuralRegressor(FFModel(input_size, output_size, hidden_sizes=[128,256,128], smooth_length=3), reg_strength=0.1)
    # yield 'transformer-5L-tiny', NeuralRegressor(
    #     TransformerRegressor(input_size, output_size, num_layers=5, d_model=32, nhead=2, dim_feedforward=32, dropout=0.5), clip=True, reg_strength=0.001
    # )
    # yield 'transformer-2L', NeuralRegressor(TransformerRegressor(input_size, output_size, num_layers=2), clip=True)
    # yield 'transformer-2L-small', NeuralRegressor(
    #     TransformerRegressor(input_size, output_size, num_layers=2, d_model=64, nhead=4, dim_feedforward=64, dropout=0.5), clip=True, reg_strength=0.001
    # )
    # yield 'transformer-2L-big', NeuralRegressor(
    #     TransformerRegressor(input_size, output_size, num_layers=2, d_model=256, nhead=16, dim_feedforward=512, dropout=0.1), clip=True, reg_strength=0.001
    # )
    # yield 'transformer-2L-noreg', NeuralRegressor(TransformerRegressor(input_size, output_size, num_layers=2), clip=True, reg_strength=0)
    # yield 'transformer-1L', NeuralRegressor(TransformerRegressor(input_size, output_size, num_layers=1), clip=True)
    #yield f'transformer-1L-PCA{components}', NeuralRegressor(TransformerRegressor(input_size, output_size=components, num_layers=1), clip=True, reg_strength=0, lr=lr*0.1)
    #yield f'transformer-2L-PCA{components}', NeuralRegressor(TransformerRegressor(input_size, output_size=components, num_layers=2), clip=True, reg_strength=0, lr=lr*0.1)
    # yield 'transformer-1L-noreg', NeuralRegressor(TransformerRegressor(input_size, output_size, num_layers=1), clip=True, reg_strength=0)
    # yield 'transformer-1L-small', NeuralRegressor(
    #     TransformerRegressor(input_size, output_size, num_layers=1, d_model=64, nhead=4, dim_feedforward=64,
    #                          dropout=0.5), clip=True, reg_strength=0.001
    # )
    # yield 'transformer-1L-big', NeuralRegressor(
    #     TransformerRegressor(input_size, output_size, num_layers=1, d_model=256, nhead=16, dim_feedforward=512,
    #                          dropout=0.1), clip=True, reg_strength=0.001
    # )
    # method, reg = 'lstm-256-4', LSTMregressor(hidden_size=256, num_layers=4)
    # yield 'RF', RandomForestRegressor()
    yield f'RF-PCA{components}', RandomForestRegressor()


suffix = '-direct' if problem=='direct' else ''

table = Table('mse')
table_path = f'../results{suffix}/tables/mse_{gas}.pdf'
table.format.show_std = False
table.format.mean_prec = 5
table.format.stat_test = None
ansia = False

errors = defaultdict(lambda :[])
test_names = []

standardize_input = False


loo = LeaveOneOut()
for i, (train, test) in enumerate(loo.split(X, y)):
    test_name = f'model{i+1}'

    if tests != 'all' and test_name != tests:
        continue

    Xtr, ytr = X[train], y[train]
    Xte, yte = X[test], y[test]

    print(f'{test_name}:')

    if standardize_input:
        zscorer = StandardScaler()
        Xtr = zscorer.fit_transform(Xtr)
        Xte = zscorer.transform(Xte)

    if reduce_input:
        pca = PCA(n_components=components)
        ytr_ = pca.fit_transform(ytr)
        yte_ = pca.transform(yte)
    else:
        yte_ = yte

    for method, reg in methods():
        print(f'{method=}')
        show_table = False
        method_errors = ResultTracker(f'../results{suffix}/errors/{gas}/{method}.pkl')
        method_convergence = ResultTracker(f'../results{suffix}/convergence/{gas}/{method}.pkl')

        if test_name not in method_errors or force:
            reg.fit(Xtr, ytr_)
            yte_pred = reg.predict(Xte)

            if reduce_input:
                yte_pred = pca.inverse_transform(yte_pred)

            if hasattr(reg, 'best_loss'):
                method_convergence.update(test_name, reg.best_loss)

            error_mean = mse(yte, yte_pred)
            method_errors.update(test_name, error_mean)

            plot_result(yte[0], yte_pred[0], f'../results{suffix}/plots/{gas}/{method}/{str(test_name)}.png', err_fun=mse)
        else:
            error_mean = method_errors.get(test_name)

        errors[method].append(error_mean)

        table.add(benchmark=test_name, method=method, v=error_mean)
        if show_table:
            table.latexPDF(table_path, resizebox=True)



method_replace={}
# method_replace={'their': 'RFbaseline'}
table.latexPDF(table_path, benchmark_order=sorted(table.benchmarks), verbose=False, resizebox=True, method_replace=method_replace, landscape=False)
for method, errors_means in errors.items():
    print(f'{method}\tMSE={np.mean(errors_means):.10f}+-{np.std(errors_means):.10f}')
