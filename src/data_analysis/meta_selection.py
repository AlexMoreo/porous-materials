import os.path
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from data import load_both_data
from nn_3w_modules import FF3W
from regression import NN3WayReg, RandomForestXYPCA
from result_table.src.format import Configuration
from result_table.src.table import LatexTable
from utils import mse, ResultTracker, plot_result, load_method_errors

methods = ([f'R3-XY{i+1}' for i in range(5)] + [f'R3-ZY{i+1}' for i in range(5)] + [f'R3-Y{i+1}' for i in range(5)] +
           [f'R3-zy{i+1}' for i in range(5)]) + [f'R3-y{i+1}' for i in range(5)]
scale = 1e6

if __name__ == '__main__':
    conf=Configuration(
        show_std=False,
        mean_prec=5,
        resizebox=.9,
        stat_test=None,
        with_mean=True,
        with_rank=True
    )
    table = LatexTable('mse-sel', configuration=conf)
    table_path = f'../../results/tables/mse-sel.pdf'

    errors = defaultdict(lambda :[])
    convergences = []
    errors = []
    for method in methods:
        convergence_path = f'../results/convergence/{method}.pkl'
        if os.path.exists(convergence_path):
            conv = load_method_errors(path=f'../results/convergence/{method}_yloss.pkl')
            err  = load_method_errors(path=f'../results/errors/{method}.pkl')
            assert len(conv) == len(err), 'wrong shape'
            for i, err_i in enumerate(err):
                table.add(benchmark=f'model{i+1:3d}', method=method, v=err_i*scale)
            convergences.append(conv)
            errors.append(err)
    convergences = np.asarray(convergences).T
    errors = np.asarray(errors).T
    selected = np.argmin(convergences, axis=1)
    sel_error = errors[np.arange(convergences.shape[0]),selected]
    for i, err_i in enumerate(sel_error):
        table.add(benchmark=f'model{i + 1:3d}', method='meta', v=err_i * scale)

    table.latexPDF(table_path, landscape=False)



