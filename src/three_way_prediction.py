from collections import defaultdict

import numpy as np
from sklearn.model_selection import LeaveOneOut

from data import load_both_data
from nn_3w_modules import FF3W
from regression import NN3WayReg
from utils import mse, ResultTracker, plot_result


# PCA reductions
Gout_pca = 8  # very good approximation (H2)
Gin_pca  = None # 12 issomehow good approximation (N2) -- do we need to simplify the input?
Vin_pca  = 8   # very good approximation (Vol)


def methods():
    yield 'R3-3L128', NN3WayReg(
        model=FF3W(
            Xdim=Gindim, Zdim=Vdim, Ydim=Goutdim, Ldim=128, hidden=[64,128,256]
        )
    ),
    yield 'R3-3L128-PCAZY8', NN3WayReg(
        model=FF3W(
            Xdim=Gindim, Zdim=Vin_pca, Ydim=Gout_pca, Ldim=128, hidden=[64,128,256]
        ),
        reduce_Z=Vin_pca,
        reduce_Y=Gout_pca
    )

path_h2 = '../data/training/dataset_for_hydrogen.csv'
path_n2 = '../data/training/dataset_for_nitrogen.csv'
Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

Vdim, Gindim, Goutdim = Vin.shape[1], Gin.shape[1], Gout.shape[1]

errors = defaultdict(lambda :[])
test_names = []

selected_tests = list(np.arange(len(Vin))+1)  # default: all tests
selected_tests = [1, 9, 17, 26, 34, 38, 48, 53, 62, 67, 73, 80, 89, 93, 105]

loo = LeaveOneOut()
for i, (train_idx, test_idx) in enumerate(loo.split(Vin, Gin, Gout)):
    if (i+1) not in selected_tests: continue

    test_name = f'model{i+1}'
    print(f'{test_name}:')

    Vin_tr, Gin_tr, Gout_tr = Vin[train_idx], Gin[train_idx], Gout[train_idx]
    Vin_te, Gin_te, Gout_te = Vin[test_idx],  Gin[test_idx],  Gout[test_idx]

    for method, reg in methods():
        print(f'{method=}')
        show_table = False
        method_errors = ResultTracker(f'../results/errors/{method}.pkl')
        method_convergence = ResultTracker(f'../results/convergence/{method}.pkl')

        if test_name not in method_errors:
            reg.fit(Gin_tr, Gout_tr, Vin_tr)
            Gout_pred, Gin_rec, Vin_rec = reg.predict(Gin_te, return_XZ=True)

            if hasattr(reg, 'best_loss'):
                method_convergence.update(test_name, reg.best_loss)

            error_mean = mse(Gout_te, Gout_pred)
            method_errors.update(test_name, error_mean)

            plot_result(Gout_te[0], Gout_pred[0], f'../results/plots/{method}/{str(test_name)}-Gout.png', err_fun=mse)
            plot_result(Gin_te[0], Gin_rec[0], f'../results/plots/{method}/{str(test_name)}-Gin.png', err_fun=mse)
            plot_result(Vin_te[0], Vin_rec[0], f'../results/plots/{method}/{str(test_name)}-Vin.png', err_fun=mse)
        else:
            error_mean = method_errors.get(test_name)

        errors[method].append(error_mean)



