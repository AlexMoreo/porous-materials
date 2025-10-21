from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from data import load_both_data
from nn_3w_modules import FF3W
from regression import NN3WayReg, RandomForestRegressorPCA
from result_table.src.format import Configuration
from result_table.src.table import LatexTable

from utils import mse, ResultTracker, plot_result

only_tables = True
# only_tables = False

only_models = None
# only_models = ['R3-xzy',]
# only_models = [f'R3-XY-v{i+1}' for i in range(10)]
# only_models = ['R3-Xyz-L0']


selected_tests = None
# selected_tests = [35]

# PCA reductions
Go_pca = 8  # very good approximation (H2)
Gi_pca  = 12 # 12 is somehow good approximation (N2) -- do we need to simplify the input?
Vin_pca  = 8   # very good approximation (Vol)


# RQ: does dropout help? looks like no!
# RQ: does PCA helps? sometimes yes, others is better the orig representation... not clear!
# RQ: complexity matters? deeper networks work better or overfit? not clear
# RQ: best architecture?
# RQ: smoothing matters?
# RQ: more importance to loss Gout (wY)?
# RQ: better using Vin as a regularization autoencodding or not?
# RQ: better using Ldim>0 or not?


hidden = [128,256,128]
Ldim = 128

# XYZ means 3way, and each signal is original
# Xyz means 3way, but y and z is pca, X is original
# XY  means 2way, the Z is still there but with wZ=0 (i.e., same capacity)
# X2YZ  means 3way, wY is double the weight of the rest

# testing idea: keep track of the y_loss independently and check whether there is a correlation tr-loss vs te-loss
# testing idea (-w): wY is times the other two

baselines = ['RF', 'RFy', 'RFxy']

def methods():
    yield 'RF', RandomForestRegressor(),
    yield 'RFy', RandomForestRegressorPCA(Xreduce_to=Gi_dim, Yreduce_to=Go_pca)
    yield 'RFxy', RandomForestRegressorPCA(Xreduce_to=Gi_pca, Yreduce_to=Go_pca)
    yield 'R3-XYZ', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden
        )
    ),
    yield 'R3-XYZw', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden
        ),
        wX=0.1, wZ=0.1, wY=1
    ),
    yield 'R3-XYZ-L0', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=0, hidden=hidden
        )
    ),
    yield 'R3-Xyz-L0', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=Vin_pca, Ydim=Go_pca, Ldim=0, hidden=hidden
        ),
        reduce_Z=Vin_pca, reduce_Y=Go_pca,
        checkpoint_id=2
    ),
    yield 'R3-XY', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden
        ),
        wZ=0
    ),
    yield 'R3-XYw', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden
        ),
        wX=0.1, wZ=0
    ),
    # yield 'R3-XY-v1', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v2', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v3', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v4', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v5', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v6', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v7', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v8', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v9', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    # yield 'R3-XY-v10', NN3WayReg(model=FF3W(Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden), wZ=0, checkpoint_id=1),
    yield 'R3-ZY', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden
        ),
        wX=0
    ),
    yield 'R3-Y', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=Ldim, hidden=hidden
        ),
        wX=0, wZ=0
    ),
    yield 'R3-Xyz', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=Vin_pca, Ydim=Go_pca, Ldim=Ldim, hidden=hidden
        ),
        reduce_Z=Vin_pca, reduce_Y=Go_pca
    ),
    yield 'R3-Xy', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_pca, Ldim=Ldim, hidden=hidden
        ),
        reduce_Y=Go_pca,
        wZ=0
    ),
    yield 'R3-zy', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=Vin_pca, Ydim=Go_pca, Ldim=Ldim, hidden=hidden
        ),
        reduce_Z=Vin_pca, reduce_Y=Go_pca,
        wX=0
    ),
    yield 'R3-xzy', NN3WayReg(
        model=FF3W(
            Xdim=Gi_pca, Zdim=Vin_pca, Ydim=Go_pca, Ldim=Ldim, hidden=hidden
        ),
        reduce_X=Gi_pca, reduce_Z=Vin_pca, reduce_Y=Go_pca,
        wX=0
    ),
    yield 'R3-y', NN3WayReg(
        model=FF3W(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_pca, Ldim=Ldim, hidden=hidden
        ),
        reduce_Y=Go_pca,
        wX=0, wZ=0
    ),

if __name__ == '__main__':
    conf=Configuration(
        show_std=False,
        mean_prec=5,
        resizebox=.9,
        stat_test=None,
        with_mean=True,
        with_rank=True
    )
    table = LatexTable('mse', configuration=conf)
    table_path = f'../results/tables/mse.pdf'
    error_scale=1e6

    path_h2 = '../data/training/dataset_for_hydrogen.csv'
    path_n2 = '../data/training/dataset_for_nitrogen.csv'
    Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

    V_dim, Gi_dim, Go_dim = Vin.shape[1], Gin.shape[1], Gout.shape[1]

    errors = defaultdict(lambda :[])
    test_names = []

    if selected_tests is None:
        selected_tests = list(np.arange(len(Vin))+1)  # default: all tests

    loo = LeaveOneOut()
    for (train_idx, test_idx) in loo.split(Gin):
        i = test_idx[0]
        if (i+1) not in selected_tests: continue

        test_name = f'model{i+1}'
        print(f'{test_name}:')

        Vin_tr, Gin_tr, Gout_tr = Vin[train_idx], Gin[train_idx], Gout[train_idx]
        Vin_te, Gin_te, Gout_te = Vin[test_idx],  Gin[test_idx],  Gout[test_idx]

        for method, reg in methods():
            print(f'{method=}')
            # if only_models is not None and method not in only_models: continue
            method_errors = ResultTracker(f'../results/errors/{method}.pkl')
            method_convergence = ResultTracker(f'../results/convergence/{method}.pkl')
            method_y_loss = ResultTracker(f'../results/convergence/{method}_yloss.pkl')

            if test_name not in method_errors:
                if only_tables:
                    continue
                if method in baselines:  # simple baseline
                    reg.fit(Gin_tr, Gout_tr)
                    Gout_pred = reg.predict(Gin_te)
                    partial_path = f'../results/plots/{method}/{str(test_name)}'
                    plot_result(Gout_te[0], Gout_pred[0], partial_path + '-Gout.png', err_fun=mse, scale_err=1e6)

                else:  # 3way prediction
                    # continue
                    reg.fit(Gin_tr, Gout_tr, Vin_tr)
                    Gout_pred, Gin_rec, Vin_rec = reg.predict(Gin_te, return_XZ=True)

                    if hasattr(reg, 'best_loss'):
                        method_convergence.update(test_name, reg.best_loss)
                    if hasattr(reg, 'best_loss_y'):
                        method_y_loss.update(test_name, reg.best_loss_y)

                    partial_path = f'../results/plots/{method}/{str(test_name)}'
                    plot_result(Gout_te[0], Gout_pred[0], partial_path+'-Gout.png', err_fun=mse, scale_err=1e6)
                    plot_result(Gin_te[0], Gin_rec[0], partial_path+'-Gin.png', err_fun=mse, scale_err=1e6)
                    plot_result(Vin_te[0], Vin_rec[0], partial_path+'-Vin.png', err_fun=mse, scale_err=1e6)

                error_mean = mse(Gout_te, Gout_pred)
                method_errors.update(test_name, error_mean)


            else:
                error_mean = method_errors.get(test_name)

            errors[method].append(error_mean)
            table.add(benchmark=f'model{i+1:3d}', method=method, v=error_mean*error_scale)

    table.latexPDF(table_path, landscape=False)



