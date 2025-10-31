from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, train_test_split
from os.path import join
from data import load_both_data
from data_analysis.curve_clustering import cluser_curves
from nn_3w_modules import FF3W, AE, AE2, AE3
from regression import NN3WayReg, DirectRegression, NearestNeighbor, MetaLearner, \
    TwoStepRegression, Ensemble
from result_table.src.format import Configuration
from result_table.src.table import LatexTable
from utils import mse, ResultTracker, plot_result
import sys

results_dir = '../results'
# only_tables = True
only_tables =  "tables" in sys.argv
add_ref_color=False
# chosen_test = 'model35'
# chosen_test = 'model53'
# chosen_test = 'model70'
chosen_test = None

with_validation = False

only_models = None
selected_tests = None

# PCA reductions
Go_pca = 8  # very good approximation (H2)
Gi_pca  = 12 # 12 is somehow good approximation (N2) -- do we need to simplify the input?
V_pca  = 8   # very good approximation (Vol)

# RQ: does dropout help? looks like no!
# RQ: does PCA helps? sometimes yes, others is better the orig representation... not clear!
# RQ: complexity matters? deeper networks work better or overfit? not clear
# RQ: best architecture?
# RQ: smoothing matters?
# RQ: more importance to loss Gout (wY)?
# RQ: better using Vin as a regularization autoencodding or not?
# RQ: better using Ldim>0 or not?


# XYZ means 3way, and each signal is original
# Xyz means 3way, but y and z is pca, X is original
# XY  means 2way, the Z is still there but with wZ=0 (i.e., same capacity)
# X2YZ  means 3way, wY is double the weight of the rest

# testing idea: keep track of the y_loss independently and check whether there is a correlation tr-loss vs te-loss
# testing idea (-w): wY is times the other two

baselines = ['RFY', 'RFy', '1NN' ]
rename_method = {}

def methods():
    rf = RandomForestRegressor()
    yield 'RFY', DirectRegression(rf),
    yield 'RFy', DirectRegression(rf, y_red=Go_pca)
    yield '1NN', NearestNeighbor()
    yield 'PAE2zy', NN3WayReg(
        model=AE2(
            Xdim=10, Zdim=10, Ydim=10, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=10, Z_red=10, Y_red=10, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=25_000
    ),
    yield 'PAE2ZY', NN3WayReg(
        model=AE2(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),
    yield 'PAExy', NN3WayReg( # nomenclature is wrong, should be PAEzy
        model=AE(
            Xdim=10, Zdim=10, Ydim=10, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=10, Z_red=10, Y_red=10, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=25_000
    ),
    yield 'PAEXY', NN3WayReg( # nomenclature is wrong, should be PAEZY; constructor is wrong, shoud be AE (relaunching...)
        model=AE2(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),
    yield 'PAEZY', NN3WayReg( # relauching... it was AE2 instead of AE
        model=AE(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),
    yield 'PAEZZY', NN3WayReg( # relauching... it was AE2 instead of AE, with more preasure towards Z
        model=AE(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),
    # yield 'PAEZ', NN3WayReg(
    #     model=AE3(
    #         Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
    #     ), wX=0, wZ=0.001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
    #     smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    # ),
    # yield 'PAEZZ', NN3WayReg(
    #     model=AE3(
    #         Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
    #     ), wX=0, wZ=0.01, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
    #     smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    # ),
    yield 'Ensamble', Ensemble(path=results_dir, methods=['PAE2zy', 'PAE2ZY', 'PAExy', 'PAEXY'])
    yield 'EnsRedo', Ensemble(path=results_dir, methods=['PAE2zy', 'PAE2ZY', 'PAExy', 'PAEZY'])
    yield 'EnsRedoZZ', Ensemble(path=results_dir, methods=['PAE2zy', 'PAE2ZY', 'PAExy', 'PAEZZY'])
    # yield 'Ensamble2', Ensemble(path=results_dir, methods=['1NN', 'PAE2zy', 'PAE2ZY', 'PAExy', 'PAEXY'])
    # yield 'Ensamble3', Ensemble(path=results_dir, methods=['PAE2zy', 'PAE2ZY', 'PAExy', 'PAEXY', 'PAEZ', 'PAEZZ'])
    # yield 'Ensamble-m', Ensemble(path=results_dir, methods=['PAE2zy', 'PAE2ZY', 'PAExy', 'PAEXY'], agg='mean')


def validation_idx(X, n_groups, val_prop=0.1, random_state=0):
    """
    Generates a stratified validation split based on cluster labels

    :param X: input data
    :param n_groups: number of groups
    :param val_prop: fractions of elements to be taken as validation data
    :param random_state: int, for replicability
    :return: a mask indicating 1=in_validation 0=not_in_validation
    """
    kmeans = KMeans(n_clusters=n_groups)
    kmeans.fit(X)
    cluster_labels = kmeans.predict(X)
    index = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(index, test_size=val_prop, random_state=random_state, stratify=cluster_labels)
    in_val_mask = np.zeros(len(cluster_labels), dtype=bool)
    in_val_mask[val_idx]=True
    return in_val_mask


def new_table():
    table_config=Configuration(
        show_std=False,
        mean_prec=3,
        resizebox=.6,
        stat_test=None,
        with_mean=True,
        with_rank=True
    )
    return LatexTable('mse', configuration=table_config)


def load_data():
    path_h2 = '../data/training/dataset_for_hydrogen.csv'
    path_n2 = '../data/training/dataset_for_nitrogen.csv'
    return load_both_data(
        path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True,
        return_index=True, exclude_id=['model41', 'model45']
    )


if __name__ == '__main__':
    table = new_table()
    table_path = join(results_dir, f'tables/mse.pdf')
    error_scale=1e6

    test_names, Vin, Gin, Gout = load_data()
    (n_instances, V_dim), Gi_dim, Go_dim = Vin.shape, Gin.shape[1], Gout.shape[1]

    val_mask = validation_idx(Gin, n_groups=5, val_prop=0.1, random_state=0)

    errors = defaultdict(lambda :[])

    if selected_tests is None:
        selected_tests = np.copy(test_names)

    # metalearner = MetaLearner(
    #     base_methods=['R3-XY', 'R3-XYZ', 'R3-y', 'R3-Y', 'R3-ZY'],
    #     prediction_dir='../results/val_predictions',
    #     X=Gin,
    #     Xnames=test_names
    # )

    loo = LeaveOneOut()
    for (train_idx, test_idx) in loo.split(Gin):
        test_name = test_names[test_idx[0]]
        print(f'{test_name}:')
        if chosen_test is not None and test_name!=chosen_test: continue

        Vin_tr, Gin_tr, Gout_tr, in_val = Vin[train_idx], Gin[train_idx], Gout[train_idx], val_mask[train_idx]
        Vin_te, Gin_te, Gout_te = Vin[test_idx],  Gin[test_idx],  Gout[test_idx]

        if not with_validation:
            in_val = None

        for method, reg in methods():
            if not only_tables:
                print(f'{method=}')
            # if only_models is not None and method not in only_models: continue
            method_errors = ResultTracker(join(results_dir, f'errors/{method}.pkl'))
            method_convergence = ResultTracker(join(results_dir, f'convergence/{method}.pkl'))
            method_val_predictions = ResultTracker(join(results_dir, f'val_predictions/{method}.pkl'))
            method_test_predictions = ResultTracker(join(results_dir, f'test_predictions/{method}.pkl'))

            if test_name not in method_errors:

                if only_tables or (test_name not in selected_tests):
                    continue

                # base name for all plots of this run
                partial_path = join(results_dir, f'plots/{method}/{str(test_name)}')

                if method in baselines:  # simple baseline
                    if isinstance(reg, TwoStepRegression) or isinstance(reg, NearestNeighbor):
                        reg.fit(Gin_tr, Gout_tr, Vin_tr)
                    else:
                        reg.fit(Gin_tr, Gout_tr)

                    if isinstance(reg, NearestNeighbor):
                        Gout_pred, Vout_pred = reg.predict(Gin_te)
                    else:
                        Gout_pred, Vout_pred = reg.predict(Gin_te), None

                elif isinstance(reg, Ensemble):
                    Gout_pred, Vout_pred = reg.predict(test_name)
                    if Gout_pred is None:
                        print('\tensemble found incomplete data')
                        continue

                else:  # 3way prediction
                    # continue
                    reg.fit(Gin_tr, Gout_tr, Vin_tr, in_val) #, show_test_model=(Gin_te, Gout_te))
                    Gout_pred, Gin_rec, Vout_pred = reg.predict(Gin_te, return_XZ=True)

                    #plotting
                    if Gin_rec is not None:
                        plot_result(Gin_te[0], Gin_rec[0], partial_path + '-Gin.png', err_fun=mse, scale_err=1e6)

                    # save predictions for validation data
                    # select = in_val if (in_val is not None) else np.ones(Gin_tr.shape[0], dtype=bool)
                    # if isinstance(reg, NN3WayReg):
                    #     Gout_val_pred = reg.predict(Gin_tr[select], return_XZ=False)
                    #     method_val_predictions.update(test_name, {
                    #         'test_names': test_names[train_idx][select],
                    #         'predictions': Gout_val_pred,
                    #         'errors': mse(Gout_tr[select], Gout_val_pred, average=False)
                    #     })

                # gather result data
                method_test_predictions.update(test_name, {
                    'Gout': Gout_pred,
                    'Vout': Vout_pred
                })
                if hasattr(reg, 'best_loss'):
                    method_convergence.update(test_name, reg.best_loss)

                # plotting
                plot_result(Gout_te[0], Gout_pred[0], partial_path + '-Gout.png', err_fun=mse, scale_err=1e6)
                if Vout_pred is not None:
                    plot_result(Vin_te[0], Vout_pred[0], partial_path + '-Vin.png', err_fun=mse, scale_err=1e6)

                # save error results
                error_mean = mse(Gout_te, Gout_pred)
                method_errors.update(test_name, error_mean)

            else:
                error_mean = method_errors.get(test_name)

            errors[method].append(error_mean)
            table.add(benchmark=test_name,
                      # f'model{i+1:3d}',
                      method=method, v=error_mean*error_scale)

        # meta learner
        # chosen_method = metalearner.predict(test_name)
        # print(f'for test={test_name} I would chose {chosen_method}')


    # reorder by cluster-id
    # groups, _ = cluser_curves(Gin, n_clusters=5)
    # order = []
    # for g in sorted(np.unique(groups)):
    #     order.extend(list(test_names[groups==g]))
    # table.reorder_benchmarks(order)
    if add_ref_color:
        for test in test_names:
            table.add(benchmark=test, method='REF', v=1.)

    table.latexPDF(table_path, landscape=False)



