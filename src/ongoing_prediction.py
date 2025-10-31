from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, train_test_split
from os.path import join
from data import load_both_data
from data_analysis.curve_clustering import cluser_curves
from nn_3w_modules import FF3W, AE, AE2
from regression import NN3WayReg, DirectRegression, NearestNeighbor, MetaLearner, \
    TwoStepRegression
from result_table.src.format import Configuration
from result_table.src.table import LatexTable
from utils import mse, ResultTracker, plot_result
import sys

results_dir = '../results'
chosen_test = 'model38'

with_validation = False

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

hidden = [128,256,128]
hidden_big = [128,256,512,256,128]
hidden_small = [128]
Ldim = 128
Ldim_big = 256
Ldim_small = 64

# XYZ means 3way, and each signal is original
# Xyz means 3way, but y and z is pca, X is original
# XY  means 2way, the Z is still there but with wZ=0 (i.e., same capacity)
# X2YZ  means 3way, wY is double the weight of the rest

# testing idea: keep track of the y_loss independently and check whether there is a correlation tr-loss vs te-loss
# testing idea (-w): wY is times the other two


def methods():
    yield 'PAE2ZY', NN3WayReg(
        model=AE2(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),

def load_data():
    path_h2 = '../data/training/dataset_for_hydrogen.csv'
    path_n2 = '../data/training/dataset_for_nitrogen.csv'
    return load_both_data(
        path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True,
        return_index=True, exclude_id=['model41', 'model45']
    )


if __name__ == '__main__':
    error_scale=1e6

    test_names, Vin, Gin, Gout = load_data()
    (n_instances, V_dim), Gi_dim, Go_dim = Vin.shape, Gin.shape[1], Gout.shape[1]

    index = np.arange(Gin.shape[0])
    train_idx = index[test_names != chosen_test]
    test_idx  = index[test_names == chosen_test]
    test_name = test_names[test_idx[0]]
    print(f'{test_name}:')

    Vin_tr, Gin_tr, Gout_tr = Vin[train_idx], Gin[train_idx], Gout[train_idx]
    Vin_te, Gin_te, Gout_te = Vin[test_idx],  Gin[test_idx],  Gout[test_idx]

    for method, reg in methods():
        print(f'{method=}')

        # continue
        reg.fit(Gin_tr, Gout_tr, Vin_tr, in_val=None, show_test_model=(Gin_te, Gout_te))




