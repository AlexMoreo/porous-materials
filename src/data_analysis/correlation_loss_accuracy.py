import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from utils import mse, ResultTracker, plot_result
from tqdm import tqdm
from three_way_prediction import methods

"""
This script analyzes whether there is any correlation between the best training loss and the final test performance.
The idea is to select, from a pool of base regressors, the one that had obtained the best training loss. The script
generates scatter plots in which every point represents a coordinate (tr-loss, te-loss). The hope is that the left-most
method (smallest tr-loss) is (almost) always the one with the smallest te-loss. 
The plots do not seem to indicate this is the case...  
"""

only_models = None
# only_models = ['R3-xzy',]
# only_models = [f'R3-XY-v{i+1}' for i in range(10)]
# only_models = ['R3-Xyz-L0']


# methods = ['R3-XYZ', 'R3-XYZ-L0', 'R3-Xyz-L0', 'R3-XY', 'R3-ZY', 'R3-Y', 'R3-Xyz', 'R3-Xy', 'R3-zy', 'R3-xzy', 'R3-y']
# methods = [f'R3-XY{i}' for i in range(5)]
methods = ['R3-XY', 'R3-XYZ', 'R3-y', 'R3-Y', 'R3-ZY']

selected_tests = None
# selected_tests = [35]

if selected_tests is None:
    selected_tests = list(np.arange(111)+1)  # default: all tests

base_dir = '../../results'

out_dir = f'{base_dir}/correlations/'
os.makedirs(out_dir, exist_ok=True)

for sel_test in tqdm(selected_tests, desc='plotting', total=111):
    method_pairs = {}
    for method in methods:
        errors = ResultTracker(f'{base_dir}/errors/{method}.pkl')
        convergence = ResultTracker(f'{base_dir}/convergence/{method}.pkl')

        method_pairs[method] = [(convergence.get(f'model{i}'), errors.get(f'model{i}')) for i in selected_tests
                                if f'model{i}' in errors and i==sel_test]

    plt.figure(figsize=(7, 5))

    all_x = []
    all_y = []
    for method, values in method_pairs.items():
        values = np.array(values)
        train_loss = values[:, 0]
        accuracy = values[:, 1]
        x = np.log(train_loss)
        y = np.log(accuracy)
        plt.scatter(x, y, label=method, alpha=0.7)
        all_x.extend(x)
        all_y.extend(y)

    smallest_x = np.argmin(all_x)
    plt.scatter(all_x[smallest_x], all_y[smallest_x], edgecolor='black', facecolor='none', s=150, linewidth=2, zorder=3)

    plt.xlabel("Train MSE")
    plt.ylabel("Test MSE")
    plt.title(f"Model {sel_test}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(out_dir, f'Model_{sel_test}.png'))
