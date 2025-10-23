from collections import defaultdict

import numpy as np
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.neighbors import KernelDensity

from data import load_both_data
from nn_3w_modules import FF3W
from regression import NN3WayReg, PCAadapt
from result_table.src.table import Table
from utils import mse, ResultTracker, plot_result
import matplotlib.pyplot as plt
from scipy.stats import norm


table = Table('mse')
table_path = f'../../results/tables/cases.pdf'
table.format.show_std = False
table.format.mean_prec = 5
table.format.stat_test = None
error_scale=1e6

path_h2 = '../../data/training/dataset_for_hydrogen.csv'
path_n2 = '../../data/training/dataset_for_nitrogen.csv'
Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

Vdim, Gindim, Goutdim = Vin.shape[1], Gin.shape[1], Gout.shape[1]

errors = defaultdict(lambda :[])
test_names = []

selected_tests = np.arange(len(Vin))+1  # default: all tests
# selected_tests = [1, 9, 17, 26, 34, 38, 48, 53, 62, 67, 73, 80, 89, 93, 105]

selected_tests = np.array(selected_tests)

kde = KernelDensity(kernel='gaussian')


def kde_find_bandwidth(X, cv=10):
    grid = {'bandwidth': np.linspace(0.0001, 1, 100)}
    kde = GridSearchCV(KernelDensity(kernel='gaussian'), grid, cv=cv)
    kde.fit(X)
    bandwidth = kde.best_params_['bandwidth']
    print(f'grid search found {bandwidth=:.5f}')
    return bandwidth


def kde_score(kde, X):
    return np.exp(kde.score_samples(X))


X = PCAadapt(components=3).fit_transform(Gin)
bandwidth = kde_find_bandwidth(X)

loo = LeaveOneOut()
likelihoods = []
for (train_idx, test_idx) in loo.split(Gin):
    i = test_idx[0]
    if (i+1) not in selected_tests: continue

    Xtr, Xte = X[train_idx], X[test_idx]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Xtr)
    log_likelihood = kde.score_samples(Xte)[0]
    likelihoods.append(log_likelihood)
    table.add(benchmark=f'model{i + 1:3d}', method='Likelihood', v=log_likelihood)

table.latexPDF(table_path, benchmark_order=sorted(table.benchmarks), verbose=False, resizebox=True, method_replace={}, landscape=False)

likelihoods = np.array(likelihoods)
order = np.argsort(likelihoods)
ordered_cases = selected_tests[order]
ordered_like  = likelihoods[order]

print("Problematic:")
for case, like in zip(ordered_cases, ordered_like):
    print(f'model-{case:2d}: {like:.4f}')

mean_ll, std_ll = np.mean(likelihoods), np.std(likelihoods)
z_scores = (likelihoods - mean_ll) / std_ll
z_threshold = 2.5
idx = np.arange(len(selected_tests))
outliers_pos = idx[np.abs(z_scores) > z_threshold]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(likelihoods, bins=20, density=True, alpha=0.6, color="skyblue", label="log-likelihoods")
x_vals = np.linspace(min(likelihoods), max(likelihoods), 200)
pdf = norm.pdf(x_vals, mean_ll, std_ll)
ax.plot(x_vals, pdf, 'r-', lw=2, label="Normal fit")
lower, upper = mean_ll - z_threshold * std_ll, mean_ll + z_threshold * std_ll
ax.axvline(lower, color='orange', linestyle='--', lw=2, label=f"-{z_threshold}σ")
ax.axvline(upper, color='orange', linestyle='--', lw=2, label=f"+{z_threshold}σ")
outlier_vals = likelihoods[outliers_pos]
ax.scatter(outlier_vals, np.zeros_like(outlier_vals), color='red', zorder=5, label="Outliers")

ax.set_title("Log-likelihood distribution with Z-score cutoff")
ax.set_xlabel("log-likelihood")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()

# Q-Q plot
import scipy.stats as stats
fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(likelihoods, dist="norm", plot=ax)
ax.set_title("Q–Q plot of log-likelihoods vs Normal distribution")
plt.show()




for pos in outliers_pos:
    print(f'Model-{selected_tests[pos]:2d} has log-likelihood {likelihoods[pos]:.4f}')

