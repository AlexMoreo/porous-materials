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

# problem = 'direct'
problem = 'inverse'
assert problem in ['direct', 'inverse'], 'wrong problem'
problemtag = '-direct'

components=8

path = f'../data/training/dataset_for_{gas}.csv'
X, y = load_data(path, cumulate_x=True, normalize=True)

if problem == 'inverse':
    X, y = y, X
    problemtag=''


pca = PCA(n_components=components)
y_ = pca.fit_transform(y)

# Cada componente es una "eigencurva" centrada en 0
# Si quieres verla como una curva realista, añade la media:
mean_curve = pca.mean_.reshape(1, -1).ravel()

x_axis = np.arange(y.shape[1])  # tus 60 puntos de muestreo
n_show = components  # número de componentes a visualizar

fig, axes = plt.subplots(n_show, 1, figsize=(8, 2 * n_show))

for i in range(n_show):
    eigencurve = pca.components_[i]

    # Escala la componente para verla más claramente (opcional)
    #scale = 2 * np.sqrt(pca.explained_variance_[i])
    scale = 1

    curve_plus = mean_curve + scale * eigencurve
    # curve_minus = mean_curve - scale * eigencurve

    axes[i].plot(x_axis, mean_curve, 'k--', label='Media')
    axes[i].plot(x_axis, curve_plus, 'g', label=f'+1σ componente {i + 1}')
    # axes[i].plot(x_axis, curve_minus, 'r', label=f'-1σ componente {i + 1}')
    axes[i].legend()
    axes[i].set_title(f"Componente principal {i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}% varianza)")
    axes[i].grid(True)

plt.tight_layout()
plt.show()
