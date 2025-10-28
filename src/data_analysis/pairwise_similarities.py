import numpy as np
from sklearn.neighbors import NearestNeighbors
from data import load_both_data
from regression import PCAadapt
import matplotlib.pyplot as plt

from utils import mse

path_h2 = '../../data/training/dataset_for_hydrogen.csv'
path_n2 = '../../data/training/dataset_for_nitrogen.csv'
Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

# PCA reductions
Go_pca = 8  # very good approximation (H2)
Gi_pca  = 12 # 12 is somehow good approximation (N2) -- do we need to simplify the input?
V_pca  = 8   # very good approximation (Vol)

adapt_Gi = PCAadapt(components=Gi_pca, force=False)
adapt_Go = PCAadapt(components=Go_pca, force=False)
adapt_V  = PCAadapt(components=V_pca,  force=False)

nn = NearestNeighbors(n_neighbors=2, metric='euclidean')  # 2 bcause the closes one is itself
nn.fit(Gin)
distances, indices = nn.kneighbors(Gin)

# keep the second closest (the first one is itself)
nearest_idx = indices[:, 1]
nearest_dist = distances[:, 1]

for i, (Gin_i, Gout_i, nearest_idx_i) in enumerate(zip(Gin, Gout, nearest_idx)):
    x = np.arange(len(Gin_i))
    nearest_curve = Gin[nearest_idx_i]
    nearest_output = Gout[nearest_idx_i]
    output_dist = mse(Gout_i, nearest_output)


    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].plot(x, Gin_i, color='tab:blue')
    axes[0].plot(x, nearest_curve, color='tab:orange')
    axes[0].set_title(f"Input {i+1}: Closest model (idx={nearest_idx_i+1}, dist={nearest_dist[i]*1e6:.3f})")

    err = mse(Gout_i, nearest_output)*1e6
    x = np.arange(len(Gout_i))
    axes[1].plot(x, Gout_i, color='tab:blue')
    axes[1].plot(x, nearest_output, color='tab:orange')
    axes[1].set_title(f"Closest output, mse={err:.4f}")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
    plt.close('all')



