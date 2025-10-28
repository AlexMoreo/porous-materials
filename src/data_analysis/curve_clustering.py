from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data import load_both_data

def cluser_curves(X, n_clusters):
    # Input = np.hstack([Gin,Vin])
    # zscore = StandardScaler()
    # X = zscore.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans.predict(X), kmeans

if __name__ == '__main__':
    path_h2 = '../../data/training/dataset_for_hydrogen.csv'
    path_n2 = '../../data/training/dataset_for_nitrogen.csv'
    Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

    Input = Gin

    n_clusters = 4
    g, kmeans = cluser_curves(Input, n_clusters=n_clusters)

    print(g)
    print(Counter(g))

    fig, axes = plt.subplots(1, n_clusters, figsize=(3*n_clusters, 3))

    for gi in range(n_clusters):
        in_cluster = Input[g==gi]

        mean_curve = kmeans.cluster_centers_[gi]
        x = np.arange(len(mean_curve))
        axes[gi].plot(x, mean_curve, color='tab:blue', linewidth=3)
        axes[gi].set_title(f'Cluster {gi} has {len(in_cluster)} instances')

        for curve in in_cluster:
            axes[gi].plot(x, curve, color='tab:green', alpha=0.4)
            axes[gi].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
