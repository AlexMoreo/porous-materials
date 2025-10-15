from sklearn.decomposition import PCA
from data import *
from utils import *

for reduce in ['input', 'nitrogen', 'hydrogen']:

    components=8

    path_nitrogen = f'../data/training/dataset_for_nitrogen.csv'
    path_hydrogen = f'../data/training/dataset_for_hydrogen.csv'

    if reduce=='input':
        X, _ = load_data(path_nitrogen, cumulate_x=True, normalize=True)
        Z = X
    elif reduce=='nitrogen':
        _, y = load_data(path_nitrogen, cumulate_x=True, normalize=True)
        Z = y
    elif reduce=='hydrogen':
        _, y = load_data(path_hydrogen, cumulate_x=True, normalize=True)
        Z = y

    pca = PCA(n_components=components)
    Z_ = pca.fit_transform(Z)

    # Each component is an "eigencurve" centered at 0
    mean_curve = pca.mean_.reshape(1, -1).ravel()
    mean_amplitude = max(mean_curve)-min(mean_curve)

    x_axis = np.arange(Z.shape[1])  # original indexes
    n_show = components

    fig, axes = plt.subplots(n_show, 1, figsize=(8, 2 * n_show))

    for i in range(n_show):
        eigencurve = pca.components_[i]
        eigen_amplitude = max(eigencurve)-min(eigencurve)
        scale = mean_amplitude / eigen_amplitude

        curve_plus = mean_curve + scale * eigencurve

        axes[i].plot(x_axis, mean_curve, 'k--', label='Mean')
        axes[i].plot(x_axis, curve_plus, 'g', label=f'+1σ component {i + 1}')
        axes[i].legend()
        axes[i].set_title(f"Principal component {i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}% var)")
        axes[i].grid(True)

    fig.suptitle('Analysis for "' + reduce + '"', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 👈 deja espacio para el título general
    plt.show()
