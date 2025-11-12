from sklearn.decomposition import PCA
from data import *
from utils import *

for reduce in ['volume', 'nitrogen', 'hydrogen']:

    components=8

    # path_n2 = f'../../data/training/dataset_for_nitrogen.csv'
    # path_h2 = f'../../data/training/dataset_for_hydrogen.csv'
    # Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

    path_h2 = '../../data/training_molg/dataset_hydrogen_molg.csv'
    path_n2 = '../../data/training_molg/dataset_nitrogen_molg.csv'
    test_names, Vin, Gin, Gout = load_both_data(
        path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=False, normalize=True,
        return_index=True,
    )
    Gin *= 108570
    Gout *= 108570

    Z = {
        'volume': Vin,
        'nitrogen': Gin,
        'hydrogen': Gout
    }[reduce]

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
        axes[i].set_title(f"Principal component {i + 1} ({pca.explained_variance_ratio_[i] * 100:.3f}% var)")
        axes[i].grid(True)

    fig.suptitle('Analysis for "' + reduce + '"', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
