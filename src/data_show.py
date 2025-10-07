import matplotlib.pyplot as plt
import numpy as np

from data import load_data

path = '../data/training/dataset_for_hydrogen.csv'
# path = '../data/training/dataset_for_nitrogen.csv'
X, Y = load_data(path, cumulate_x=True, normalize=True)

block_size = 5
n_samples = X.shape[0]

for start in range(0, n_samples, block_size):
    end = min(start + block_size, n_samples)
    current_block = end - start

    fig, axes = plt.subplots(current_block, 2, figsize=(10, 2.5 * current_block))

    # Ensure axes is always 2D
    if current_block == 1:
        axes = np.array([axes])

    for i in range(current_block):
        idx = start + i
        # Plot X
        axes[i, 0].plot(np.arange(X.shape[1]), X[idx, :], color='steelblue')
        axes[i, 0].set_title(f'Sample {idx} – X (input)')
        axes[i, 0].set_xlabel('Feature index')
        axes[i, 0].set_ylabel('Value')

        # Plot Y
        axes[i, 1].plot(np.arange(Y.shape[1]), Y[idx, :], color='darkorange')
        axes[i, 1].set_title(f'Sample {idx} – Y (output)')
        axes[i, 1].set_xlabel('Feature index')
        axes[i, 1].set_ylabel('Value')

    plt.tight_layout()
    plt.show(block=True)  # Waits for the window to close before continuing