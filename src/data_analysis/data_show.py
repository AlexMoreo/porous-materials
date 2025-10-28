import matplotlib.pyplot as plt
import numpy as np

from data import load_data, load_both_data

path_n2 = f'../../data/training/dataset_for_nitrogen.csv'
path_h2 = f'../../data/training/dataset_for_hydrogen.csv'
Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

block_size = 5
n_samples = Vin.shape[0]

to_show = [32, 41, 45]
for start in range(0, n_samples, block_size):
    end = min(start + block_size, n_samples)
    current_block = end - start

    fig, axes = plt.subplots(current_block, 3, figsize=(10, 2.5 * current_block))

    # Ensure axes is always 2D
    if current_block == 1:
        axes = np.array([axes])

    for i in range(current_block):
        idx = start + i

        # Plot 1
        axes[i, 0].plot(np.arange(Gin.shape[1]), Gin[idx, :], color='steelblue')
        axes[i, 0].set_title(f'Model {idx+1}: N2')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')

        # Plot 2
        axes[i, 1].plot(np.arange(Vin.shape[1]), Vin[idx, :], color='darkorange')
        axes[i, 1].set_title(f'Model {idx+1}: Vol')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')

        # Plot 3
        axes[i, 2].plot(np.arange(Gout.shape[1]), Gout[idx, :], color='darkgreen')
        axes[i, 2].set_title(f'Model {idx+1}: H2')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')

    plt.tight_layout()
    plt.show(block=True)  # Waits for the window to close before continuing