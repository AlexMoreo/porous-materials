import os

import matplotlib.pyplot as plt
import numpy as np
import argparse
from data import load_predicted_data


def parse_args():
    parser = argparse.ArgumentParser(description="Trains a neural regressor.")

    # 1. prediction dir (required)
    parser.add_argument(
        "prediction",
        type=str,
        help="Directory containing the predictions."
    )

    # 2. output directory for saving the plots (optional, will be simply shown if not provided)
    parser.add_argument(
        "--plotdir",
        type=str,
        help="Path to the csv containing the input data."
    )

    # 3. cumulate output
    parser.add_argument(
        "--cumulate",
        action='store_true',
        help="Shows the cumulative curve for the volume (default: False)"
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    show = args.plotdir is None
    if not show: #if plots are to be saved, not showed
        os.makedirs(args.plotdir, exist_ok=True)

    path_gout = f'{args.prediction}/Gout.csv'
    path_vout = f'{args.prediction}/Vout.csv'
    test_names, Gout = load_predicted_data(path_gout, return_index=True)
    Vout = load_predicted_data(path_vout, return_index=False)
    if args.cumulate:
        Vout = np.cumsum(Vout, axis=-1)

    n_samples = Vout.shape[0]
    for idx in range(0, n_samples):
        fig, axes = plt.subplots(1, 2, figsize=(5*4, 2.5*4))

        # Plot 1
        axes[0].plot(np.arange(Gout.shape[1]), Gout[idx, :], color='steelblue', marker='o')
        axes[0].set_title(f'Gas-out')
        axes[0].set_xlabel('index')
        axes[0].set_ylabel('value')

        # Plot 2
        axes[1].plot(np.arange(Vout.shape[1]), Vout[idx, :], color='darkorange', marker='o')
        axes[1].set_title(f'Volume')
        axes[1].set_xlabel('index')
        axes[1].set_ylabel('value')

        fig.suptitle(f'{test_names[idx]}')

        plt.tight_layout()
        if show:
            plt.show(block=True)  # Waits for the window to close before continuing
        else:
            savepath = f'{args.plotdir}/{test_names[idx]}.pdf'
            print(f'saving plot to {savepath}')
            plt.savefig(savepath)
    print('[done]')