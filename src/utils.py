import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import makedirs

def plot_result(out_axis, out_true, out_pred, savefig=None, err_fun=None):
    plt.cla()
    plt.figure(figsize=(8, 6))

    plt.plot(out_axis, out_true, label='True values', color='red', linewidth=2, linestyle='-', marker='o')
    plt.plot(out_axis, out_pred, label='Predicted values', color='green', linewidth=2, linestyle='-', marker="s")

    plt.title('True vs Predicted Values')
    plt.xlabel('P')
    plt.ylabel('Adsorption')
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.ylim(0,3500)

    if err_fun is not None:
        err_val = err_fun(out_true, out_pred)
        plt.plot(out_axis, out_pred, label=f'MSE = {err_val:.4f}', color='green', linestyle='-',linewidth=2)

    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))

    if savefig:
        makedirs(Path(savefig).parent, exist_ok=True)
        plt.savefig(savefig)
        print(f"plot save at: {savefig}")
    else:
        plt.show()


def mse(out_true, out_pred):
    return np.mean((out_true-out_pred)**2)