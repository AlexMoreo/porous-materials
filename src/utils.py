import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import makedirs

def plot_result(out_axis, out_true, out_pred, savefig=None, err_fun=None):
    plt.figure(figsize=(8, 6))

    plt.plot(out_axis, out_true, label='True values', color='blue', linewidth=2, marker='+')
    plt.plot(out_axis, out_pred, label='Predicted values', color='red', linestyle='--', linewidth=2, marker="+")

    plt.title('True vs Predicted Values')
    plt.xlabel('X Axis')
    plt.ylabel('Values')

    if err_fun is not None:
        err_val = err_fun(out_true, out_pred)
        plt.plot(out_axis, out_pred, label=f'MSE = {err_val:.4f}', color='red', linestyle='--',linewidth=2)

    plt.legend()

    if savefig:
        makedirs(Path(savefig).parent, exist_ok=True)
        plt.savefig(savefig)
        print(f"plot save at: {savefig}")
    else:
        plt.show()


def mse(out_true, out_pred):
    return np.mean((out_true-out_pred)**2)