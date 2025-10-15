import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from os import makedirs
import pickle


def plot_result(out_true, out_pred, savefig=None, err_fun=None, scale_err=1):
    plt.cla()
    plt.figure(figsize=(8, 6))

    x_axis=np.arange(len(out_true))
    plt.plot(x_axis, out_true, label='True values', color='green', linewidth=2, linestyle='-', marker='o')
    plt.plot(x_axis, out_pred, label='Predicted values', color='red', linewidth=1, linestyle='-', marker="s", fillstyle='none')

    plt.title('True vs Predicted Values')
    plt.xlabel('P')
    plt.ylabel('Adsorption')
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    # plt.ylim(0,3500)

    if err_fun is not None:
        err_val = err_fun(out_true, out_pred)*scale_err
        err_name='MSE'
        if scale_err>1:
            err_name+='(*)'
        plt.plot(x_axis, out_pred, label=f'MSE = {err_val:.6f}', color='red', linestyle='-',linewidth=2)

    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))

    if savefig:
        makedirs(Path(savefig).parent, exist_ok=True)
        plt.savefig(savefig)
        # print(f"plot save at: {savefig}")
    else:
        plt.show()
    plt.close()


def plot_result__depr(out_axis, out_true, out_pred, savefig=None, err_fun=None):
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
    plt.close()


def mse(out_true, out_pred):
    return np.mean((out_true-out_pred)**2)


class ResultTracker:
    # keeps track of results with disc persistency
    def __init__(self, path):
        self.path = path
        self.results = self.__load_errors_file()

    def update(self, target, value):
        self.results[target] = value
        self.__dump()

    def __load_errors_file(self):
        if os.path.exists(self.path):
            results = pickle.load(open(self.path, 'rb'))
        else:
            os.makedirs(Path(self.path).parent, exist_ok=True)
            results = {}
        return results

    def __dump(self):
        pickle.dump(self.results, open(self.path, 'wb'), pickle.HIGHEST_PROTOCOL)

    def get(self, target):
        return self.results[target]

    def __contains__(self, target):
        return target in self.results