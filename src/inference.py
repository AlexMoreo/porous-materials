import os
import pickle
from os.path import join

from data import load_test_data
from training import *
from regression import closest_to_mean
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Trains a neural regressor.")

    # 1. saved-dir (optional)
    parser.add_argument(
        "--saved",
        type=str,
        default="./neuralregressor",
        help="Directory containing the saved models (default: ./neuralregressor)."
    )

    # 2. input csv file (required)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the csv containing the input data."
    )

    # 2. output-dir (required)
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory where to dump the inferences."
    )

    return parser.parse_args()


def save_prediction(idx: np.ndarray, pred: np.ndarray, path: str):
    """
    Save predictions as a CSV file with columns var1, var2, ...

    Parameters:
        idx: (np.ndarray): Array of indexes (curves' names)
        pred (np.ndarray): Array of predictions with shape (n_instances, n_dimensions).
        path (str): Path to the CSV file where results will be saved.
    """
    # Create column names: var1, var2, ...
    n_features = pred.shape[1]
    col_names = [f"var{i+1}" for i in range(n_features)]

    # Build DataFrame
    df = pd.DataFrame(pred, columns=col_names, index=idx)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save to CSV
    df.to_csv(path, index=True)


if __name__ == '__main__':
    args = parse_args()

    model_path = args.saved
    out_dir = args.out

    idx, Gin = load_test_data(path=args.input, normalize=True, return_index=True)

    model_params = pickle.load(open(join(model_path, 'params.dict'), 'rb'))
    Gi_dim = model_params['Gi_dim']
    V_dim = model_params['V_dim']
    Go_dim = model_params['Go_dim']
    assert Gin.shape[1] == Gi_dim, \
        (f'Unexpected dimension for input gas. Expected by the model: {Gi_dim}, '
         f'found in {args.input}: {Gin.shape[1]}')

    print('Loading AE-type 1 over (z,y)')
    PAEzy = NewPAEzy().load_model(join(model_path, 'best_model_AEzy.pt'))

    print('Loading AE-type 1 over (Z,Y)')
    PAEZY = NewPAEZY(Gi_dim, V_dim, Go_dim).load_model(join(model_path, 'best_model_AEZY.pt'))

    print('Loading AE-type 2 over (z,y)')
    PAE2zy = NewPAE2zy().load_model(join(model_path, 'best_model_AE2zy.pt'))

    print('Loading AE-type 2 over (Z,Y)')
    PAE2ZY = NewPAE2ZY(Gi_dim, V_dim, Go_dim).load_model(join(model_path, 'best_model_AE2ZY.pt'))

    print("[Done]")

    Y1, _, Z1 = PAEzy.predict(Gin, return_XZ=True)
    Y2, _, Z2 = PAEZY.predict(Gin, return_XZ=True)
    Y3, _, Z3 = PAE2zy.predict(Gin, return_XZ=True)
    Y4, _, Z4 = PAE2ZY.predict(Gin, return_XZ=True)

    # out-gas is taken as an ensemble of 4 models, and returns the "closest to mean" curve
    Ypred = np.asarray([closest_to_mean(curves=list(preds)) for preds in zip(Y1, Y2, Y3, Y4)])

    # out-vol is taken from PAE2zy
    Zpred = Z3

    # saving predictions to file
    path_Gout = join(out_dir, 'Gout.csv')
    path_Vout = join(out_dir, 'Vout.csv')

    print(f"Saving predicted gas-out to {path_Gout}")
    save_prediction(idx, Ypred, path_Gout)

    print(f"Saving predicted gas-out to {path_Gout}")
    save_prediction(idx, Zpred, path_Vout)

    print("[Done]")



