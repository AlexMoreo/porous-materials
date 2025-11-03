import os
import pickle
from os.path import join
from training import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Trains a neural regressor.")

    # 1. saved-dir (optional)
    parser.add_argument(
        "--saved",
        type=str,
        default="./neuralregressor",
        help="Directory containing the saved models (default: ./neuralregressor)."
    )

    # 2. output-dir (required)
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory where to dump the inferences."
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_path = args.saved
    out_dir = args.out

    model_params = pickle.load(open(join(model_path, 'params.dict'), 'rb'))
    os.makedirs(out_dir, exist_ok=True)

    Gi_dim = model_params['Gi_dim']
    V_dim = model_params['V_dim']
    Go_dim = model_params['Go_dim']

    print('Training AE-type 1 over (z,y)')
    PAEzy = NewPAEzy().load_model(join(model_path, 'AEzy'))

    print('Training AE-type 1 over (Z,Y)')
    PAEZY = NewPAEZY(Gi_dim, V_dim, Go_dim).load_model(join(model_path, 'AEZY'))

    print('Training AE-type 2 over (z,y)')
    PAE2zy = NewPAE2zy().load_model(join(model_path, 'AE2zy'))

    print('Training AE-type 2 over (Z,Y)')
    PAE2ZY = NewPAE2ZY(Gi_dim, V_dim, Go_dim).load_model(join(model_path, 'AE2ZY'))

    print("[Done]")

