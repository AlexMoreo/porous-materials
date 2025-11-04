import os
import pickle

from data import load_both_data
from nn_modules import AE2, AE
from regression import NN3WayReg
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Trains a neural regressor.")

    # 1. gas-input (required)
    parser.add_argument(
        "--gas-input",
        type=str,
        required=True,
        help="Path to the gas input csv file"
    )

    # 2. gas-output (required)
    parser.add_argument(
        "--gas-output",
        type=str,
        required=True,
        help="Path to the gas output csv file"
    )

    # 3. save-dir (optional)
    parser.add_argument(
        "--save",
        type=str,
        default="./neuralregressor",
        help="Directory where to store the learned models (default: ./neuralregressor)."
    )

    # 4. excluded models (optional)
    parser.add_argument(
        "--exclude-models",
        nargs="+",  # list of string model names
        default=["model41", "model45"],
        help="List of models to exclude (default: model41, model45)."
    )

    return parser.parse_args()


def NewPAEzy(model_path=None, cuda=False):
    return NN3WayReg(
        model=AE(
            Xdim=10, Zdim=10, Ydim=10, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=10, Z_red=10, Y_red=10, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=25_000,
        checkpoint_dir=model_path, checkpoint_id='AEzy'
    )


def NewPAEZY(Gi_dim, V_dim, Go_dim, model_path=None, cuda=False):
    return NN3WayReg(
        model=AE(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000,
        checkpoint_dir=model_path, checkpoint_id='AEZY'
    )


def NewPAE2zy(model_path=None, cuda=False):
    return NN3WayReg(
        model=AE2(
            Xdim=10, Zdim=10, Ydim=10, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=10, Z_red=10, Y_red=10, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=25_000,
        checkpoint_dir=model_path, checkpoint_id='AE2zy'
    )


def NewPAE2ZY(Gi_dim, V_dim, Go_dim, model_path=None, cuda=False):
    return NN3WayReg(
        model=AE2(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000,
        checkpoint_dir=model_path, checkpoint_id='AE2ZY'
    )


if __name__ == '__main__':

    args = parse_args()

    model_path = args.save
    path_h2 = args.gas_output
    path_n2 = args.gas_input

    print(f'Input gas: {path_n2}')
    print(f'Output gas: {path_h2}')
    print(f'Save directoty: {model_path}')
    print(f'Exclude models: {args.exclude_models}')

    test_names, Vin, Gin, Gout = load_both_data(
        path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True,
        return_index=True, exclude_id=args.exclude_models
    )

    os.makedirs(model_path, exist_ok=True)

    Gi_dim = Gin.shape[1]
    V_dim = Vin.shape[1]
    Go_dim = Gout.shape[1]

    model_params = {
        'Gi_dim': Gi_dim,
        'V_dim': V_dim,
        'Go_dim': Go_dim
    }
    pickle.dump(model_params, open(os.path.join(model_path, 'params.dict'), 'wb'), pickle.HIGHEST_PROTOCOL)

    print('Training AE-type 1 over (z,y)')
    PAEzy = NewPAEzy(model_path, cuda=True).fit(Gin, Gout, Vin)

    print('Training AE-type 1 over (Z,Y)')
    PAEZY = NewPAEZY(Gi_dim, V_dim, Go_dim, model_path, cuda=True).fit(Gin, Gout, Vin)

    print('Training AE-type 2 over (z,y)')
    PAE2zy = NewPAE2zy(model_path, cuda=True).fit(Gin, Gout, Vin)

    print('Training AE-type 2 over (Z,Y)')
    PAE2ZY = NewPAE2ZY(Gi_dim, V_dim, Go_dim, model_path, cuda=True).fit(Gin, Gout, Vin)

    print("[Done]")

