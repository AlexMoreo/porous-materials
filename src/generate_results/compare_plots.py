import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from tqdm import tqdm

from data import load_both_data

# --- methods and paths ---
# gas = 'nitrogen'
# gas = 'hydrogen'

baseline_path = None
# methods = ['RF', 'RFy', 'RFxy', 'R3-XYZ', 'R3-XYZ-L0', 'R3-Xyz-L0',
#            'R3-XY','R3-ZY','R3-Y','R3-Xyz','R3-Xy','R3-zy','R3-y']
# methods = (
#         [f'R3-XY{i}' for i in range(5)]
#            # + [f'R3-ZY{i}' for i in range(5)]
#            # + [f'R3-Y{i}' for i in range(5)]
#          + [f'R3-y{i}' for i in range(5)]
# )
# def var_size(method): return [f'{method}-small', method, f'{method}-big']
methods = [#'RFY', #'1NN',  #'AE-ZY',
           # 'R3-XYZ', 'R3-XYZ-tiny', 'R3-XYZ-xl', 'R3-Xyz', 'R3-Xyz-s',
           # 'R3-YZ', 'R3-YZ-tiny', 'R3-YZ-xl', 'R3-yz', #'R3-yz-s',
           #'AE-ZY', 'AE-zy',
           'PAE2zy', 'PAE2ZY', 'PAExy', 'PAEZY', 'EnsRedo'
           ]


# --- selected models to display ---
from experiments import load_data
path_h2 = '../../data/training/dataset_for_hydrogen.csv'
path_n2 = '../../data/training/dataset_for_nitrogen.csv'
test_names, *_ = load_both_data(
    path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True,
    return_index=True, exclude_id=['model41', 'model45']
)
all_ids = test_names
# selected_ids = [1, 9, 17, 26, 34, 38, 48, 53, 62, 67, 73, 80, 89, 93, 105]
# selected_ids = [3,10,35,37,68,70]
# selected_ids = [1,2,3,4]
# selected_ids = list(range(10,40))
# methods = ['R3-XY']+[f'R3-XY-v{i+1}' for i in range(10)]
methods_path = '../../results/plots'
for gas_suffix in ['Vin', 'Gout']:
    if gas_suffix == 'Gout':
        methods = ['RFY'] + methods
    print(f'Generating plots for {gas_suffix}')
    gas_path = f'../../results/plots-{gas_suffix}'
    os.makedirs(gas_path, exist_ok=True)

    n_ids = len(all_ids)
    batch_size = 5  # creates grids of 20 x n_models
    n_batches = (n_ids//batch_size) + (1 if n_ids%batch_size>1 else 0)

    for batch_i in range(n_batches):

        selected_ids = all_ids[(batch_i*batch_size):((batch_i+1)*batch_size)]

        # --- Crea figura y subplots ---
        n_rows = len(selected_ids)
        n_cols = len(methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), constrained_layout=False)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        from_, to_ = selected_ids[0], selected_ids[-1]

        # --- Cargar y mostrar cada imagen ---
        for i, model_id in tqdm(enumerate(selected_ids), desc=f'generating plots {from_}-{to_}', total=len(selected_ids)):
            for j, method in enumerate(methods):
                img_path = os.path.join(methods_path, method, f"{model_id}-{gas_suffix}.png")
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    axes[i, j].imshow(img)
                else:
                    axes[i, j].text(0.5, 0.5, "No image", ha='center', va='center', fontsize=12)
                axes[i, j].axis('off')

                # column titles (only first)
                if i == 0:
                    axes[i, j].set_title(method, fontsize=14, weight='bold')

                # row labels (only first)
                if j == 0:
                    axes[i, j].set_ylabel(f"Model {model_id}", fontsize=12)


        path_out = f'{gas_path}/plot_comparison_{from_}-{to_}.png'

        print(f'plots generated, saving images in {path_out}...', end='')
        plt.tight_layout(pad=0.)
        plt.savefig(path_out,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        print('[Done]')

    print('[Done!]')
