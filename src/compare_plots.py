import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from tqdm import tqdm

# --- methods and paths ---
# gas = 'nitrogen'
gas = 'hydrogen'

baseline_path = None
# methods = ['RF', 'RFy', 'RFxy', 'R3-XYZ', 'R3-XYZ-L0', 'R3-Xyz-L0',
#            'R3-XY','R3-ZY','R3-Y','R3-Xyz','R3-Xy','R3-zy','R3-y']
methods = ['R3-XYZ-L0', 'R3-Xyz-L0']


# --- selected models to display ---
# selected_ids = [1, 9, 17, 26, 34, 38, 48, 53, 62, 67, 73, 80, 89, 93, 105]
# selected_ids = [35,35]
selected_ids = list(range(10,40))
# methods = ['R3-XY']+[f'R3-XY-v{i+1}' for i in range(10)]
methods_path = '../results/plots'
gas_suffix = '-Gout'  # gas out representation

# --- Crea figura y subplots ---
n_rows = len(selected_ids)
n_cols = len(methods)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), constrained_layout=False)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

# --- Cargar y mostrar cada imagen ---
for i, model_id in tqdm(enumerate(selected_ids), desc='generating plots', total=len(selected_ids)):
    for j, method in enumerate(methods):
        if j==0 and baseline_path is not None: #baseline method
            img_path = os.path.join(baseline_path, f"model{model_id:02d}.png")
        else:
            img_path = os.path.join(methods_path, method, f"model{model_id}{gas_suffix}.png")
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

path_out = f'../results/{gas}{gas_suffix}_var_comparison.png'
print(f'plots generated, saving images in {path_out}')
plt.tight_layout(pad=0.)
plt.savefig(path_out,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
print('[Done!]')
