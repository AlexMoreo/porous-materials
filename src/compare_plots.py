import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# --- methods and paths ---
# gas = 'nitrogen'
gas = 'hydrogen'

# if gas=='nitrogen':
#     methods = ['baseline', "RF", "RF-PCA12-PCAin12", "ff-128-256-128", "ff-64-128-128-64-PCA12", "ff-64-128-128-64-PCA12-0", "ff-128-256-512-256-128-PCA12-0"]
#     baseline_path = '../attachements/reverse_from_N2'
#     methods_path  = f'../results/plots/{gas}'
# else: # no baseline
#     methods = ["RF", "ff-128-256-128", "ff-128-256-128-y10"]
#     baseline_path = None
#     methods_path  = f'../results/plots/{gas}'

baseline_path = None
methods = ["R3-3L128", "R3-3L128-PCAZY8", "R3-3L128-PCAZY8-dr"]


# --- selected models to display ---
selected_ids = [1, 9, 17, 26, 34, 38, 48, 53, 62, 67, 73, 80, 89, 93, 105]
methods_path = '../results/plots'
gas_suffix = '-Gout'  # gas out representation

# --- Crea figura y subplots ---
n_rows = len(selected_ids)
n_cols = len(methods)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))

# --- Cargar y mostrar cada imagen ---
for i, model_id in enumerate(selected_ids):
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

plt.tight_layout()
# plt.show()
plt.savefig(f'../results/{gas}{gas_suffix}_comparison.png',
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
