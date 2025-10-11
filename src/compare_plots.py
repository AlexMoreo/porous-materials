import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# --- Configura tus métodos y rutas base ---
methods = ["RF", "ff-128-256-128", "ff-64-128-128-64-PCA8"]
baseline_path = '../attachements_mail/results/reverse_from_N2'
methods_path  = '../results/plots/nitrogen'


# --- IDs de los problemas/curvas que quieres visualizar ---
selected_ids = [1, 9, 17, 26, 34, 38, 48, 53, 62, 67, 73, 80, 89, 93, 105]

# --- Crea figura y subplots ---
n_rows = len(selected_ids)
n_cols = len(methods)+1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))

# --- Cargar y mostrar cada imagen ---
for i, model_id in enumerate(selected_ids):
    for j, method in enumerate(['baseline']+methods):
        if j==0: #baseline method
            img_path = os.path.join(baseline_path, f"model{model_id:02d}.png")
        else:
            img_path = os.path.join(methods_path, method, f"model{model_id}.png")
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axes[i, j].imshow(img)
        else:
            axes[i, j].text(0.5, 0.5, "No image", ha='center', va='center', fontsize=12)
        axes[i, j].axis('off')

        # Column titles (solo en la primera fila)
        if i == 0:
            axes[i, j].set_title(method, fontsize=14, weight='bold')

        # Row labels (solo en la primera columna)
        if j == 0:
            axes[i, j].set_ylabel(f"Model {model_id}", fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('../results/plots/nitrogen/comparison.png',
    dpi=300,                # ← aumenta la resolución (300 o incluso 600)
    bbox_inches="tight",    # ← ajusta los márgenes al contenido
    pad_inches=0.1,         # ← opcional: deja un poco de borde
)
