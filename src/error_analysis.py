from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd

from utils import load_method_errors


# Ejemplo: nombres de los modelos
test_idx = [f'model{idx}' for idx in np.arange(111)+1]
method_names = ["R3-XY", "R3-ZY", "R3-y", "R3-Y", "R3-zy"]
error_path_prefix = f'../results/errors'
E = np.asarray([load_method_errors(f'{error_path_prefix}/{method}.pkl', test_idx) for method in method_names]).T
n_samples, n_models = E.shape

E = pd.DataFrame(E, columns=method_names)
E.head()

# Normalización por columnas (cada modelo)
E_scaled = StandardScaler().fit_transform(E)

# PCA a 2D
pca = PCA(n_components=2)
E_pca = pca.fit_transform(E_scaled)

# Modelo con menor error por ejemplo
best_model_idx = np.argmin(E.values, axis=1)
colors = plt.cm.tab10(best_model_idx / len(method_names))

plt.figure(figsize=(6,5))
plt.scatter(E_pca[:,0], E_pca[:,1], c=colors)
plt.title("Ejemplos agrupados por el modelo que mejor predice")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(E_scaled)

plt.figure(figsize=(6,5))
plt.scatter(E_pca[:,0], E_pca[:,1], c=clusters, cmap="viridis")
plt.title("Clustering de ejemplos según patrón de error")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()

cluster_profiles = pd.DataFrame(E_scaled, columns=method_names)
cluster_profiles["cluster"] = clusters

import pandas as pd
mean_profiles = cluster_profiles.groupby("cluster")[method_names].mean()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 3))
sns.heatmap(mean_profiles, annot=True, cmap="coolwarm", center=0)
plt.title("Perfil medio de errores por cluster")
plt.tight_layout()
plt.savefig("cluster_profiles.png", dpi=300)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# suponiendo que X tiene las features de entrada
# X_train, X_test, y_train, y_test = train_test_split(X, clusters, test_size=0.2, random_state=0)

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# print(classification_report(y_test, clf.predict(X_test)))

