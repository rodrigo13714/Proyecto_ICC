import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
dataset = pd.read_csv("smogon.csv")

# Inicializar el modelo TF-IDF
vec = TfidfVectorizer(ngram_range=(2, 3))
x = vec.fit_transform(dataset["moves"])

# Reducir la dimensionalidad utilizando PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x.toarray())

# Inicializar el modelo KMeans
km = KMeans(n_clusters=18, n_init=10)
kmlista = km.fit_predict(x)

# Asignar etiquetas de cluster al DataFrame original
dataset["Cluster"] = kmlista

# Realizar una coincidencia parcial para identificar el cluster "Tipo fuego"
cluster_fuego = None
for i, phrase in enumerate(dataset["moves"]):
    if "burn the target" in phrase:
        cluster_fuego = kmlista[i]

# Cambiar el nombre del cluster que contiene la frase "burn the target" a "Tipo fuego"
if cluster_fuego is not None:
    dataset.loc[dataset["Cluster"] == cluster_fuego, "Cluster"] = "Tipo fuego"

# Mostrar el número total de tokens (elementos de su vocabulario)
num_tokens = len(vec.vocabulary_)
print("Numero total de tokens en la matriz:", num_tokens)

# Guardar el DataFrame con etiquetas de cluster
dataset.to_csv("Smogon_agrupados.csv")

# Crear un cluster plot con colores distintos para cada grupo de cluster
plt.figure(figsize=(10, 8))
for i in range(18):
    cluster_data = x_pca[kmlista == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')

plt.title("Cluster Plot")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.show()
