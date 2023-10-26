import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


data = pd.read_csv("Smogon_agrupados.csv")
data = data.drop("Tipo", axis=1)
#borrar el indice
data.drop(data.columns[0], axis=1, inplace=True)

pca=PCA(n_components=10)
pca.fit(data)
x_pca = pca.transform(data)

print("Número de filas y columnas del DataFrame original:")
print(data.shape)
print("Número de filas y columnas del DataFrame de componentes principales:")
print(x_pca.shape)
cabeceras = ["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8", "PCA9", "PCA10"]
tablaPCA = pd.DataFrame(data=x_pca, columns=cabeceras)
print(tablaPCA)

km = KMeans(n_clusters=18, n_init=40)
clusters_lista = km.fit_predict(tablaPCA)
print(clusters_lista)

tablaPCA["cluster"] = clusters_lista