import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

dataset = pd.read_csv("smogon.csv")
print(dataset)

km = KMeans(18, n_init=10)

vec = TfidfVectorizer(ngram_range=(2, 3))
x = vec.fit_transform(dataset["moves"])
z = sorted(vec.vocabulary_.keys())
tf = pd.DataFrame(data=x.toarray(), columns=z)

km = KMeans(n_clusters=18, n_init=10)
kmlista = km.fit_predict(tf)
tf["Tipo"] = kmlista
print(vec.vocabulary_)
print(kmlista)

# Mostrar el número total de tokens (elementos de su vocabulario)
num_tokens = len(vec.vocabulary_)
print("\nNumero total de tokens en la matriz:", num_tokens,"\n")

print(tf)
tf.to_csv("Smogon_agrupados.csv")
