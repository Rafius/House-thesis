from load_dataset import load_dataset, get_knn
from tests import test1
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

# Numero de vecinos
k = 4
enable_prints = False

print("Obtenemos el dataset con las variables que vamos a usar, ya transformadas")
df = load_dataset(k, enable_prints)

print("Dividimos el dataset en conjuntos de prueba")


print("Normalizamos los datos")


print("Experimentos para performance")
# test1(df)


print("end")

# Notas
# Buscar siempre en X_train, coger puntos knn, coger los 3 mas cercanos y hacer un promedio del precio para una columnas nueva, usar numpy https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/, crear metodo que te devuelva los 3 vecinos
#
# todo se aprende y ajusta de train
#
# hacer normalizacion con train
#
# para comparar normalizar todo como si fuera train y hace un fit del modelo
#
# carga de datos
#
# limpieza de datos
#
# mantener mediana en valores nulos
#
# normalizar
#
# probar modelos
#
# tener un metodo que tenga train test,
#
# Hacer un experimento quitando todos los datos vacios.
#
# Hacer un intento intentando correguir los datos, media, mediana.
#
# Hacer un intento haciendo media de propiedades que faltan con los vecinos.
#
# Hacer un intento eliminando columnas y otra con filas.
#
# Hacer una prueba de discretizar el precio +-50
#
# En los experimentos >>> print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std())) 0.98 accuracy with a standard deviation of 0.02
#
# https://towardsdatascience.com/leveraging-geolocation-data-for-machine-learning-essential-techniques-192ce3a969bc
