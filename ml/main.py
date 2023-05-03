from load_dataset import load_dataset
from tests import test1


# Numero de vecinos
k = 4
enable_prints = False

# Obtenemos el dataset con las variables que vamos a usar, ya transformadas
df = load_dataset(k, enable_prints)

# Dividimos el dataset en conjuntos de prueba



# Normalizamos los datos

# df = normalize_data()

# Experimentos para performance
# test1(df)


print(df)

print("end")

#
# Buscar siempre en X_train, coger puntos knn, coger los 3 mas cercanos y hacer un promedio del precio para una columnas nueva, usar numpy https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/, crear metodo que te devuelva los 3 vecinos
#
# todo se aprende y ajusta de train
#
# hacer normalizacion con train
#
# para comprar normalizar todo como si fuera train y hace un fit del modelo
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
# Xavier Baró Solé4:16 PM
# He compartit un fitxer de Jam amb la reunió: https://jamboard.google.com/d/1FY29kw6oDV71sWROp5tFxPKzLnfLcTEKGUpNi9i-8lE/edit?usp=meet_whiteboard
# He compartit un fitxer de Jam amb la reunió: https://jamboard.google.com/d/1ol_UxCaNbu_2hxxBWo2Dh0m9YzNNVj4HRy5-vj86OCU/edit?usp=meet_whiteboard
# You4:18 PM
# rafius93@gmail.com
# Xavier Baró Solé4:19 PM
# https://jamboard.google.com/d/1ol_UxCaNbu_2hxxBWo2Dh0m9YzNNVj4HRy5-vj86OCU/edit?usp=sharing
# Xavier Baró Solé4:37 PM
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# Xavier Baró Solé5:49 PM
# X => A | B | C
# X_train = B + C    X_test = A
# X_train = A + C    X_test = B
# X_train = A + B    X_test = C
# Xavier Baró Solé5:50 PM
# https://scikit-learn.org/stable/modules/cross_validation.html
# Xavier Baró Solé5:52 PM
# E = 100 +- 5