from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def pca_city(df):
    x_pca = pca(df)
    data = np.array(x_pca)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=df["price"], cmap='hot')

    plt.show()
    data = np.array(x_pca)

    plt.figure()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(data[:, 0], data[:, 1], data[:, 2], c=df["city"])
    axs[0, 1].scatter(data[df["city"] == 0, 0], data[df["city"] == 0, 1], data[df["city"] == 0, 2])
    axs[1, 0].scatter(data[df["city"] == 1, 0], data[df["city"] == 1, 1], data[df["city"] == 1, 2])
    axs[1, 1].scatter(data[df["city"] == 2, 0], data[df["city"] == 2, 1], data[df["city"] == 2, 2])

    axs[0, 0].set_title("todas las ciudades")
    axs[0, 1].set_title("ciudad 0")
    axs[1, 0].set_title("ciudad 1")
    axs[1, 1].set_title("ciudad 2")


def pca(df):
    # Crea una instancia de PCA
    pca = PCA()

    # Ajusta el PCA a las columnas seleccionadas
    x_pca = pca.fit_transform(df)

    variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance)

    # Graficar la varianza explicada acumulada
    plt.plot(cumulative_variance)
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada acumulada por los componentes principales')

    plt.xticks(np.arange(0, len(variance), 1))
    plt.show()

    return x_pca
