import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from numpy import mean
from shapely.geometry import Point
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors
from math import radians
import math
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev
from numpy import std, sqrt
import os.path
from openpyxl import load_workbook
import scipy.stats as stats


# Numero de vecinos


def load_dataset(enable_prints=False):
    """
    Carga el dataset y elige las columnas
    """
    df = pd.read_csv('houses_v2.csv', sep=',', encoding='utf-8')

    features = ['price', 'city', 'builtArea', 'usableArea', 'bedrooms', 'bathrooms', 'floor', 'elevator',
                'houseHeating', 'terrace', 'swimmingPool', "latitude", "longitude"]

    df = df.loc[:, features]

    if enable_prints:
        # Cantidad de filas
        print(len(df))

        # Numero de columnas
        print(len(df.columns))

        print(df.columns)
        print(df.isnull().sum())

    # df = get_coordinates(df)

    df = df.dropna(subset=['latitude'])
    df = df.dropna(subset=['price'])

    df = transform_dataset(df, enable_prints)

    df = df.loc[:, features]

    return df


def transform_dataset(df, print_plots=False):
    """
    Transforma las variables a numéricas
    """

    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    df['builtArea'] = pd.to_numeric(df['builtArea'], errors='coerce')

    df['usableArea'] = pd.to_numeric(df['usableArea'], errors='coerce')

    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].replace('', 1)
    df['bedrooms'].fillna(1, inplace=True)

    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['bathrooms'] = df['bathrooms'].replace('', 1)
    df['bathrooms'].fillna(1, inplace=True)

    # boolean_dict = {True: 1, False: 0}
    #
    # df["elevator"] = df['elevator'].map(boolean_dict)
    # df["houseHeating"] = df['houseHeating'].map(boolean_dict)
    # df["terrace"] = df['terrace'].map(boolean_dict)
    # df["swimmingPool"] = df['swimmingPool'].map(boolean_dict)

    floor_dict = {'Principal': 1, 'Entresuelo': 0.5, 'Bajo': 0, 'Sótano': -1, 'Subsótano': -0.5, 'Más de 20': 21}

    df['floor'] = df['floor'].replace('(\d)(ª)', r'\1', regex=True)
    df['floor'] = df['floor'].replace(floor_dict)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')

    if print_plots:
        loc_geom = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        df = gpd.GeoDataFrame(df, geometry=loc_geom)

        df.plot()
        df.hist(bins=50, figsize=(15, 15))
        plt.show()

    return df


def replace_null_with_median(X_train, X_test=None):
    """
    Reemplaza los valores nulos con la mediana de la columna de X_train
    """

    median = X_train.median()

    X_train = X_train.fillna(median)

    if X_test is not None:
        X_test = X_test.fillna(median)

    return X_train, X_test


def replace_null_with_mean(X_train, X_test=None):
    """
    Reemplaza los valores nulos con la media de la columna de X_train
    """

    df_mean = X_train.mean()

    X_train = X_train.fillna(df_mean)

    if X_test is not None:
        X_test = X_test.fillna(df_mean)

    return X_train, X_test


def get_knn(X_train, X_test, k):
    """
    Obtiene los k-vecinos
    """
    latitudes = X_train['latitude'].apply(radians)
    longitudes = X_train['longitude'].apply(radians)

    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')

    nn_df = pd.DataFrame({'lat': latitudes, 'lon': longitudes})
    nn.fit(nn_df)
    distances, indices = nn.kneighbors(nn_df)

    neighbors_list = []

    for i in range(len(X_test)):
        neighbors = []
        for j in range(k):
            neighbor_idx = indices[i][j]

            # Evitamos que se añada a si mismo como vecino
            if neighbor_idx != i:
                neighbors.append(neighbor_idx)

        neighbors_list.append(neighbors)

    X_test["neighbors"] = neighbors_list

    return X_test


def get_knn_mean_price(dataset, y_train):
    """
    Crea una variable con la media de precio de los vecinos
    """
    for i, row in dataset.iterrows():
        prices = []
        for neighbor in row.neighbors:
            prices.append(y_train.iloc[neighbor])

        dataset.loc[i, 'neighbors_price_mean'] = np.mean(prices)

    return dataset


def get_distance_to_center(dataset):
    """
    Obtiene la distancia de cada vivienda al centro de su ciudad
    """
    cities = {
        'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
        'Madrid': {'lat': 40.4168, 'lon': -3.7038},
        'Málaga': {'lat': 36.7213, 'lon': -4.4213},
    }

    dataset['distance_to_center'] = None

    for i, row in dataset.iterrows():
        lat, lon = row['latitude'], row['longitude']
        for j, coordinates in enumerate(cities.values()):
            if row['city'] == j:
                distance = haversine(lat, lon, coordinates['lat'], coordinates['lon'])
                dataset.at[i, 'distance_to_center'] = distance

    return dataset


def haversine(lat1, lon1, lat2, lon2):
    """
    Función de la fórmula de Haversine
    """
    R = 6371  # Radio de la Tierra en kilómetros

    # Conversión de grados a radianes
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Diferencia de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


def clean_address(address):
    """
    Limpia la dirección
    """
    to_remove = ["Piso en alquiler en", "Apartamento en alquiler en", "Ático en alquiler en",
                 "Dúplex en alquiler en", "Casa adosada en alquiler en", "Chalet en alquiler en",
                 "Casa unifamiliar en alquiler", "Estudio en alquiler en", "Loft en alquiler en",
                 "Casa pareada en alquiler en", "Chalet unifamiliar en alquiler en", "Chalet adosado en alquiler en",
                 "Casa en alquiler en",
                 "Finca rústica en alquiler en", "Chalet pareado en alquiler en "]

    for substring in to_remove:
        address = address.replace(substring, "")

    address = address.split("cerca")[0]

    return address


def get_coordinates(df):
    """
    Obtiene las coordenadas de una calle
    """
    url = 'http://api.positionstack.com/v1/forward'

    for index, row in df.iterrows():
        row = df.iloc[index]
        address = clean_address(row['title'])
        city = row["city"]
        region = city + ', España'
        params = {
            'access_key': 'e8ab79b0edab30e9b58f2773fa1caf5e',
            'query': address,
            'limit': 1,
            "region": region
        }

        response = requests.get(url, params=params)
        location = response.json()['data']

        if len(location) > 0:
            location = location[0]
            if not location: continue
            df.at[index, 'latitude'] = location['latitude']
            df.at[index, 'longitude'] = location['longitude']
            print(index, location["latitude"], location["longitude"], address, region)
        else:
            print(f"Could not find location for address: {index}{address} {region}")

    return df


def normalize_data(X_train, X_test=None):
    """
    Normaliza los datos de entrada
    """
    scaler = StandardScaler().fit(X_train)

    X_train_normalize = scaler.transform(X_train)
    X_test_normalize = None

    if X_test is not None:
        X_test_normalize = scaler.transform(X_test)

    return X_train_normalize, X_test_normalize


def add_columns(columns_to_add, X_train, X_test, y_train, k):
    """
    Añade columnas al dataset
    """
    for column in columns_to_add:
        if column == "distance_to_center":
            X_train = get_distance_to_center(X_train)
            X_test = get_distance_to_center(X_test)

        if column == "neighbors":
            X_train = get_knn(X_train, X_train, k)
            X_test = get_knn(X_train, X_test, k)

        if column == "neighbors_price_mean":
            X_train = get_knn_mean_price(X_train, y_train)
            X_test = get_knn_mean_price(X_test, y_train)
            X_train = X_train.drop("neighbors", axis=1)
            X_test = X_test.drop("neighbors", axis=1)

    return X_train, X_test


def estimate_houses_to_buy_rent_prices():
    """
    Estima los precios de alquiler de los pisos en venta
    """

    df_buy = pd.read_csv('houses_to_buy_v2.csv')

    df_buy = df_buy.dropna(subset=['price'])

    df_buy = df_buy.dropna(subset=['latitude'])

    features = ["price", 'city', 'builtArea', "usableArea", "floor", 'bedrooms', 'bathrooms', 'elevator',
                'houseHeating', 'terrace', 'swimmingPool', "latitude", "longitude"]

    df_buy_final = df_buy.loc[:, features]

    df_buy_final = transform_dataset(df_buy_final)

    df_buy_final = get_distance_to_center(df_buy_final)

    df_buy_final, _ = replace_null_with_median(df_buy_final)

    df_buy_final, _ = normalize_data(df_buy_final)

    # Usar el mejor modelo para predecir

    # predictions = dt.predict(df_buy_final)
    #
    # nuevos_datos_con_predicciones = np.concatenate((df_buy, predictions.reshape(-1, 1)), axis=1)
    # columnas = list(df_buy.columns) + ["rentPrice"]
    # df_buy = pd.DataFrame(nuevos_datos_con_predicciones, columns=columnas)
    #
    # df_buy.to_json('houses_to_buy.json', orient='records')


def clean_missing_values(X_train, X_test, options):
    """
    Limpia las valores nulos del dataset usando la media o la mediana
    """
    if options["type"] == "median":
        return replace_null_with_median(X_train, X_test)

    elif options["type"] == "mean":
        return replace_null_with_mean(X_train, X_test)


histogram_ranges = [
    {"min_range": 0, "max_range": 500},
    {"min_range": 501, "max_range": 1000},
    {"min_range": 1001, "max_range": 2000},
    {"min_range": 2001, "max_range": 5000},
    {"min_range": 5001, "max_range": 10000},
    {"min_range": 10001, "max_range": 100000}
]


def show_results(exp_results):
    """
    Recibe el resultado de los experimentos, y seleccionad el de menor MAE
    """
    #
    # best_experiment = get_best_experiment(exp_results)
    # results_by_model = get_results_per_model(exp_results)
    # boxplot_mae_per_k(exp_results)
    for ranges in histogram_ranges:
        min_range = ranges["min_range"]
        max_range = ranges["max_range"]
        histogram_abs_price_error(exp_results, min_range, max_range)
        histogram_abs_percentage_error(exp_results, min_range, max_range)

    # Dos tipos de boxplot un boxplot por modelo, donde cada caja es un conjunto de parametros distintos solo para los que tiene parametros
    # Histograma con abs del precio real - precio estimado para ver el error, y ver errores entre 0-50€, 50€-100€
    # Histograma con abs del precio real % precio estimado para ver el error, y ver errores entre 0-50€, 50€-100€


def get_best_experiment(exp_results):
    """
    Recibe el resultado de los experimentos, y seleccionad el de menor MAE
    """
    experiments = []
    for exp_result in exp_results.values():
        for result in exp_result["results"].values():
            for model_name, model in result.items():
                for model_info in model.values():
                    experiment = {
                        "model_name": model_name,
                        "model_info": model_info
                    }
                    experiments.append(experiment)

    sorted_experiments = sorted(experiments, key=lambda x: x["model_info"]["mae"])
    best_experiment = sorted_experiments[0]
    print("numero de experimentos: ", len(experiments))
    print(best_experiment["model_info"]["mae"])
    print(best_experiment["model_name"])

    return best_experiment


def get_results_per_model(exp_results):
    results_by_models = {}
    for k in exp_results.values():
        for result in k["results"].values():
            for model_name, model_results in result.items():
                if model_name not in results_by_models:
                    results_by_models[model_name] = []

                for mode_result in model_results.values():
                    results_by_models[model_name].append(mode_result)

    for model_name, result in results_by_models.items():
        results_by_models[model_name] = sorted(result, key=lambda x: x["mae"])

    return results_by_models


def boxplot_mae_per_k(exp_results):
    mae_per_k = []
    for k_data in exp_results.values():
        temp_array = []

        for result in k_data["results"].values():
            for model_name, model_results in result.items():

                for mode_result in model_results.values():
                    temp_array.append(mode_result["mae"])

        mae_per_k.append(np.mean(temp_array))

    plt.boxplot(mae_per_k)

    plt.title('Media de mae por k-fold')
    plt.xlabel('Mae')
    plt.ylabel('Valores')

    plt.show()


def histogram_abs_price_error(exp_results, min_range, max_range):
    errors = []

    for exp_result in exp_results.values():
        for result in exp_result["results"].values():
            for model_name, model in result.items():
                for model_info in model.values():
                    for i, price in enumerate(model_info["y_test"]):
                        if price < min_range or price > max_range: continue

                        prediction = model_info["predictions"][i]
                        test = abs(prediction - price)
                        errors.append(test)

    plt.hist(errors, rwidth=0.85)

    plt.title(f'Error medio de los experimentos entre {min_range}€ y {max_range}€')
    plt.xlabel('Error medio')
    plt.ylabel('Frecuencia')
    plt.show()


def histogram_abs_percentage_error(exp_results, min_range, max_range):
    errors = []

    for exp_result in exp_results.values():
        for result in exp_result["results"].values():
            for model_name, model in result.items():
                for model_info in model.values():
                    for i, price in enumerate(model_info["y_test"]):
                        if price < min_range or price > max_range: continue

                        prediction = model_info["predictions"][i]
                        test = get_error_percentage(price, prediction)
                        errors.append(test)

    plt.hist(errors, rwidth=0.85)

    plt.title(f'Error porcentual de los experimentos entre {min_range}€ y {max_range}€')
    plt.xlabel('Error porcentual')
    plt.ylabel('Frecuencia')
    plt.show()


def calculate_confidence_interval(X_test, y_test, mse):
    """
    Sirve para calcular el intervalo de confianza
    """
    n = len(y_test)
    p = X_test.shape[1]
    se = np.sqrt(mse / (n - p - 1))

    # Calcular el intervalo de confianza del 95% para las predicciones
    alpha = 0.05
    t = stats.t.ppf(1 - alpha / 2, n - p - 1)
    ci = t * se * np.sqrt(1 + 1 / n)

    return ci


def get_error_percentage(y_test, y_pred):
    return abs(y_pred - y_test) / y_test * 100


def discretize_price(price):
    return np.floor_divide(price, 10) * 10
