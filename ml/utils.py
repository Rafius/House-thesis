import pickle

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from numpy import mean
from shapely.geometry import Point
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from math import radians
import math
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev
from numpy import std, sqrt
import os.path
from openpyxl import load_workbook
import scipy.stats as stats
import re
from sklearn import preprocessing

le = preprocessing.LabelEncoder()


def load_dataset(enable_prints):
    """
    Carga el dataset y elige las columnas
    """
    df = pd.read_csv('houses_v2.csv', sep=',', encoding='utf-8')

    # df = get_postal_code(df)
    # if "latitude" not in df:
    #     df = get_coordinates(df)

    features = ['price', 'city', 'builtArea', 'usableArea', 'bedrooms', 'bathrooms', 'floor', 'elevator',
                'houseHeating', 'terrace', 'swimmingPool', "lat", "lon"]

    df = df.loc[:, features]

    if enable_prints:
        # Cantidad de filas
        print(len(df))

        # Numero de columnas
        print(len(df.columns))

        print(df.columns)
        print(df.isnull().sum())

    df = df.dropna(subset=['lat'])
    df = df.dropna(subset=['price'])

    df = transform_dataset(df, enable_prints)

    df = df.loc[:, features]

    return df


def transform_dataset(df, enable_prints):
    """
    Transforma las variables a numéricas
    """

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    # df = df.drop(df[df['price'] > 5000].index)

    df['builtArea'] = pd.to_numeric(df['builtArea'], errors='coerce')

    df['usableArea'] = pd.to_numeric(df['usableArea'], errors='coerce')

    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].replace('', 1)
    df['bedrooms'].fillna(1, inplace=True)

    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['bathrooms'] = df['bathrooms'].replace('', 1)
    df['bathrooms'].fillna(1, inplace=True)

    boolean_dict = {True: 1, False: 0}

    # df["elevator"] = df['elevator'].map(boolean_dict)
    # df["houseHeating"] = df['houseHeating'].map(boolean_dict)
    # df["terrace"] = df['terrace'].map(boolean_dict)
    # df["swimmingPool"] = df['swimmingPool'].map(boolean_dict)
    # df["city"] = le.fit_transform(df["city"])

    floor_dict = {'Principal': 1, 'Entresuelo': 0.5, 'Bajo': 0, 'Sótano': -1, 'Subsótano': -0.5, 'Más de 20': 21}

    df['floor'] = df['floor'].replace('(\d)(ª)', r'\1', regex=True)
    df['floor'] = df['floor'].replace(floor_dict)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')

    if enable_prints:
        # print_geolocation_plot(df)
        print_dataset_histograms(df)

    return df


def get_postal_code(df):
    postal_codes = pd.read_excel('codigos-postales.xlsx')

    provincias_deseadas = ['Barcelona', 'Málaga', 'Madrid']
    postal_codes_filtered = postal_codes[postal_codes['provincia'].isin(provincias_deseadas)]

    for index, house in df.iterrows():
        location = house["location"]
        result = re.search(r'\((.*?)\)', location)

        if result:
            location = result.group(1)

            if "." in location:
                location = re.split(r'\.', location)[-1].strip()

            if "Capital" in location:
                location = re.sub(r'\bCapital\b', '', location).strip()

        postal_code_row = postal_codes_filtered[postal_codes_filtered["poblacion"] == location]

        if not postal_code_row.empty:
            postal_code = postal_code_row.iloc[0]["codigopostalid"]
            latitude = postal_code_row.iloc[0]["lat"] / 1000000000
            longitude = postal_code_row.iloc[0]["lon"] / 1000000000
            df.at[index, 'postal_code'] = postal_code
            df.at[index, 'lat'] = latitude
            df.at[index, 'lon'] = longitude

        # else:
        #     print("not found", location)

    df.to_csv('houses_to_buy.csv', index=False)
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
    latitudes = X_train['lat'].apply(radians)
    longitudes = X_train['lon'].apply(radians)

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
        lat, lon = row['lat'], row['lon']
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
        city = row["city"] + row["location"]
        region = city + ', España'
        params = {
            'access_key': '32b0f1dfeb0dcdecaba6fa5353098981',
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

        df.to_csv('houses_v0.csv', index=False)

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
    # Elegimos las caracteristicas

    features = ["price", 'builtArea', "usableArea", "floor", 'bedrooms', 'bathrooms', 'elevator',
                'houseHeating', 'terrace', 'swimmingPool', "lat", "lon"]

    # Entrenamos el modelo
    df = pd.read_csv('houses_v2.csv')
    df = df.drop(df[df['price'] > 5000].index)
    df = df.loc[:, features]

    df = transform_dataset(df, False)
    df, _ = replace_null_with_median(df)

    X = df.loc[:, features]
    X = X.drop("price", axis=1)
    X, _ = normalize_data(X)
    y = df["price"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    df_buy = pd.read_csv('houses_to_buy_v2.csv')

    df_buy_final = df_buy.loc[:, features]

    # df_buy = get_postal_code(df_buy)

    df_buy_final = transform_dataset(df_buy_final, False)

    # df_buy.to_csv('houses_to_buy_v2.csv', index=False)

    df_buy_final, _ = replace_null_with_median(df_buy_final)

    df_buy_final = df_buy_final.drop("price", axis=1)
    df_buy_final, _ = normalize_data(df_buy_final)

    # Usar el mejor modelo para predecir
    predictions = rf.predict(df_buy_final)

    df_buy["rentPrice"] = predictions

    df_buy = df_buy.dropna(subset=['price'])
    df_buy.to_json('houses_to_buy.json', orient='records')


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
    Recibe el resultado de los experimentos, y pinta los datos en distintos gráficos
    """
    #
    # best_experiments = get_best_experiment(exp_results)
    # mae_per_model = get_mae_per_model(exp_results)
    # boxplot_mae_per_model(mae_per_model)
    # mae_per_experiment = get_mae_per_experiment(exp_results)
    # boxplot_mae_per_experiment(mae_per_experiment)
    # print_mae_percentage_vs_real_price(exp_results)
    # print_predicted_price_vs_real_price(exp_results)
    get_best_and_worse_houses(exp_results)

    # print_real_price_vs_predicted_price(best_experiments[0])
    # print_real_price_vs_predicted_price(best_experiments[1])
    # print_real_price_vs_predicted_price(best_experiments[2])

    # for ranges in histogram_ranges:
    #     min_range = ranges["min_range"]
    #     max_range = ranges["max_range"]
    #     histogram_abs_price_error(exp_results, min_range, max_range)
    #     histogram_abs_percentage_error(exp_results, min_range, max_range)


def print_mae_percentage_vs_real_price(exp_results):
    for k_name, k in exp_results.items():
        for result in k["results"].values():
            for model_name, model_results in result.items():
                for model_result in model_results.values():
                    y_test = model_result["y_test"]
                    mae_percentage = get_error_percentage(y_test, model_result["predictions"])
                    plt.scatter(mae_percentage, y_test)
                    plt.xlabel("Error porcentual")
                    plt.ylabel("Precios reales")
                    plt.title("Comparación de precios reales y porcentaje de error")
                    plt.show()


def print_predicted_price_vs_real_price(exp_results):
    for k_name, k in exp_results.items():
        for result in k["results"].values():
            for model_name, model_results in result.items():
                for model_result in model_results.values():
                    y_test = model_result["y_test"]
                    predictions = model_result["predictions"]
                    plt.scatter(predictions, y_test)
                    plt.xlabel("Predicciones")
                    plt.ylabel("Precios reales")
                    plt.title(f"Comparación de precios reales y predicciones {model_name}")
                    plt.show()


def get_best_experiment(exp_results):
    """
    Recibe el resultado de los experimentos, y seleccionad el de menor MAE %
    """
    experiments = []
    for exp_result in exp_results.values():
        for result in exp_result["results"].values():
            for model_name, model in result.items():
                for model_info in model.values():
                    experiment = {
                        "model_name": model_name,
                        "model_info": model_info,
                        "mae_percentage": np.mean(get_error_percentage(model_info["y_test"], model_info["predictions"]))
                    }
                    experiments.append(experiment)

    sorted_experiments = sorted(experiments, key=lambda x: x["mae_percentage"])
    best_experiment = sorted_experiments[0]
    print("numero de experimentos: ", len(experiments))
    print(best_experiment["mae_percentage"])
    print(best_experiment["model_name"])

    return sorted_experiments


def get_mae_per_model(exp_results):
    mae_per_model = {}
    for k_name, k in exp_results.items():
        for result in k["results"].values():
            for model_name, model_results in result.items():
                if model_name not in mae_per_model:
                    mae_per_model[model_name] = {}

                for model_result in model_results.values():
                    experiment_id = model_result["model"]["id"]

                    if experiment_id not in mae_per_model[model_name]:
                        mae_per_model[model_name][experiment_id] = []

                    mae_percentage = np.mean(get_error_percentage(model_result["y_test"], model_result["predictions"]))

                    model_result["mae_percentage"] = mae_percentage
                    mae_per_model[model_name][experiment_id].append(model_result)

    final_mae_per_model = {}
    for model_name, experiments in mae_per_model.items():
        if model_name not in final_mae_per_model:
            final_mae_per_model[model_name] = {}

        for experiment_name, experiment in experiments.items():
            for iteration in experiment:
                if experiment_name not in final_mae_per_model[model_name]:
                    final_mae_per_model[model_name][experiment_name] = []
                final_mae_per_model[model_name][experiment_name].append(iteration["mae_percentage"])

    return final_mae_per_model


def boxplot_mae_per_model(mae_per_model):
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20, 20))

    axs = axs.flatten()
    for i, (model_name, model_data) in enumerate(mae_per_model.items()):
        values = []

        for value in model_data.values():
            values.extend([value])

        axs[i].boxplot(values)
        axs[i].set_title(model_name)
        axs[i].set_ylim(20, 50)

    plt.tight_layout()

    plt.show()


def get_mae_per_experiment(exp_results):
    mae_per_experiment = {}
    for k_name, k in exp_results.items():
        for result in k["results"].values():
            for model_name, model_results in result.items():
                for model_result in model_results.values():
                    experiment_id = model_result["model"]["id"]
                    if experiment_id not in mae_per_experiment:
                        mae_per_experiment[experiment_id] = {}

                    if model_name not in mae_per_experiment[experiment_id]:
                        mae_per_experiment[experiment_id][model_name] = []

                    mae_percentage = np.mean(get_error_percentage(model_result["y_test"], model_result["predictions"]))

                    model_result["mae_percentage"] = mae_percentage
                    mae_per_experiment[experiment_id][model_name].append(model_result)

    for experiment_name, experiment in mae_per_experiment.items():
        for model_name, model in experiment.items():
            mae_per_experiment[experiment_name][model_name] = sorted(model, key=lambda x: x["mae_percentage"])

    final_mae_per_experiment = {}
    for experiment_id, experiments in mae_per_experiment.items():
        for model_name, experiment in experiments.items():
            for iteration in experiment:
                if experiment_id not in final_mae_per_experiment:
                    final_mae_per_experiment[experiment_id] = {}

                if model_name not in final_mae_per_experiment[experiment_id]:
                    final_mae_per_experiment[experiment_id][model_name] = []

                final_mae_per_experiment[experiment_id][model_name].append(iteration["mae_percentage"])

    return final_mae_per_experiment


def boxplot_mae_per_experiment(mae_per_experiment):
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20, 20))

    axs = axs.flatten()
    for i, (model_name, model_data) in enumerate(mae_per_experiment.items()):
        values = []

        for value in model_data.values():
            values.extend([value])

        axs[i].boxplot(values)
        axs[i].set_title(model_name)
        axs[i].set_ylim(20, 50)

        valores_x = ['LogisticRegression', 'LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor',
                     'MLPRegressor', 'GradientBoostingRegressor', 'XGBRegressor',
                     'DecisionTreeClassifier', 'RandomForestClassifier', 'MLPClassifier', 'SVM', 'KNN']

        axs[i].set_xticklabels(valores_x, rotation=15, ha='right')

    plt.tight_layout()

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
                        errors.append(abs(prediction - price))

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
                        errors.append(get_error_percentage(price, prediction))

    plt.hist(errors, rwidth=0.85)

    plt.title(f'Error porcentual de los experimentos entre {min_range}€ y {max_range}€')
    plt.xlabel('Error porcentual')
    plt.ylabel('Frecuencia')
    plt.show()


def calculate_confidence_interval(X_test, y_test, mae):
    """
    Sirve para calcular el intervalo de confianza
    """
    n = len(y_test)
    p = X_test.shape[1]
    se = np.sqrt(mae / (n - p - 1))

    # Calcular el intervalo de confianza del 95% para las predicciones
    alpha = 0.05
    t = stats.t.ppf(1 - alpha / 2, n - p - 1)
    ci = t * se * np.sqrt(1 + 1 / n)

    return ci


def get_error_percentage(y_test, y_pred):
    return abs(y_pred - y_test) / y_test * 100


def discretize_price(price):
    return np.floor_divide(price, 10) * 10


def print_geolocation_plot(df):
    BBox = (df.lon.min(), df.lon.max(),
            df.lat.min(), df.lat.max())

    ruh_m = plt.imread('D:/Dropbox/UOC/en curso/tfg/houses-tesis/ml/mapa.png')
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df.lon, df.lat, zorder=1, alpha=0.2, c='b', s=10)
    ax.set_title('Plotting Spatial Data on Spain Map')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(ruh_m, zorder=0, extent=BBox, aspect='equal')
    plt.show()


def print_dataset_histograms(df):
    columns = ["price", "builtArea", "usableArea"]
    data = df[columns]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, columna in enumerate(columns):
        axs[i].hist(data[columna], bins=50)
        axs[i].set_xlabel("Valor")
        axs[i].set_ylabel("Frecuencia")
        axs[i].set_title(f"Histograma de {columna}")

    plt.tight_layout()
    plt.show()

    columns = ["bedrooms", "bathrooms", "floor"]
    data = df[columns]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, columna in enumerate(columns):
        axs[i].hist(data[columna], bins=50)
        axs[i].set_xlabel("Valor")
        axs[i].set_ylabel("Frecuencia")
        axs[i].set_title(f"Histograma de {columna}")

    plt.tight_layout()
    plt.show()

    elevator = df["elevator"].value_counts(normalize=True)
    houseHeating = df["houseHeating"].value_counts(normalize=True)
    terrace = df["terrace"].value_counts(normalize=True)
    swimmingPool = df["swimmingPool"].value_counts(normalize=True)


def get_best_and_worse_houses(exp_results):
    df = pd.read_csv('houses_v2.csv', sep=',', encoding='utf-8')

    final_results = [{} for _ in range(len(df))]

    if os.path.exists("final_results.pkl"):
        with open("final_results.pkl", "rb") as infile:
            final_results = pickle.load(infile)

    if not final_results[0]:
        results = [[] for _ in range(len(df))]

        for k_name, k in exp_results.items():
            for result in k["results"].values():
                for model_name, model_results in result.items():
                    for model_result in model_results.values():
                        percentage_error = get_error_percentage(model_result["y_test"], model_result["predictions"])

                        # error = abs(model_result["y_test"] - model_result["predictions"])
                        # ranking = error.rank(method='min')
                        ranking = percentage_error.rank(method='min')

                        for id, rank in ranking.items():
                            # results[id].append({"rank": rank, "predictions": model_result["predictions"]})
                            results[id].append({"rank": rank, "error": percentage_error[id]})

        for index, result_row in enumerate(results):
            if not result_row: continue
            errors = []
            ranks = []
            for result in result_row:
                errors.append(result["error"])
                ranks.append(result["rank"])

            final_results[index] = {
                "id": index,
                "mean_ranking": np.mean(ranks),
                "lowest_error": np.min(errors),
                "mean_error": np.mean(errors),
                "highest_error": np.max(errors)
            }

        with open("final_results.pkl", "wb") as outfile:
            pickle.dump(final_results, outfile)

    # Filtrar y eliminar los diccionarios con valores vacíos para "mean_ranking"
    final_results = [diccionario for diccionario in final_results if
                     "mean_ranking" in diccionario and diccionario["mean_ranking"]]

    sorted_by_ranking = sorted(final_results, key=lambda x: x["mean_ranking"])

    number_houses = 10

    bests = sorted_by_ranking[:number_houses]
    #
    # for best in bests:
    #     test = df.iloc[best["id"]]
    #     print(test)

    worsts = sorted_by_ranking[-number_houses:]
    for j, worst in enumerate(worsts):
        test2 = df.iloc[worst["id"]]
        print(j)
        print(test2)


def print_real_price_vs_predicted_price(experiment):
    print(experiment["mae_percentage"])

    model_name = experiment["model_name"]
    experiment_name = experiment["model_info"]["model"]["id"]
    plt.scatter(experiment["model_info"]["predictions"], experiment["model_info"]["y_test"])

    x_line = np.linspace(0, 20000, 100)
    y_line = np.linspace(0, 20000, 100)

    plt.plot(x_line, y_line, color='red', linestyle='--')
    plt.xlabel("Precio estimado")
    plt.ylabel("Precios reales")
    plt.title(f"Comparación de precios reales y estimados {model_name} {experiment_name}")
    plt.xlim(0, 20000)
    plt.ylim(0, 20000)
    plt.show()


def calculate_euclidean_distance():
    df = pd.read_csv('houses_v2.csv', sep=',', encoding='utf-8')

    for index,row in df.iterrows():
        lat1 = row["latitude"]
        lat2 = row["lat"]
        lon1 = row["longitude"]
        lon2 = row["lon"]

        distance = haversine(lat1, lon1, lat2, lon2)
        df.at[index, "distance"] = distance











