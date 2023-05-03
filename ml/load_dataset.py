import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors
from math import radians
import math


# Carga el dataset y elige las columnas
def load_dataset(k, enable_prints=False):
    df = pd.read_csv('houses_v2.csv', sep=',', encoding='utf-8')

    features = ['price', 'city', 'builtArea', 'usableArea', 'bedrooms', 'bathrooms', 'floor', 'elevator',
                'houseHeating',
                'terrace', 'swimmingPool', "latitude", "longitude"]

    df = df.loc[:, features]

    if enable_prints:
        # Cantidad de filas
        print(len(df))

        # Numero de columnas
        print(len(df.columns))

        print(len(df.columns))
        print(df.columns)
        print(df.isnull().sum())

    # df = get_coordinates(df)

    df = df.dropna(subset=['latitude'])
    df = df.dropna(subset=['price'])

    df = get_distance_to_center(df)

    df = get_knn(df, k)

    df = get_knn_mean_price(df)


    df = transform_dataset(df, enable_prints)

    return df


# Transforma las variables a numéricas
def transform_dataset(df, print_plots):
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    df['builtArea'] = pd.to_numeric(df['builtArea'], errors='coerce')

    df['usableArea'] = pd.to_numeric(df['usableArea'], errors='coerce')

    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].replace('', 1)

    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['bathrooms'] = df['bathrooms'].replace('', 1)

    floor_dict = {'Principal': 1, 'Entresuelo': 0.5, 'Bajo': 0, 'Sótano': -1, 'Subsótano': -0.5, 'Más de 20': 21}

    df['floor'] = df['floor'].replace('(\d)(ª)', r'\1', regex=True)
    df['floor'] = df['floor'].replace(floor_dict)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')

    if print_plots:
        loc_geom = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        df = gpd.GeoDataFrame(df, geometry=loc_geom)

        # Plot the GeoDataFrame
        df.plot()
        df.hist(bins=50, figsize=(15, 15))
        plt.show()

    return df


# Obtiene los k-vecinos
def get_knn(df, k):
    latitudes = df['latitude'].apply(radians)
    longitudes = df['longitude'].apply(radians)

    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')

    nn_df = pd.DataFrame({'lat': latitudes, 'lon': longitudes})
    nn.fit(nn_df)
    distances, indices = nn.kneighbors(nn_df)

    neighbors_list = []

    for i in range(len(df)):
        neighbors = []
        for j in range(k):
            neighbor_idx = indices[i][j]

            # Evitamos que se añada a si mismo como vecino
            if neighbor_idx != i:
                neighbors.append(neighbor_idx)

        neighbors_list.append(neighbors)

    df["neighbors"] = neighbors_list

    return df


# Crea una variable con la media de precio de los vecinos
def get_knn_mean_price(df):
    for i, row in df.iterrows():
        prices = []
        for neighbor in row.neighbors:
            prices.append(df.iloc[neighbor].price)

        df.loc[i, 'neighbors_price_mean'] = np.mean(prices)

    return df


# Obtiene la distancia de cada vivienda al centro de su ciudad
def get_distance_to_center(df):
    cities = {
        'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
        'Madrid': {'lat': 40.4168, 'lon': -3.7038},
        'Málaga': {'lat': 36.7213, 'lon': -4.4213},
    }

    df['distanceToCenter'] = None

    for i, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        for j, coordinates in enumerate(cities.values()):
            if row['city'] == j:
                distance = haversine(lat, lon, coordinates['lat'], coordinates['lon'])
                df.at[i, 'distanceToCenter'] = distance

    return df


# Función de la fórmula de Haversine
def haversine(lat1, lon1, lat2, lon2):
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


# Limpia la dirección
def clean_address(address):
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


# Obtiene las coordenadas de una calle
def get_coordinates(df):
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

    df.to_csv('houses_to_buy_v2.csv', index=False)

    return df