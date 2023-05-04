from numpy import std, sqrt
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from tabulate import tabulate
from xgboost import XGBRegressor
from utils import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from utils import get_distance_to_center, get_knn, get_knn_mean_price
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev

# Numero de vecinos
k = 4

# Numero de divisiones
k_fold_splits= 5

def base_test(df, columns_to_add):
    """
    Funcion para usar como base del experimento
    """
    # print("Dividimos el dataset en conjuntos de prueba")

    X = df.drop("price", axis=1)
    y = df["price"]

    # print("Dividimos el dataset en train y test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print("Aplico kfold")
    kf = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)

    all_results = []

    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), MLPRegressor(),
              GradientBoostingRegressor(), XGBRegressor(), DecisionTreeClassifier(), RandomForestClassifier(),
              MLPClassifier(), svm.SVC(), KNeighborsClassifier(n_neighbors=5)]

    model_names = ['Linear Regression', 'Decision Tree Regressor', 'RandomForestRegressor', 'MLPRegressor',
                   'GradientBoostingRegressor', 'XGBRegressor', " DecisionTreeClassifier", "RandomForestClassifier",
                   "MLPClassifier", "SVM", "KNN"]

    for i, (train_index, _) in enumerate(kf.split(X_train)):
        print("KFold", i)
        X_train_fold = X.iloc[train_index]

        # print("Transformamos los valores nulos a la mediana de la columna de X_train")

        X_train_fold, X_test_fold = replace_null_with_median(X_train_fold, X_test)

        # print("Añadimos variables nuevas")

        if len(columns_to_add) != 0:
            X_train_fold, X_test_fold = add_columns(columns_to_add, df, X_train_fold, X_test_fold, k)

        # print("Normalizamos los datos")

        scaler = StandardScaler()
        scaler.fit(X_train_fold)

        X_train_fold = scaler.transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)

        # print("Entrenamos los modelos")

        results = []

        for j, model in enumerate(models):
            model.fit(X_train_fold, y_train.iloc[train_index])
            y_pred = model.predict(X_test_fold)

            mae = mean_absolute_error(y_test, y_pred)

            results.append({'model': model_names[j], 'MAE': mae, "fold": i})

        all_results.append(results)

    # Crear tabla final

    for results in all_results:
        sorted_results = sorted(results, key=lambda x: x['MAE'])
        table = []
        for result in sorted_results:
            table.append([result['model'], result['MAE'], result['fold']])
        print("\n", tabulate(table, headers=['Model', 'MAE', 'Fold'], floatfmt='.2f',
                             tablefmt='orgtbl'))

    table = []
    for i, model_name in enumerate(model_names):
        maes = [result[i]['MAE'] for result in all_results]
        mean_mae = mean(maes)
        std_mae = stdev(maes)
        std_mean_mae = std(maes) / sqrt(len(maes))

        table.append([model_name, mean_mae, std_mae, mean_mae + std_mean_mae, mean_mae - std_mean_mae])

    table_sorted = sorted(table, key=lambda x: x[1])
    print("\n", tabulate(table_sorted, headers=['Model', 'MAE (Mean)', 'MAE (Std)', 'MAE+StdMean', 'MAE-StdMean'],
                         floatfmt='.2f', tablefmt='orgtbl'))


def test1(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset
    """
    print("Experimiento 1")
    base_test(df, [])


def test2(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset y añadiendo el precio medio de los
    k-vecinos
    """
    print("Experimiento 2")

    columns_to_add = ["neighbors", "neighbors_price_mean"]
    base_test(df, columns_to_add)

def test3(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset y añadiendo la distancia al centro de
    la ciudad
    """
    print("Experimiento 3")

    columns_to_add = ["distance_to_center"]
    base_test(df, columns_to_add)


def test4(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset, añadiendo la distancia al centro de
    la ciudad y añadiendo el precio medio de los k-vecinos
    """
    print("Experimiento 4")

    columns_to_add = ["neighbors", "neighbors_price_mean", "distance_to_center"]
    base_test(df, columns_to_add)


def test5(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores
    """
    print("Experimiento 5")

    base_test(df, [])


def test6(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores y añadiendo el precio medio de los
    k-vecinos
    """
    print("Experimiento 6")

    df = df.dropna()

    test2(df)


def test7(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores y añadiendo la distancia al centro de
    la ciudad
    """
    print("Experimiento 7")

    df = df.dropna()

    test3(df)

def test8(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores, añadiendo el precio medio de los
    k-vecinos y la distancia al centro de
    la ciudad
    """
    print("Experimiento 8")

    df = df.dropna()
    test4(df)



