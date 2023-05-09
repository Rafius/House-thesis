import os
import pickle
from datetime import datetime
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor
from utils import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Numero de vecinos
k = 4

# Número de divisiones
k_fold_splits = 5


def clean_missing_values(X_train, X_test, options):
    if options["type"] == "median":
        return replace_null_with_median(X_train, X_test)

    elif options["type"] == "mean":
        return None, None
        # X_train, X_test = replace_null_with_mean(X_train, X_test)


def run_experiment(experiment, X_train, y_train, X_test, model_instance):
    mv_options = {"type": "median", "options": {}}

    if "missing_values" in experiment:
        mv_options.update(experiment["missing_values"])

    X_train, X_test = clean_missing_values(X_train, X_test, options=mv_options)

    # print("Añadimos variables nuevas")

    # if len(columns_to_add) != 0:
    #     X_train, X_test = add_columns(columns_to_add, df, X_train, X_test, k)

    # print("Normalizamos los datos")

    X_train, X_test = normalize_data(X_train, X_test)

    model_instance.fit(X_train, y_train)
    return model_instance.predict(X_test)


def base_test(df):
    """
    Función para usar como base del experimento
    """
    # print("Dividimos el dataset en conjuntos de prueba")

    X = df.drop("price", axis=1)
    y = df["price"]

    # print("Dividimos el dataset en train y test")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        {
            "model": LinearRegression,
            "model_args": {},
            "model_name": 'LinearRegression'
        },
        {
            "model": DecisionTreeRegressor,
            "model_args": {},
            "model_name": 'DecisionTreeRegressor'
        },
        {
            "model": RandomForestRegressor,
            "model_args": {},
            "model_name": 'RandomForestRegressor'
        },
        {
            "model": MLPRegressor,
            "model_args": {},
            "model_name": 'MLPRegressor'
        },
        {
            "model": GradientBoostingRegressor,
            "model_args": {},
            "model_name": 'GradientBoostingRegressor'
        },
        {
            "model": XGBRegressor,
            "model_args": {},
            "model_name": 'XGBRegressor'
        },
        {
            "model": DecisionTreeClassifier,
            "model_args": {},
            "model_name": 'DecisionTreeClassifier'
        },
        {
            "model": RandomForestClassifier,
            "model_args": {},
            "model_name": 'RandomForestClassifier'
        },
        {
            "model": MLPClassifier,
            "model_args": {},
            "model_name": 'MLPClassifier'
        },
        {
            "model": svm.SVC,
            "model_args": {},
            "model_name": 'SVM'
        },
        {
            "model": KNeighborsClassifier,
            "model_args": {},
            "model_name": 'KNN'
        }
    ]

    experiments = [
        {
            "force": True,
            "id": "Test1",
            "name": "Knn with all models",
            "models": models,
            "options": {},
            "missing_values": {
                "type": "median",
                "options": {}
            },
        },
        # {
        #     "force": False,
        #     "id": "MLPC1",
        #     "name": "MLPClassifier",
        #     "options": {
        #         "model": MLPClassifier,
        #         "model_args": {
        #         }
        #     },
        # }
    ]

    exp_results = {}
    if os.path.exists("results.pkl"):
        with open("results.pkl", "rb") as infile:
            exp_results = pickle.load(infile)

    print(exp_results)

    # print("Aplico k-fold")
    kf = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("KFold", i)
        if f"k_{i}" not in exp_results:
            exp_results[f"k_{i}"] = {"train_index": train_index, "test_index": test_index, "results": {}}

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        for experiment in experiments:

            if not experiment["force"] and experiment["id"] in exp_results[f"k_{i}"]["results"]:
                print("Skipped experiment", experiment)
                continue

            for model in experiment["models"]:
                model_class = model['model']
                model_args = model['model_args']
                model_name = model['model_name']
                model_instance = model_class(**model_args)

                tic = datetime.now()
                y_pred = run_experiment(experiment, X_train, y_train, X_test, model_instance)
                toc = datetime.now()

                experiment_time_ms = (toc - tic).total_seconds() * 1000

                mae = mean_absolute_error(y_test, y_pred)

                if experiment["id"] not in exp_results[f"k_{i}"]["results"]:
                    exp_results[f"k_{i}"]["results"][experiment["id"]] = {}

                exp_results[f"k_{i}"]["results"][experiment["id"]][model_name] = {'model': experiment, 'MAE': mae,
                                                                                  "fold": i,
                                                                                  "predictions": y_pred,
                                                                                  "experiment_time_ms": experiment_time_ms}

                # Guardar resultados
                with open("results.pkl", "wb") as outfile:
                    pickle.dump(exp_results, outfile)

    # print_results(all_results, model_names, experiment)


def test1(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset
    """
    print("Experiment 1")
    base_test(df, [], "experiment1")


def test2(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset y añadiendo el precio medio de los
    k-vecinos
    """
    print("Experiment 2")

    columns_to_add = ["neighbors", "neighbors_price_mean"]
    base_test(df, columns_to_add, "experiment2")


def test3(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset y añadiendo la distancia al centro de
    la ciudad
    """
    print("Experiment 3")

    columns_to_add = ["distance_to_center"]
    base_test(df, columns_to_add, "experiment3")


def test4(df):
    """
    Este experimento se realizará con todas las columnas originales del dataset, añadiendo la distancia al centro de
    la ciudad y añadiendo el precio medio de los k-vecinos
    """
    print("Experiment 4")

    columns_to_add = ["neighbors", "neighbors_price_mean", "distance_to_center"]
    base_test(df, columns_to_add, "experiment4")


def test5(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores
    """
    print("Experiment 5")

    df = df.dropna()
    print()

    base_test(df, [], "experiment5")


def test6(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores y añadiendo el precio medio de los
    k-vecinos
    """
    print("Experiment 6")

    df = df.dropna()

    columns_to_add = ["neighbors", "neighbors_price_mean"]
    base_test(df, columns_to_add, "experiment6")


def test7(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores y añadiendo la distancia al centro de
    la ciudad
    """
    print("Experiment 7")

    df = df.dropna()

    columns_to_add = ["distance_to_center"]
    base_test(df, columns_to_add, "experiment7")


def test8(df):
    """
    Este experimento se realizará con todas las filas que tengan todos los valores, añadiendo el precio medio de los
    k-vecinos y la distancia al centro de
    la ciudad
    """
    print("Experiment 8")

    df = df.dropna()
    columns_to_add = ["neighbors", "neighbors_price_mean", "distance_to_center"]
    base_test(df, columns_to_add, "experiment8")


def test9(df):
    """
    Este experimento se realizará borrando la columna floor
    la ciudad
    """
    print("Experiment 9")

    df = df.drop("floor", axis=1)
    base_test(df, [], "experiment9")


def test10(df):
    """
    Este experimento se realizará borrando la columna usableArea
    la ciudad
    """
    print("Experiment 10")

    df = df.drop("usableArea", axis=1)
    base_test(df, [], "experiment10")
