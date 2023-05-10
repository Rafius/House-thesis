import os
import pickle
from datetime import datetime
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor
from utils import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings
from pca import *

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Numero de vecinos
k = 4

# Número de divisiones
k_fold_splits = 5

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


def run_experiment(experiment, X_train, y_train, X_test, model_instance):
    mv_options = {"type": "median", "options": {}}

    if "missing_values" in experiment:
        mv_options.update(experiment["missing_values"])

    X_train, X_test = clean_missing_values(X_train, X_test, options=mv_options)

    # print("Añadimos variables nuevas")

    if "columns_to_add" in experiment and (experiment["columns_to_add"]) != 0:
        X_train, X_test = add_columns(experiment["columns_to_add"], X_train, X_test, y_train, k)

    if "columns_to_remove" in experiment and (experiment["columns_to_remove"]) != 0:
        X_train = X_train.drop(experiment["columns_to_remove"], axis=1)
        X_test = X_test.drop(experiment["columns_to_remove"], axis=1)

    if "remove_empty_values" in experiment and experiment["remove_empty_values"]:
        X_train = X_train.dropna()
        X_test = X_test.dropna()

    # print("Normalizamos los datos")

    X_train, X_test = normalize_data(X_train, X_test)

    # pca(X_test)
    # pca(X_train)

    model_instance.fit(X_train, y_train)
    return model_instance.predict(X_test)


def base_test(df, experiments):
    """
    Función para usar como base del experimento
    """
    # print("Dividimos el dataset en conjuntos de prueba")

    X = df.drop("price", axis=1)
    y = df["price"]

    exp_results = {}
    if os.path.exists("results.pkl"):
        with open("results.pkl", "rb") as infile:
            exp_results = pickle.load(infile)

    # print(exp_results)

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
            print(experiment["id"])
            if not experiment["force"] and experiment["id"] in exp_results[f"k_{i}"]["results"]:
                # print("Skipped experiment", experiment)
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
                                                                                  "experiment_time_ms":
                                                                                      experiment_time_ms}

                # Guardar resultados
                with open("results.pkl", "wb") as outfile:
                    pickle.dump(exp_results, outfile)

    # print_results(all_results, model_names, experiment)
    pick_best_experiment(exp_results)
