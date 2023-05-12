import os
import pickle
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from utils import *
import warnings
from pca import *
from experiments import experiments

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Numero de vecinos
k = 4

# Número de divisiones
k_fold_splits = 5


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


def base_test(df):
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

            for model in experiment["models"]:
                model_class = model['model']
                model_args = model['model_args']
                model_name = model['model_name']
                print(model_name)

                if experiment["id"] not in exp_results[f"k_{i}"]["results"]:
                    exp_results[f"k_{i}"]["results"][experiment["id"]] = {}

                for index, args in enumerate(model_args) or [None]:
                    print(args)

                    if not experiment["force"] and model_name in exp_results[f"k_{i}"]["results"][experiment["id"]] \
                            and index in exp_results[f"k_{i}"]["results"][experiment["id"]][model_name]:
                        continue

                    model_instance = model_class(**args)

                    tic = datetime.now()
                    y_pred = run_experiment(experiment, X_train, y_train, X_test, model_instance)
                    toc = datetime.now()

                    experiment_time_ms = (toc - tic).total_seconds() * 1000

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    ci = calculate_confidence_interval(X_test, y_test, y_pred, mse)

                    if model_name not in exp_results[f"k_{i}"]["results"][experiment["id"]]:
                        exp_results[f"k_{i}"]["results"][experiment["id"]][model_name] = {}

                    experiment_result = {
                        'model': experiment,
                        'MAE': mae,
                        'MSE': mse,
                        "ci": ci,
                        "fold": i,
                        "predictions": y_pred,
                        "experiment_time_ms": experiment_time_ms,
                        "args": args
                    }

                    exp_results[f"k_{i}"]["results"][experiment["id"]][model_name][index] = experiment_result

                    # Guardar resultados
                    with open("results.pkl", "wb") as outfile:
                        pickle.dump(exp_results, outfile)

    pick_best_experiment(exp_results)
