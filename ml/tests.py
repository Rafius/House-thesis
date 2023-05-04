from numpy import std, sqrt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from tabulate import tabulate
from xgboost import XGBRegressor
from utils import replace_null_with_median
from sklearn.tree import DecisionTreeRegressor
from utils import get_distance_to_center, get_knn, get_knn_mean_price
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev

# Numero de vecinos
k = 4


def test1(df):
    """
    Este experimento se realizará con todas las columnas
    """

    print("Dividimos el dataset en conjuntos de prueba")

    X = df.drop("price", axis=1)
    y = df["price"]

    print("Dividimos el dataset en train y test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Aplico kfold")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []

    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), MLPRegressor(),
              GradientBoostingRegressor(), XGBRegressor()]

    model_names = ['Linear Regression', 'Decision Tree Regressor', 'RandomForestRegressor', 'MLPRegressor',
                   'GradientBoostingRegressor', 'XGBRegressor']

    for i, (train_index, _) in enumerate(kf.split(X_train)):
        X_train_fold = X.iloc[train_index]

        print("Transformamos los valores nulos a la mediana de la columna de X_train")

        X_train_fold, X_test_fold = replace_null_with_median(X_train_fold, X_test)

        print("Añadimos variables nuevas")

        X_train_fold = get_distance_to_center(X_train_fold)
        X_test_fold = get_distance_to_center(X_test_fold)

        X_train_fold = get_knn(X_train_fold, X_train_fold, k)
        X_test_fold = get_knn(X_train_fold, X_test_fold, k)

        X_train_fold = get_knn_mean_price(df, X_train_fold)
        X_test_fold = get_knn_mean_price(df, X_test_fold)

        X_train_fold = X_train_fold.drop("neighbors", axis=1)
        X_test_fold = X_test_fold.drop("neighbors", axis=1)

        print("Normalizamos los datos")

        scaler = StandardScaler()
        scaler.fit(X_train_fold)

        X_train_fold = scaler.transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)

        print("Entrenamos los  modelo")

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
                   floatfmt='.2f',
                   tablefmt='orgtbl'))

