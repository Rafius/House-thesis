from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor
from utils import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Numero de vecinos
k = 4

# Número de divisiones
k_fold_splits = 5


def base_test(df, columns_to_add, experiment):
    """
    Función para usar como base del experimento
    """
    # print("Dividimos el dataset en conjuntos de prueba")

    X = df.drop("price", axis=1)
    y = df["price"]

    # print("Dividimos el dataset en train y test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print("Aplico k-fold")
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

        X_train_fold, X_test_fold = normalize_data(X_train_fold, X_test_fold)

        results = []

        for j, model in enumerate(models):
            model.fit(X_train_fold, y_train.iloc[train_index])
            y_pred = model.predict(X_test_fold)

            mae = mean_absolute_error(y_test, y_pred)

            results.append({'model': model_names[j], 'MAE': mae, "fold": i})

        all_results.append(results)

    # Crear tabla final

    print_results(all_results, model_names, experiment)


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
