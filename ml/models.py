from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

models = [
    {
        "model": LogisticRegression,
        "model_args": [
            {},
            {
                "max_iter": 1000,
                "solver": "liblinear"
            },
            {
                "max_iter": 2000,
                "solver": "liblinear"
            },
            {
                "max_iter": 5000,
                "solver": "liblinear"
            },
            {
                "max_iter": 1000,
                "n_jobs": -1,
                "solver": "newton-cg"
            },
            {
                "max_iter": 2000,
                "n_jobs": -1,
                "solver": "newton-cg"
            },
            {
                "max_iter": 5000,
                "n_jobs": -1,
                "solver": "newton-cg"
            },
            {
                "max_iter": 1000,
                "n_jobs": -1,
                "solver": "lbfgs"
            },
            {
                "max_iter": 2000,
                "n_jobs": -1,
                "solver": "lbfgs"
            },
            {
                "max_iter": 5000,
                "n_jobs": -1,
                "solver": "lbfgs"
            },
            {
                "max_iter": 1000,
                "n_jobs": -1,
                "solver": "sag"
            },
            {
                "max_iter": 2000,
                "n_jobs": -1,
                "solver": "sag"
            },
            {
                "max_iter": 5000,
                "n_jobs": -1,
                "solver": "sag"
            },
        ],
        "model_name": 'LogisticRegression'
    },
    {
        "model": LinearRegression,
        "model_args": [{}],
        "model_name": 'LinearRegression'
    },
    {
        "model": DecisionTreeRegressor,
        "model_args": [{}],
        "model_name": 'DecisionTreeRegressor'
    },
    {
        "model": RandomForestRegressor,
        "model_args": [{}],
        "model_name": 'RandomForestRegressor'
    },
    {
        "model": MLPRegressor,
        "model_args": [{}],
        "model_name": 'MLPRegressor'
    },
    {
        "model": GradientBoostingRegressor,
        "model_args": [{}],
        "model_name": 'GradientBoostingRegressor'
    },
    {
        "model": XGBRegressor,
        "model_args": [{}],
        "model_name": 'XGBRegressor'
    },
    {
        "model": DecisionTreeClassifier,
        "model_args": [{}],
        "model_name": 'DecisionTreeClassifier'
    },
    {
        "model": RandomForestClassifier,
        "model_args": [{}],
        "model_name": 'RandomForestClassifier'
    },
    {
        "model": MLPClassifier,
        "model_args": [{}],
        "model_name": 'MLPClassifier'
    },
    {
        "model": svm.SVC,
        "model_args": [{}],
        "model_name": 'SVM'
    },
    {
        "model": KNeighborsClassifier,
        "model_args": [{}],
        "model_name": 'KNN'
    }
]
