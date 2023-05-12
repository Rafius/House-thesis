from models import models

force = False

experiments = [
    {
        "force": force,
        "id": "Test1",
        "name": "Missing values with median",
        "models": models,
        "missing_values": {
            "type": "median",
            "options": {}
        },
    },
    {
        "force": force,
        "id": "Test2",
        "name": "Missing values with mean",
        "models": models,
        "columns_to_add": ["neighbors", "neighbors_price_mean"],
        "missing_values": {
            "type": "mean",
        },
    },
    {
        "force": force,
        "id": "Test3",
        "name": "Add new column neighbors mean price",
        "models": models,
        "columns_to_add": ["neighbors", "neighbors_price_mean"],
        "missing_values": {
            "type": "mean",
        },
    },
    {
        "force": force,
        "id": "Test4",
        "name": "Add new column distance to center",
        "models": models,
        "columns_to_add": ["distance_to_center"],
    },
    {
        "force": force,
        "id": "Test5",
        "name": "Add new column neighbors mean price and distance to center",
        "models": models,
        "columns_to_add": ["neighbors", "neighbors_price_mean", "distance_to_center"],
    },
    {
        "force": force,
        "id": "Test6",
        "name": "Remove empty values",
        "models": models,
        "remove_empty_values": True
    },
    {
        "force": force,
        "id": "Test7",
        "name": "Remove empty values and add new column neighbors mean price ",
        "models": models,
        "remove_empty_values": True,
        "columns_to_add": ["neighbors", "neighbors_price_mean"],
    },
    {
        "force": force,
        "id": "Test8",
        "name": "Remove empty values and add distance to center",
        "models": models,
        "remove_empty_values": True,
        "columns_to_add": ["distance_to_center"],
    },
    {
        "force": force,
        "id": "Test9",
        "name": "Remove empty values and and distance to center",
        "models": models,
        "remove_empty_values": True,
        "columns_to_add": ["neighbors", "neighbors_price_mean", "distance_to_center"],
    },
    {
        "force": force,
        "id": "Test10",
        "name": "Remove floor column",
        "models": models,
        "remove_empty_values": True,
        "columns_to_remove": ["floor"],
    },
    {
        "force": force,
        "id": "Test11",
        "name": "Remove usableArea column",
        "models": models,
        "remove_empty_values": True,
        "columns_to_remove": ["usableArea"],
    }
]
