global_hyperparameters = {
    # Training parameters
    "epochs": 5000,
    "batch_size": 32,
    "pool_size": 1024,
    # Model architecture parameters
    "n_channels": 16,
    "hidden_channels": 128,
    "fire_rate": .5,
}

configs = [
    {
        "training_mode": "growing",
        "filter": "sobel",
        "dataset": "BloodMNIST",
        "image_index": 0,
    },
    {
        "training_mode": "persisting",
        "filter": "sobel",
        "dataset": "BloodMNIST",
        "image_index": 0,
    },
    {
        "training_mode": "regenerating",
        "filter": "sobel",
        "dataset": "BloodMNIST",
        "image_index": 0,
    },
    {
        "training_mode": "growing",
        "filter": "sobel",
        "dataset": "RetinaMNIST",
        "image_index": 0,
    },
    {
        "training_mode": "persisting",
        "filter": "sobel",
        "dataset": "RetinaMNIST",
        "image_index": 0,
    },
    {
        "training_mode": "regenerating",
        "filter": "sobel",
        "dataset": "RetinaMNIST",
        "image_index": 0,
    },
    {
        "training_mode": "growing",
        "filter": "sobel",
        "dataset": "DermaMNIST",
        "image_index": 4,
    },
    {
        "training_mode": "persisting",
        "filter": "sobel",
        "dataset": "DermaMNIST",
        "image_index": 4,
    },
    {
        "training_mode": "regenerating",
        "filter": "sobel",
        "dataset": "DermaMNIST",
        "image_index": 4,
    },
    {
        "training_mode": "growing",
        "filter": "sobel",
        "dataset": "PathMNIST",
        "image_index": 6,
    },
    {
        "training_mode": "persisting",
        "filter": "sobel",
        "dataset": "PathMNIST",
        "image_index": 6,
    },
    {
        "training_mode": "regenerating",
        "filter": "sobel",
        "dataset": "PathMNIST",
        "image_index": 6,
    },
]

def add_global_hyperparameters(config: dict) -> dict:
    for key, val in global_hyperparameters.items():
        if not key in config.keys():
            config[key] = val
    return config

configs = list(map(add_global_hyperparameters, configs))
