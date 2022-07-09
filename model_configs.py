global_hyperparameters = {
    # Training parameters
    "epochs": 5000,
    "batch_size": 48,
    "pool_size": 1024,
    # Model architecture parameters
    "n_channels": 16,
    "hidden_channels": 128,
    "fire_rate": .5,
}

configs = [
    {
        "training_mode": "regeneration",
        "filter": "sobel",
        "dataset": "DermaMNIST",
        "image_index": 12,
    },
    {
        "training_mode": "regeneration",
        "filter": "sobel",
        "dataset": "PathMNIST",
        "image_index": 8,
    },
]

def add_global_hyperparameters(config: dict) -> dict:
    for key, val in global_hyperparameters.items():
        if not key in config.keys():
            config[key] = val
    return config

configs = list(map(add_global_hyperparameters, configs))
