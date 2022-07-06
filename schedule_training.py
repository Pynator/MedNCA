import torch
from model.model import CAModel
from training.train import train
from model_configs import configs
from torch.utils.tensorboard import SummaryWriter
from os.path import join


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in configs:

        model = CAModel(
            n_channels=config["n_channels"],
            hidden_channels=config["hidden_channels"],
            fire_rate=config["fire_rate"],
            filter_type=config["filter"],
            device=device
        )

        writer = SummaryWriter()
        with open(join(writer.get_logdir(), 'parameters.txt'), 'w') as f:
            for key, val in config.items():
                f.write(f"\n{key}: {val}")

        train(
            model=model,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            pool_size=config["pool_size"],
            mode=config["training_mode"],
            med_mnist_mod=config["dataset"],
            med_mnist_index=config["image_index"],
            writer=writer,
            device=device,
        )
