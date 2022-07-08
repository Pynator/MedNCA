import torch
from model.model import CAModel
from training.train import train
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10000
    BATCH_SIZE = 32
    POOL_SIZE = 1024
    N_CHANNELS=16
    MODE = "regeneration"

    model = CAModel(n_channels=N_CHANNELS, hidden_channels=128, fire_rate=0.5, device=device)

    writer = SummaryWriter()

    train(
        model=model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        pool_size=POOL_SIZE,
        device=device,
        mode=MODE,
        n_channels=N_CHANNELS,
        writer=writer
    )
