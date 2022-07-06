import torch
from model.model import CAModel
from training.train import train
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAModel(n_channels=16, hidden_channels=128, fire_rate=0.5, device=device)

    EPOCHS = 10000
    BATCH_SIZE = 32
    POOL_SIZE = 1024
    MODE = "regeneration"

    writer = SummaryWriter()

    train(
        model=model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        pool_size=POOL_SIZE,
        device=device,
        mode=MODE,
        writer=writer
    )
