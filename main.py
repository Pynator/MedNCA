from model.model import CAModel
from training.train import train
import torch


if __name__ == "__main__":
    BATCH_SIZE = 32
    POOL_SIZE = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CAModel(n_channels=16, hidden_channels=128, fire_rate=.5, device=device)
    train(model=model, epochs=5000, batch_size=BATCH_SIZE, pool_size=POOL_SIZE, device=device, mode="persistence")
