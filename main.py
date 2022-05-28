from model.model import CAModel
from training.train import train
import torch
from torchsummary import summary
if __name__ == "__main__":
    BATCH_SIZE = 100
    model = CAModel(n_channels=16, hidden_channels=64, fire_rate=.5, device="cuda")
    seed = torch.zeros(size=(BATCH_SIZE, 16, 28, 28))
    seed[:, 3:, 14, 14] = 1
    train(model=model, seed=seed, epochs=5000)