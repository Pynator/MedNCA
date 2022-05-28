from typing import Tuple, Callable
import torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def load_image(mod: str, index: int, path="./images") -> torch.Tensor:
    path = join(path, mod)
    file_names = [f for f in listdir(path) if isfile(join(path, f))]
    file_name = join(path, sorted(file_names)[index])
    img = Image.open(file_name)

    tensor_transform = transforms.ToTensor()
    tensor = tensor_transform(img)
    return tensor


def train_step(
    model: torch.nn.Module,
    x,
    target_image,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, int]:
    """
    TODO
    """
    iter = np.random.randint(64, 96)
    for i in range(iter):
        x = model(x)
    loss = loss_fn(x[:, :3], target_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return x, loss.item()
    

def train(
    model: torch.nn.Module,
    seed: torch.Tensor,
    epochs: int,
    device: str
) -> torch.nn.Module:
    """
    TODO
    """
    writer = SummaryWriter()

    optimizer = torch.optim.Adam(params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer)

    loss_fn = torch.nn.MSELoss()
    target_image = load_image(mod="BloodMNIST", index=0).to(device)
    seed = seed.to(device)
    for epoch in tqdm(range(epochs)):

        x, loss = train_step(model, seed, target_image, optimizer, loss_fn)

        if epoch % 10 == 0:
            writer.add_scalar("Loss", loss, global_step=epoch)
            writer.add_scalar("Learning Rate", lr_scheduler.get_lr(), global_step=epoch)
            writer.add_image("Original", target_image, global_step=epoch)
            writer.add_image("Result", x[0, :3], global_step=epoch)
            writer.flush()

        lr_scheduler.step()

    writer.close()
