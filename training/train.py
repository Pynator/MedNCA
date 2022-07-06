from typing import Tuple, Callable
import torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    x: torch.Tensor,
    target_image: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str
) -> Tuple[torch.Tensor, int]:
    """
    TODO
    """
    iter = np.random.randint(64, 96)
    for i in range(iter):
        x = model(x)
    loss = loss_fn(
        x[:, :4],
        torch.concat([
            target_image,
            torch.ones(size=(1, 28, 28)).to(device=device)
        ], dim=0).unsqueeze(dim=0).repeat(x.size()[0], 1, 1, 1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return x, loss.item()
    

def train(
    model: torch.nn.Module,
    epochs: int,
    device: torch.device,
    batch_size: int,
    pool_size: int,
    writer: SummaryWriter,
    mode: str = "growing",
    med_mnist_mod: str = "BloodMNIST",
    med_mnist_index: int = 0,
) -> torch.nn.Module:
    """
    TODO
    """
    optimizer = torch.optim.Adam(params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    loss_fn = torch.nn.MSELoss()

    # Get target image
    target_image = load_image(mod=med_mnist_mod, index=med_mnist_index).to(device)

    # Construct initial pool
    seed = torch.zeros(size=(16, 28, 28), device=device)
    seed[3:, 14, 14] = 1
    pool = seed.unsqueeze(0).repeat((pool_size, 1, 1, 1))

    for epoch in tqdm(range(epochs)):

        if mode == "growing":
            batch = pool[:batch_size]
        elif mode == "persistence":
            batch_indices = torch.randperm(pool_size, device=device)[:batch_size]
            batch = pool[batch_indices]
            # Prevent catastrophic forgetting: Set 1 batch element to the original seed
            batch[0] = seed
        elif mode == "regeneration":
            batch_indices = torch.randperm(pool_size, device=device)[:batch_size]
            batch = pool[batch_indices]
            if epoch > 0:
                coord, coord_indices = torch.randint(7, 21, (int(batch_size / 4), 2, 2), device=device).sort()
                for i in range(int(batch_size / 4)):
                    batch[
                        i,
                        :,
                        coord[i, 0, 0]:coord[i, 0, 1],
                        coord[i, 1, 0]:coord[i, 1, 1]
                    ] = 0
            batch[0] = seed

        x, loss = train_step(model, batch, target_image, optimizer, loss_fn, device)

        if mode == "persistence" or mode == "regeneration":
            pool[batch_indices] = x.detach().clone()

        if epoch % 10 == 0:
            log_and_save(writer, epoch, loss, lr_scheduler, target_image, x, model)

        lr_scheduler.step()

    log_and_save(writer, epoch, loss, lr_scheduler, target_image, x, model, pool)

    writer.close()


def log_and_save(
    writer: SummaryWriter,
    epoch: int,
    loss: int,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    target_image: torch.Tensor,
    x: torch.Tensor,
    model: torch.nn.Module,
    pool: torch.Tensor = None
) -> None:
    """
    Handels training loop logging to tensorboard and saves the pytorch model.
    """
    writer.add_scalar("Loss", loss, global_step=epoch)
    writer.add_scalar("Learning Rate", lr_scheduler.get_last_lr()[0], global_step=epoch)
    writer.add_image("Original", target_image, global_step=epoch)
    writer.add_image("Result", x[0, :3].clip(min=0, max=1), global_step=epoch)

    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, global_step=epoch)
        writer.add_histogram(f"{name}.grad", weight.grad, global_step=epoch)
    
    if pool is not None:
        fig, axes = plt.subplots(4, 4)
        for i, axis in enumerate(axes.flat):
            if i >= pool.shape[0]:
                break
            plt.axis("off")
            axis.imshow(pool[i, :3].permute((1,2,0)).clip(0, 1).cpu().detach().numpy())
        writer.add_figure("Pool", figure=fig, global_step=epoch)

    writer.flush()
    torch.save(model.state_dict(), join(writer.get_logdir(), "model"))
