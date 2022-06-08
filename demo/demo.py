from model.model import CAModel
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Demo:
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.text = self.ax.text(
            0.5,
            0.9,
            "",
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=self.ax.transAxes,
            ha="center",
        )

        self.model = CAModel(n_channels=16, hidden_channels=128, fire_rate=0.5, device="cpu")
        self.model.load_state_dict(torch.load("demo/demo_model"))

        self.world = torch.zeros(1, 16, 28, 28)
        self.world[:, 3:, 14, 14] = 1

    def start(self) -> None:
        anim = FuncAnimation(self.fig, self.update, frames=None, blit=True)
        plt.show()

    def update(self, frame) -> list:
        self.world = self.model(self.world)

        self.text.set_text(frame)
        image = np.transpose(self.world[0, :3].detach().numpy().clip(0, 1), (1, 2, 0))
        a = self.ax.imshow(image)
        return [a, self.text]
