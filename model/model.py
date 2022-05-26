from xml.dom.minidom import Identified
import torch
import torch.nn as nn

import numpy as np

def get_living_mask(x):
        alpha = x[:, 3:4, :, :]
        return nn.functional.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

class CAModel(nn.Module):

    def __init__(self, n_channels, hidden_channels, fire_rate, device=None):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device("cpu")

        self.dmodel = torch.nn.Sequential(
            nn.Conv2d(3*n_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=False)
        )

        with torch.no_grad():
            self.dmodel[2].weight.zero_()

        self.to(device)

    def perceive(self, x, angle=0.0):
        # identity filter
        identify = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            dtype=torch.float32,
        )

        # sobel filter in x dimension
        dx = torch.tensor(
            [
                [-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1]
            ]
        )

        # scale the filter
        scalar = 8.0
        dx = dx / scalar

        # sobel filter in y dimension
        dy = dx.t()

        c, s = np.cos(angle), np.sin(angle)

        kernel = torch.stack(
            [identify, c * dx - s*dy, s*dx + c*dy]
        ) # (3, 3, 3)

        kernel = kernel.repeat((self.n_channels, 1, 1)) # (3 * n_channels, 3, 3)

        kernel = kernel[:, None, ...] # (1, n_channels, 3, 3)

        kernel = kernel.to(device)

        y = nn.functional.conv2d(x, kernel, padding=1, groups=self.n_channels)

        return y

    def forward(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate

        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(self.device, torch.float32)

        x = x + dx * mask

        post_life_mask = get_living_mask(x)

        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

        return x * life_mask

if __name__ == "__main__":
    device = "cuda:0"
    model = CAModel(n_channels=16, hidden_channels=128, fire_rate=0.5, device=device)

    input = torch.ones((1, 16, 3, 3)).to(device)

    print(model(input).shape)