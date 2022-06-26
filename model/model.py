from typing import Tuple
import torch
import torch.nn as nn

import numpy as np


def get_living_mask(x):
    alpha = x[:, 3:4, :, :]
    return nn.functional.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1


class CAModel(nn.Module):

    def __init__(self, n_channels, hidden_channels, fire_rate, device=None, filter_type='sobel'):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device("cpu")
        self.filter_type = filter_type

        self.dmodel = torch.nn.Sequential(
            nn.Conv2d(3*n_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=False)
        )

        with torch.no_grad():
            self.dmodel[-1].weight.zero_()

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

        filter1, filter2 = self.get_filters()

        c, s = np.cos(angle), np.sin(angle)

        kernel = torch.stack(
            [identify, c * filter1 - s*filter2, s*filter1 + c*filter2]
        ) # (3, 3, 3)

        kernel = kernel.repeat((self.n_channels, 1, 1)) # (3 * n_channels, 3, 3)

        kernel = kernel[:, None, ...] # (1, n_channels, 3, 3)

        kernel = kernel.to(self.device)

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

    def get_filters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.filter_type == 'sobel':
            # Sobel filters measuring the x and y gradient
            factor = 1 / 8
            dx = torch.tensor(
                [
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]
                ]
            )
            dx = dx * factor

            dy = torch.tensor(
                [
                    [-1,-2,-1],
                    [ 0, 0, 0],
                    [ 1, 2, 1]
                ]
            )
            dy = dy * factor

            return dx, dy

        if self.filter_type == 'gauss_and_laplace':
            # Discrete gaussian (low-pass) and laplacian (high-pass) filters
            factor = 1 / 16
            gauss = torch.tensor(
                [
                    [1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]
                ]
            )
            gauss = gauss * factor

            factor = 1
            laplace = torch.tensor(
                [
                    [0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]
                ]
            )
            laplace = laplace * factor

            return gauss, laplace

        if self.filter_type == 'fixed_random':
            # Numbers from -1 to 1 in random (but fixed) order
            factor = 1 / 4
            rnd1 = torch.tensor(
                [
                    [ 3, 2,-1],
                    [-3, 0, 4],
                    [ 1,-2,-4]
                ]
            )
            rnd1 = rnd1 * factor

            rnd2 = torch.tensor(
                [
                    [ 1,-4, 0],
                    [-2,-1, 3],
                    [ 2, 4,-3]
                ]
            )
            rnd2 = rnd2 * factor

            return rnd1, rnd2

        raise ValueError("Invalid filter_type!")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CAModel(n_channels=16, hidden_channels=128, fire_rate=0.5, device=device)

    input = torch.ones((1, 16, 3, 3)).to(device)

    print(model(input).shape)
