import torch
import torch.nn as nn

class LLEM(nn.Module):
    """
    Lightweight Low-Light Enhancement Module
    """

    def __init__(self, c):
        super().__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.enhance(x)
        return x * (1 + w)
