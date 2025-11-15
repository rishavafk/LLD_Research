import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    def __init__(self, block_size=3, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, 0:1, :, :]) < gamma).float()
        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        return x * mask * (mask.numel() / mask.sum())
