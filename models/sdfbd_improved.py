import torch
import torch.nn as nn

class SDFBF_Improved(nn.Module):
    """
    Improved shallow–deep feature fusion block.
    Adds:
      - SE attention
      - learnable fusion weights
    """

    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 8, c, 1),
            nn.Sigmoid()
        )

    def forward(self, shallow, deep):
        w1 = torch.sigmoid(self.w1)
        w2 = torch.sigmoid(self.w2)

        fused = w1 * shallow + w2 * deep
        fused = fused * self.se(fused)
        return fused
