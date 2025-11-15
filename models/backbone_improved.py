import torch
import torch.nn as nn

# --- SE Attention ---
class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# --- Basic Conv ---
def conv(c1, c2, k, s):
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True)
    )


# --- CSP Block ---
class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        mid = c2 // 2
        self.conv1 = conv(c1, mid, 1, 1)
        self.conv2 = conv(c1, mid, 1, 1)

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                conv(mid, mid, 3, 1),
                conv(mid, mid, 3, 1)
            ) for _ in range(n)
        ])

        self.concat = conv(2 * mid, c2, 1, 1)
        self.se = SEBlock(c2)

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        out = torch.cat([y1, y2], dim=1)
        out = self.concat(out)
        return self.se(out)


# --- CSPDarknet-Tiny Backbone ---
class ImprovedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv(3, 32, 3, 1)
        self.stage1 = CSPBlock(32, 64)
        self.stage2 = CSPBlock(64, 128)
        self.stage3 = CSPBlock(128, 256)

    def forward(self, x):
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return p3, p4, p5
