import torch.nn as nn
from .backbone_improved import ImprovedBackbone
from .sdfbf_improved import SDFBF_Improved
from .llem import LLEM
from .head import YOLOHead
from .dropblock import DropBlock2D

class ImprovedYOLO(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = ImprovedBackbone()

        self.llem3 = LLEM(64)
        self.llem4 = LLEM(128)
        self.llem5 = LLEM(256)

        self.sdf3 = SDFBF_Improved(64)
        self.sdf4 = SDFBF_Improved(128)
        self.sdf5 = SDFBF_Improved(256)

        self.drop = DropBlock2D(3, 0.1)

        self.head = YOLOHead([64, 128, 256], num_classes)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)

        p3 = self.llem3(p3)
        p4 = self.llem4(p4)
        p5 = self.llem5(p5)

        p3 = self.sdf3(p3, p3)
        p4 = self.sdf4(p4, p4)
        p5 = self.sdf5(p5, p5)

        p3 = self.drop(p3)
        p4 = self.drop(p4)
        p5 = self.drop(p5)

        return self.head(p3, p4, p5)
