# lldyolo.py - assemble backbone, sdfbf, head
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SimpleBackbone
from .sdfbf import SDFBF
from .head import YOLOHead
from ..config import NUM_CLASSES, ANCHORS

class LLDYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.sdfbf1 = SDFBF(64,128,128)
        self.sdfbf2 = SDFBF(128,256,256)
        # fixed head channels
        self.head = YOLOHead([256,256,256], num_classes=NUM_CLASSES, anchors_per_scale=ANCHORS)
    def forward(self,x):
        c1,c2,c3 = self.backbone(x)
        f1 = self.sdfbf1(c1,c2)
        f2 = self.sdfbf2(f1,c3)
        p3 = f2
        p2 = F.interpolate(f2, scale_factor=0.5, mode='nearest')
        p1 = F.interpolate(f2, scale_factor=0.25, mode='nearest')
        return self.head([p1,p2,p3])
