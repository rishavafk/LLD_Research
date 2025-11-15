# backbone.py - small conv backbone (kept simple)
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,k,s,p,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class DBB_C2f(nn.Module):
    """ simplified DBB + C2f-like block for training (multi branch) """
    def __init__(self, ch):
        super().__init__()
        self.branch1 = ConvBNAct(ch, ch, k=1, s=1, p=0)
        self.branch3 = ConvBNAct(ch, ch, k=3, s=1, p=1)
        self.pool = nn.Sequential(nn.AvgPool2d(3,1,1), ConvBNAct(ch,ch,k=1,s=1,p=0))
        self.fuse = ConvBNAct(ch, ch, k=1, s=1, p=0)
    def forward(self,x):
        return self.fuse(self.branch1(x) + self.branch3(x) + self.pool(x))

class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3,32,3,2,1)
        self.l1 = nn.Sequential(ConvBNAct(32,64,3,2,1), DBB_C2f(64))
        self.l2 = nn.Sequential(ConvBNAct(64,128,3,2,1), DBB_C2f(128))
        self.l3 = nn.Sequential(ConvBNAct(128,256,3,2,1), DBB_C2f(256))
    def forward(self,x):
        x = self.stem(x)
        c1 = self.l1(x)
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        return c1,c2,c3
