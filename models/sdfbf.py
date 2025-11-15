# sdfbf.py - Shallow-Deep Feature Bidirectional Fusion (SDFBF)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNAct

class SDFBF(nn.Module):
    def __init__(self, shallow_ch, deep_ch, out_ch):
        super().__init__()
        self.shallow_proj = ConvBNAct(shallow_ch, shallow_ch, k=1, s=1, p=0)
        self.deep_proj = ConvBNAct(deep_ch, deep_ch, k=1, s=1, p=0)
        self.sw_conv = nn.Conv2d(shallow_ch,1,1)
        self.dw_conv = nn.Conv2d(deep_ch,1,1)
        self.deep_to_shallow = ConvBNAct(deep_ch, shallow_ch, k=1, s=1, p=0)
        self.shallow_to_deep = ConvBNAct(shallow_ch, deep_ch, k=1, s=1, p=0)
        self.fuse_conv = ConvBNAct(shallow_ch*2, out_ch, k=1, s=1, p=0)
    def forward(self, SFM_in, DFM_in):
        SFM1 = self.shallow_proj(SFM_in)
        DFM1 = self.deep_proj(DFM_in)
        SW = torch.sigmoid(self.sw_conv(SFM1))
        DW = torch.sigmoid(self.dw_conv(DFM1))
        # upsample deep to shallow resolution
        DFM1_up = F.interpolate(DFM1, size=SFM1.shape[2:], mode='nearest')
        DW_up = F.interpolate(DW, size=SFM1.shape[2:], mode='nearest')
        deep_to_shallow = self.deep_to_shallow(DFM1_up)
        SFM_pos = SFM1 * SW
        DFM_pos_shallow = deep_to_shallow * DW_up
        SFM_cross = (1.0 - SW) * DFM_pos_shallow
        SFM_total = SFM1 + SFM_pos + SFM_cross
        SFM1_down = F.interpolate(SFM1, size=DFM1.shape[2:], mode='nearest')
        SW_down = F.interpolate(SW, size=DFM1.shape[2:], mode='nearest')
        shallow_to_deep = self.shallow_to_deep(SFM1_down)
        DFM_cross = (1.0 - DW) * shallow_to_deep
        DFM_cross_up = F.interpolate(DFM_cross, size=SFM1.shape[2:], mode='nearest')
        DFM_cross_up_shallow = self.deep_to_shallow(DFM_cross_up)
        DFM_total_up_shallow = deep_to_shallow + DFM_pos_shallow + DFM_cross_up_shallow
        concat = torch.cat([SFM_total, DFM_total_up_shallow], dim=1)
        out = self.fuse_conv(concat)
        return out
