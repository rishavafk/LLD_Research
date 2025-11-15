# head.py - simplified YOLO head returning raw preds per scale
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_chs, num_classes, anchors_per_scale):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors_per_scale[0])
        self.conv_list = nn.ModuleList()
        for c in in_chs:
            self.conv_list.append(nn.Conv2d(c, (5 + num_classes) * self.num_anchors, kernel_size=1))
    def forward(self, feats):
        return [conv(f) for conv,f in zip(self.conv_list, feats)]
