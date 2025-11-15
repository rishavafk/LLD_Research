import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Improved CIoU Loss
# -------------------------------
def bbox_ciou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + 1e-6

    bx1 = (b1_x1 + b1_x2) / 2
    by1 = (b1_y1 + b1_y2) / 2
    bx2 = (b2_x1 + b2_x2) / 2
    by2 = (b2_y1 + b2_y2) / 2

    d2 = (bx2 - bx1) ** 2 + (by2 - by1) ** 2

    ciou = iou - d2 / c2
    return 1 - ciou.clamp(-1, 1)


# -------------------------------
# Focal Objectness Loss
# -------------------------------
def focal_obj_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    pt = torch.where(target == 1, p, 1 - p)
    focal = (alpha * (1 - pt) ** gamma * bce)
    return focal.mean()


# -------------------------------
# Balanced classification loss
# -------------------------------
def balanced_cls_loss(pred, target):
    w_pos = 2.0
    w_neg = 0.5
    loss = w_pos * target * torch.log(pred + 1e-6) + \
           w_neg * (1 - target) * torch.log(1 - pred + 1e-6)
    return -loss.mean()


# -------------------------------
# Overall loss
# -------------------------------
class ImprovedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        box_loss = bbox_ciou(pred["boxes"], target["boxes"]).mean()
        obj_loss = focal_obj_loss(pred["obj"], target["obj"])
        cls_loss = balanced_cls_loss(torch.sigmoid(pred["cls"]), target["cls"])
        return box_loss + obj_loss + cls_loss
