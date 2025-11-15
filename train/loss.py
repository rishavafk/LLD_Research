# loss.py - focal BCE + MPDIoU + FocalMPD composite
import torch
import torch.nn.functional as F
from .config import FOCAL_ALPHA, FOCAL_GAMMA, EPS

def focal_bce_loss(pred_logits, target, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    prob = torch.sigmoid(pred_logits)
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)
    mod = (1.0 - p_t) ** gamma
    loss = alpha_factor * mod * ce
    return loss.mean()

def mpdiou_between_boxes(pred_xyxy, gt_xyxy, img_size):
    # pred_xyxy, gt_xyxy: (N,4) in pixel coords
    x1 = torch.max(pred_xyxy[:,0], gt_xyxy[:,0])
    y1 = torch.max(pred_xyxy[:,1], gt_xyxy[:,1])
    x2 = torch.min(pred_xyxy[:,2], gt_xyxy[:,2])
    y2 = torch.min(pred_xyxy[:,3], gt_xyxy[:,3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_p = (pred_xyxy[:,2]-pred_xyxy[:,0]).clamp(0) * (pred_xyxy[:,3]-pred_xyxy[:,1]).clamp(0)
    area_g = (gt_xyxy[:,2]-gt_xyxy[:,0]).clamp(0) * (gt_xyxy[:,3]-gt_xyxy[:,1]).clamp(0)
    union = area_p + area_g - inter + 1e-9
    iou = inter / union
    d1 = (pred_xyxy[:,0]-gt_xyxy[:,0])**2 + (pred_xyxy[:,1]-gt_xyxy[:,1])**2
    d2 = (pred_xyxy[:,2]-gt_xyxy[:,2])**2 + (pred_xyxy[:,3]-gt_xyxy[:,3])**2
    D = img_size**2 + img_size**2
    mpd = iou - 0.5*(d1/D) - 0.5*(d2/D)
    return (1 - mpd).mean()

def focal_mpdiou_loss(pred_xyxy, gt_xyxy, img_size, iou, mu=0.9, d=0.2):
    # iou: scalar or tensor IoU between pred and gt; build focaler IoU approx
    # simple focaler: clamp between d .. mu
    iou_f = torch.where(iou < d, torch.zeros_like(iou),
                        torch.where(iou > mu, torch.ones_like(iou),
                                    (iou - d) / (mu - d)))
    loss_mpdiou = mpdiou_between_boxes(pred_xyxy, gt_xyxy, img_size)
    return loss_mpdiou + (iou - iou_f).mean()
