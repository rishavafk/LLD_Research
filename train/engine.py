# engine.py - compute_loss + simple decoder for eval
import torch
import math
import numpy as np
from ..config import STRIDES, ANCHORS, NUM_CLASSES, IMG_SIZE, W_BOX, W_OBJ, W_CLS, EPS
from ..loss import focal_bce_loss, mpdiou_between_boxes

def build_targets(preds, boxes_list, labels_list, device):
    """
    Simple target assigner: chooses best anchor across scales by IoU of wh
    Returns per-scale targets similar to earlier script.
    """
    batch = preds[0].shape[0]
    targets_per_scale = []
    for scale_idx, p in enumerate(preds):
        _, _, H, W = p.shape
        A_num = len(ANCHORS[scale_idx])
        obj_mask = torch.zeros((batch, A_num, H, W), dtype=torch.bool, device=device)
        box_target = torch.zeros((batch, A_num, H, W, 4), dtype=torch.float32, device=device)
        cls_target = torch.zeros((batch, A_num, H, W, NUM_CLASSES), dtype=torch.float32, device=device)
        obj_target = torch.zeros((batch, A_num, H, W), dtype=torch.float32, device=device)
        targets_per_scale.append((obj_mask, box_target, cls_target, obj_target))
    # assign
    for b in range(batch):
        gts = boxes_list[b]
        labs = labels_list[b]
        if gts is None or gts.numel() == 0: continue
        gts = gts.to(device)
        cx = (gts[:,0] + gts[:,2]) / 2.0
        cy = (gts[:,1] + gts[:,3]) / 2.0
        gw = (gts[:,2] - gts[:,0]).clamp(min=1.0)
        gh = (gts[:,3] - gts[:,1]).clamp(min=1.0)
        for gi in range(gts.shape[0]):
            best_iou = -1; best_scale=None; best_anchor=None; best_gx=None; best_gy=None
            for scale_idx, stride in enumerate(STRIDES):
                H = preds[scale_idx].shape[2]; W = preds[scale_idx].shape[3]
                anchors = torch.tensor(ANCHORS[scale_idx], device=device).float()
                gt_wh = torch.tensor([gw[gi], gh[gi]], device=device)
                inter_w = torch.min(gt_wh[0], anchors[:,0])
                inter_h = torch.min(gt_wh[1], anchors[:,1])
                inter = inter_w * inter_h
                area_gt = gt_wh[0]*gt_wh[1]
                area_anchor = anchors[:,0]*anchors[:,1]
                union = area_gt + area_anchor - inter + 1e-9
                ious = inter / union
                max_iou, max_idx = torch.max(ious, dim=0)
                if float(max_iou) > best_iou:
                    best_iou = float(max_iou)
                    best_scale = scale_idx
                    best_anchor = int(max_idx)
                    gx = int((cx[gi].item()) / stride)
                    gy = int((cy[gi].item()) / stride)
                    gx = max(0, min(gx, W-1)); gy = max(0, min(gy, H-1))
                    best_gx = gx; best_gy = gy
            if best_scale is None: continue
            obj_mask, box_target, cls_target, obj_target = targets_per_scale[best_scale]
            a_idx = best_anchor; gy = best_gy; gx = best_gx; bidx = b
            obj_mask[bidx, a_idx, gy, gx] = True
            tx = (cx[gi] / STRIDES[best_scale]) - gx
            ty = (cy[gi] / STRIDES[best_scale]) - gy
            tw = math.log((gw[gi].item()+EPS) / (ANCHORS[best_scale][a_idx][0] + EPS))
            th = math.log((gh[gi].item()+EPS) / (ANCHORS[best_scale][a_idx][1] + EPS))
            box_target[bidx, a_idx, gy, gx, 0] = tx
            box_target[bidx, a_idx, gy, gx, 1] = ty
            box_target[bidx, a_idx, gy, gx, 2] = tw
            box_target[bidx, a_idx, gy, gx, 3] = th
            lbl = int(labs[gi].item()) if labs is not None else 0
            cls_target[bidx, a_idx, gy, gx, lbl] = 1.0
            obj_target[bidx, a_idx, gy, gx] = 1.0
    return targets_per_scale

def compute_loss(preds, boxes_list, labels_list, device):
    # preds: list of [B, A*(5+C), H, W]
    batch = preds[0].shape[0]
    targets_per_scale = build_targets(preds, boxes_list, labels_list, device)

    total_box = 0.0; total_obj = 0.0; total_cls = 0.0
    for s,p in enumerate(preds):
        bs,_,H,W = p.shape
        A_num = len(ANCHORS[s])
        p = p.view(bs, A_num, 5+NUM_CLASSES, H, W).permute(0,1,3,4,2).contiguous()
        pred_tx = p[...,0]; pred_ty = p[...,1]; pred_tw = p[...,2]; pred_th = p[...,3]
        pred_obj = p[...,4]; pred_cls = p[...,5:]
        obj_mask, box_target, cls_target, obj_target = targets_per_scale[s]
        obj_loss = focal_bce_loss(pred_obj, obj_target)
        total_obj += obj_loss
        if obj_target.sum() > 0:
            pos_mask = obj_target.bool()
            pred_cls_pos = pred_cls[pos_mask]
            tgt_cls_pos = cls_target[pos_mask]
            if pred_cls_pos.numel() > 0:
                cls_loss = F.binary_cross_entropy_with_logits(pred_cls_pos, tgt_cls_pos, reduction='mean')
                total_cls += cls_loss
        # box loss
        if obj_mask.sum() > 0:
            stride = STRIDES[s]
            anchors = torch.tensor(ANCHORS[s], device=device).float()
            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            xx = xx.unsqueeze(0).unsqueeze(0).repeat(bs, A_num, 1, 1)
            yy = yy.unsqueeze(0).unsqueeze(0).repeat(bs, A_num, 1, 1)
            pred_cx = (torch.sigmoid(pred_tx) + xx) * stride
            pred_cy = (torch.sigmoid(pred_ty) + yy) * stride
            anchor_wh = anchors.view(1, A_num, 1, 1, 2)
            pred_w = torch.exp(pred_tw) * anchor_wh[...,0]
            pred_h = torch.exp(pred_th) * anchor_wh[...,1]
            pred_boxes_list = []
            gt_boxes_list = []
            for b in range(bs):
                pos_idx = obj_mask[b].nonzero(as_tuple=False)
                for item in pos_idx:
                    a_idx, gy, gx = int(item[0]), int(item[1]), int(item[2])
                    pcx = pred_cx[b,a_idx,gy,gx]; pcy = pred_cy[b,a_idx,gy,gx]
                    pw = pred_w[b,a_idx,gy,gx]; ph = pred_h[b,a_idx,gy,gx]
                    px1 = pcx - pw/2; py1 = pcy - ph/2; px2 = pcx + pw/2; py2 = pcy + ph/2
                    pred_boxes_list.append([px1,py1,px2,py2])
                    tx = box_target[b,a_idx,gy,gx,0]; ty = box_target[b,a_idx,gy,gx,1]; tw = box_target[b,a_idx,gy,gx,2]; th = box_target[b,a_idx,gy,gx,3]
                    gcx = (tx + gx) * stride; gcy = (ty + gy) * stride
                    gw = math.exp(tw) * ANCHORS[s][a_idx][0]; gh = math.exp(th) * ANCHORS[s][a_idx][1]
                    gx1 = gcx - gw/2; gy1 = gcy - gh/2; gx2 = gcx + gw/2; gy2 = gcy + gh/2
                    gt_boxes_list.append([gx1,gy1,gx2,gy2])
            if len(pred_boxes_list)>0:
                pred_xy = torch.stack([torch.stack([torch.tensor(x, device=device) for x in b]) for b in [pred_boxes_list]][0]).float()
                gt_xy = torch.stack([torch.stack([torch.tensor(x, device=device) for x in b]) for b in [gt_boxes_list]][0]).float()
                box_loss = mpdiou_between_boxes(pred_xy, gt_xy, IMG_SIZE)
                total_box += box_loss
    loss = W_BOX*total_box + W_OBJ*total_obj + W_CLS*total_cls
    return loss, {'box_loss': float(total_box) if isinstance(total_box, torch.Tensor) else total_box,
                  'obj_loss': float(total_obj) if isinstance(total_obj, torch.Tensor) else total_obj,
                  'cls_loss': float(total_cls) if isinstance(total_cls, torch.Tensor) else total_cls}
