# metrics.py - confusion matrix & simple precision/recall per-image by taking highest-scoring detection
import numpy as np
from ..config import NUM_CLASSES, CLASS_NAMES

def confusion_and_metrics(preds_list, gts_list, num_classes=NUM_CLASSES):
    # preds_list: list of arrays Nx6 per image [x1,y1,x2,y2,score,cls]
    # gts_list: list of arrays M per image of ground truth labels (integers)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for preds, gts in zip(preds_list, gts_list):
        # determine predicted label for image: choose highest score detection if exists otherwise -1
        if preds.shape[0] == 0:
            pred_label = -1
        else:
            best = preds[np.argmax(preds[:,4])]
            pred_label = int(best[5])
        # for research paper simplistic: if any gt exists, count each gt vs predicted
        if len(gts)==0:
            continue
        for gt in gts:
            gl = int(gt)
            if pred_label == -1:
                # treat as predicted background -> we map to a reserved index? here we increment false-negative to same class
                # increment row gl at column gl? we want per-class recall: if pred_label matches gl increment diag else wrong column
                cm[gl, gl] += 0  # treat no-pred as FN; keep as zero
            else:
                cm[gl, pred_label] += 1
    # compute per-class precision/recall/f1
    metrics = []
    for c in range(num_classes):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp + fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        metrics.append({'precision': prec, 'recall': rec, 'f1': f1, 'tp':int(tp), 'fp':int(fp), 'fn':int(fn)})
    return cm, metrics
