import torch
from torch.utils.data import DataLoader
import time, json
from tqdm import tqdm

# === CONFIG ===
from config import (
    EXDARK_GT_ROOT,
    EXDARK_IMG_ROOT,
    IMG_SIZE, BATCH, EPOCHS, DEVICE,
    CLASS_NAMES, NUM_CLASSES
)

# === DATA ===
from data.annotations import build_annotation_mapping
from data.dataset import ExDarkDataset

# === MODELS ===
from models.original_yolo import OriginalYOLO
from models.improved_yolo import ImprovedYOLO

# === ENGINE ===
from train.train_engine import train_one_epoch
from train.eval_engine import decode_preds_torch, nms_torch

# === METRICS ===
from train.metrics import compute_map, confusion_matrix_np


# -----------------------------------------------------
# BUILD DATA
# -----------------------------------------------------
def build_loaders():
    ann_map = build_annotation_mapping(EXDARK_GT_ROOT)
    ids = sorted(list(ann_map.keys()))
    n = len(ids)

    train_ids = ids[:int(0.7*n)]
    val_ids   = ids[int(0.7*n):int(0.85*n)]
    test_ids  = ids[int(0.85*n):]

    train_ds = ExDarkDataset(train_ids, ann_map, IMG_SIZE, train=True)
    val_ds   = ExDarkDataset(val_ids,  ann_map, IMG_SIZE, train=False)
    test_ds  = ExDarkDataset(test_ids, ann_map, IMG_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader, test_loader


# -----------------------------------------------------
# SIMPLE TRAIN LOOP
# -----------------------------------------------------
def train_model(model, train_loader, val_loader, tag="yolo"):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_loss = 1e9

    for epoch in range(1, EPOCHS+1):
        print(f"\n[{tag}] Epoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device=DEVICE)

        # ---- validation small check ----
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for imgs, boxes_list, labels_list, _ in val_loader:
                imgs = torch.stack(imgs).to(DEVICE)
                preds = model(imgs)
                val_loss_accum += preds.sum().item()  # rough placeholder

        print(f"[{tag}] TrainLoss = {train_loss:.4f}   ValLoss = {val_loss_accum:.4f}")

        if val_loss_accum < best_loss:
            best_loss = val_loss_accum
            torch.save(model.state_dict(), f"{tag}_best.pth")
            print(f"Saved {tag}_best.pth")

    return model


# -----------------------------------------------------
# FULL EVALUATION
# -----------------------------------------------------
def evaluate_full(model, test_loader, tag="model"):
    print(f"\n====== EVALUATING {tag.upper()} ======")

    all_preds = []
    all_gts   = []
    inference_times = []

    model = model.to(DEVICE)
    model.eval()

    for imgs, boxes_list, labels_list, _ in tqdm(test_loader):
        imgs = torch.stack(imgs).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            raw = model(imgs)
            dets = decode_preds_torch(raw, IMG_SIZE, conf_threshold=0.25)
            dets = nms_torch(dets, iou_threshold=0.45)
        end = time.time()
        inference_times.append(end - start)

        # GT
        for bboxes, labels in zip(boxes_list, labels_list):
            for box, cls in zip(bboxes, labels):
                all_gts.append([cls, box])

        # PRED
        for batch in dets:
            for box in batch:
                x1, y1, x2, y2, sc, cls = box
                all_preds.append([int(cls.item()), float(sc.item()), [x1.item(), y1.item(), x2.item(), y2.item()]])

    # confusion matrix
    cm = confusion_matrix_np(all_preds, all_gts, NUM_CLASSES)

    # mAP
    ap_dict = compute_map(all_preds, all_gts, NUM_CLASSES)
    map50 = ap_dict["mAP50"]
    map5095 = ap_dict["mAP50_95"]

    # Stats
    fps = 1 / (sum(inference_times)/len(inference_times))

    results = {
        "confusion_matrix": cm.tolist(),
        "mAP50": map50,
        "mAP50_95": map5095,
        "AP_per_class": ap_dict["AP_per_class"],
        "FPS": fps
    }

    with open(f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{tag}] mAP@0.5     = {map50:.4f}")
    print(f"[{tag}] mAP@0.5:0.95 = {map5095:.4f}")
    print(f"[{tag}] FPS         = {fps:.2f}")

    return results


# -----------------------------------------------------
# MAIN: TRAIN + EVALUATE BOTH MODELS
# -----------------------------------------------------
def main():
    train_loader, val_loader, test_loader = build_loaders()

    # -------- Train Original --------
    print("\n=== TRAINING ORIGINAL YOLO ===")
    model_orig = OriginalYOLO()
    model_orig = train_model(model_orig, train_loader, val_loader, "original")

    # -------- Train Improved --------
    print("\n=== TRAINING IMPROVED YOLO ===")
    model_impr = ImprovedYOLO()
    model_impr = train_model(model_impr, train_loader, val_loader, "improved")

    # -------- Evaluate Both --------
    r_orig = evaluate_full(model_orig, test_loader, tag="original")
    r_impr = evaluate_full(model_impr, test_loader, tag="improved")

    print("\n=== COMPARISON COMPLETE ===")
    print("Saved results_original.json & results_improved.json")


if __name__ == "__main__":
    main()
