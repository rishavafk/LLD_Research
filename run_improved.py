import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Import config ----
from config import (
    EXDARK_IMG_ROOT,
    EXDARK_GT_ROOT,
    IMG_SIZE,
    BATCH,
    EPOCHS,
    DEVICE
)

# ---- Dataset + transforms ----
from data.annotations import build_annotation_mapping
from data.dataset import ExDarkDataset
from data.transforms import get_train_augs, get_test_augs

# ---- Improved model + loss ----
from models.improved_yolo import ImprovedYOLO
from train.loss_improved import ImprovedLoss

# ---- Trainer utilities ----
from train.train_engine import train_one_epoch
from train.eval_engine import evaluate_model


def create_dataloaders():
    """Create train / val loaders from ExDark"""

    print("[DATA] Building annotation mapping...")
    ann_map = build_annotation_mapping(EXDARK_GT_ROOT)

    ids = list(ann_map.keys())
    ids.sort()

    # 70/15/15 split
    n = len(ids)
    train_ids = ids[:int(0.7*n)]
    val_ids   = ids[int(0.7*n):int(0.85*n)]
    test_ids  = ids[int(0.85*n):]

    print(f"[SPLIT] Train={len(train_ids)}  Val={len(val_ids)}  Test={len(test_ids)}")

    train_ds = ExDarkDataset(train_ids, ann_map, img_size=IMG_SIZE, train=True)
    val_ds   = ExDarkDataset(val_ids,   ann_map, img_size=IMG_SIZE, train=False)
    test_ds  = ExDarkDataset(test_ids,  ann_map, img_size=IMG_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader, test_loader



def train_improved():
    print("\n======== TRAINING IMPROVED MODEL ========")

    # Load dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()

    # Create model + loss
    model = ImprovedYOLO().to(DEVICE)
    criterion = ImprovedLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val = 9999

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # ---- Train ----
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device=DEVICE
        )

        # ---- Validation ----
        val_loss = evaluate_model(
            model, val_loader, device=DEVICE, max_images=100, use_nms=True
        )["loss"]

        print(f"[Epoch {epoch}] TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}")

        # ---- Save best model ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "improved_yolo_best.pth")
            print("Saved improved_yolo_best.pth")

    print("\n======== RUN COMPLETE ========")



if __name__ == "__main__":
    train_improved()
