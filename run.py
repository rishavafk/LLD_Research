# run.py - high level script to run training
from data.annotations import build_annotation_mapping
from data.dataset import ExDarkDataset
from models.lldyolo import LLDYOLO
from train.trainer import train_model
from config import EXDARK_GT_ROOT, EXDARK_IMG_ROOT, EXDARK_SPLIT_FILE, BATCH, NUM_WORKERS, PIN_MEMORY, IMG_SIZE, EPOCHS

from torch.utils.data import DataLoader

def load_split(split_file, ann_map):
    train=[]; val=[]; test=[]
    with open(split_file, "r", errors="ignore") as f:
        for line in f:
            l=line.strip()
            if l=="" or l.lower().startswith("name"): continue
            parts = l.split()
            if len(parts)!=5: continue
            name = parts[0]; split = parts[4]
            if name not in ann_map: continue
            try:
                s = int(split)
            except:
                continue
            if s==1: train.append(name)
            elif s==2: val.append(name)
            elif s==3: test.append(name)
    print("Split:", len(train), len(val), len(test))
    return train,val,test

def collate_fn(batch):
    # batch is list of tuples from dataset
    imgs, boxes, labels, names = zip(*batch)
    return imgs, boxes, labels, names

def main():
    ann_map = build_annotation_mapping(EXDARK_GT_ROOT)
    train_ids,val_ids,test_ids = load_split(EXDARK_SPLIT_FILE, ann_map)
    train_ds = ExDarkDataset(train_ids, ann_map, img_root=EXDARK_IMG_ROOT, img_size=IMG_SIZE, train=True)
    val_ds = ExDarkDataset(val_ids, ann_map, img_root=EXDARK_IMG_ROOT, img_size=IMG_SIZE, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=max(1,NUM_WORKERS//2), pin_memory=PIN_MEMORY, collate_fn=collate_fn)

    model = LLDYOLO()
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, save_tag="lldyolo")
if __name__ == "__main__":
    main()
