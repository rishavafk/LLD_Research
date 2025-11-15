# trainer.py - training loop and simple evaluation
import torch, time, json
from tqdm import tqdm
from ..config import DEVICE, EPOCHS
from .engine import compute_loss
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=2e-4, save_tag="model"):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE!='cpu'))
    best_val = 1e9
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0; cnt = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, boxes_list, labels_list, names in pbar:
            # imgs is a tuple of tensors due to collate -> convert to batch
            imgs = torch.stack(imgs).to(DEVICE)
            # boxes_list: tuple of tensors
            boxes_batch = [b.to(DEVICE) for b in boxes_list]
            labels_batch = [l.to(DEVICE) for l in labels_list]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE!='cpu')):
                preds = model(imgs)
                loss, metrics = compute_loss(preds, boxes_batch, labels_batch, DEVICE)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach().item())
            cnt += 1
            pbar.set_postfix({'loss': running/(cnt if cnt>0 else 1)})
        avg_train = running / (cnt if cnt>0 else 1)
        # validation (simple)
        model.eval()
        vloss = 0.0; vcnt = 0
        with torch.no_grad():
            for imgs, boxes_list, labels_list, names in val_loader:
                imgs = torch.stack(imgs).to(DEVICE)
                boxes_batch = [b.to(DEVICE) for b in boxes_list]
                labels_batch = [l.to(DEVICE) for l in labels_list]
                preds = model(imgs)
                loss, _ = compute_loss(preds, boxes_batch, labels_batch, DEVICE)
                vloss += float(loss.item()); vcnt += 1
        val_avg = vloss / (vcnt if vcnt>0 else 1)
        print(f"[Epoch {epoch}] Avg Loss = {avg_train:.4f}  ValLoss={val_avg:.4f}")
        torch.save(model.state_dict(), f"{save_tag}_{epoch}epoch.pth")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), f"{save_tag}_best.pth")
    print("Training finished. Best val:", best_val)
    return model
