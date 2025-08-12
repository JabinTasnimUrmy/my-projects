import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from datasets.violence_dataset import ViolenceTensorDataset
from models.violence_model import ViolenceClassifier
from tqdm import tqdm
import numpy as np

# These two functions are correct and do not need to be changed.
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = total_correct = total_samples = 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for clips, labels in pbar:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(clips)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * clips.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += clips.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_correct/total_samples:.4f}")
    return total_loss / total_samples, total_correct / total_samples

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    pbar = tqdm(loader, desc="  val  ", leave=False)
    with torch.no_grad():
        for clips, labels in pbar:
            clips, labels = clips.to(device), labels.to(device)
            with autocast():
                logits = model(clips)
                loss = criterion(logits, labels)
            total_loss += loss.item() * clips.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += clips.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_correct/total_samples:.4f}")
    return total_loss / total_samples, total_correct / total_samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Hyperparameters for Fine-Tuning R(2+1)D
    epochs = 20
    warmup_epochs = 3
    batch_size = 12
    head_lr = 1e-3
    backbone_lr = 1e-6

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])
    val_transform = None

    DATA_DIR = "preprocessed_tensors"
    train_ds = ViolenceTensorDataset(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds = ViolenceTensorDataset(os.path.join(DATA_DIR, "val"), transform=val_transform)

    print(f"[INFO] Total Train clips: {len(train_ds):,},  Total Val clips: {len(val_ds):,}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = ViolenceClassifier(dropout_p=0.5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    # TRAIN THE HEAD 
    print("\n[INFO] PHASE 1: Freezing backbone and training the new classifier head...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr)

    for ep in range(1, warmup_epochs + 1):
        print(f"\nWarmup Epoch {ep:02d}/{warmup_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"[STATS] Epoch {ep:02d} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        # Save the best model from the warmup phase 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/violence_model_best.pth")
            print(f"[INFO] Validation accuracy improved to {val_acc:.4f}. Saved best model.")

    # FINE-TUNE THE ENTIRE MODEL
    print("\n[INFO] PHASE 2: Unfreezing all layers and fine-tuning with a low learning rate...")
    for param in model.parameters():
        param.requires_grad = True

    # Set the learning rates for the head and backbone 
    # Get the IDs of the fc layer parameters to exclude them from the backbone group
    fc_params_ids = set(id(p) for p in model.model.fc.parameters())
    
    # Create the backbone parameter group, excluding the fc layer
    backbone_params = [p for p in model.parameters() if id(p) not in fc_params_ids]
    
    # Create the optimizer with different learning rates for the head and backbone
    optimizer = optim.AdamW([
        {'params': model.model.fc.parameters(), 'lr': head_lr / 10},
        {'params': backbone_params, 'lr': backbone_lr}
    ], weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    for ep in range(1, epochs + 1):
        print(f"\nFinetune Epoch {ep:02d}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr_head = optimizer.param_groups[0]['lr']
        current_lr_backbone = optimizer.param_groups[1]['lr']
        scheduler.step()

        print(f"[STATS] Epoch {ep:02d} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR (Head): {current_lr_head:.1e}, LR (Backbone): {current_lr_backbone:.1e}")
        
        if np.isnan(val_loss): break
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/violence_model_best.pth")
            print(f"[INFO] Validation accuracy improved to {val_acc:.4f}. Saved best model.")
        else:
            print(f"[INFO] Validation accuracy did not improve from best: {best_val_acc:.4f}")
            
    print(f"\n[INFO] Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()