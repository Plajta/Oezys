"""
Train ResNet18 on the imaginary dataset (multi-channel AFM RGB composites).
Saves best model to app/src/processing/classifier_best.pth
"""

import os
import sys

# Prevent MPS from reserving the full GPU memory budget upfront
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = "data/imaginary"
OUT_PATH   = "app/src/processing/classifier_best.pth"
NUM_CLASSES = 5
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
LR_FINE     = 1e-4
FREEZE_EPOCHS = 15   # epochs with frozen backbone
N_FOLDS     = 5
SEED        = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImaginaryDataset(Dataset):
    def __init__(self, img_paths: list[str], labels: list[int], transform):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"), dtype=np.uint8)
        return self.transform(img), self.labels[idx]


def load_dataset(data_dir: str):
    imgs_dir   = os.path.join(data_dir, "imgs")
    labels_dir = os.path.join(data_dir, "labels")
    img_paths, labels = [], []
    for fname in sorted(os.listdir(imgs_dir)):
        if not fname.endswith(".bmp"):
            continue
        stem = os.path.splitext(fname)[0]
        lbl_file = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(lbl_file):
            continue
        raw_label = int(open(lbl_file).read().strip())
        img_paths.append(os.path.join(imgs_dir, fname))
        labels.append(raw_label - 1)   # 1-5 → 0-4
    return img_paths, labels

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def class_weights(labels: list[int]) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    w = torch.tensor([total / counts[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
    return w / w.sum() * NUM_CLASSES


def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(lbls)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
    n = len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    img_paths, labels = load_dataset(DATA_DIR)
    labels_arr = np.array(labels)
    dist = Counter(labels)
    print(f"Dataset: {len(img_paths)} images | {dist}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_val_f1, best_model_state = 0.0, None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(img_paths, labels_arr)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{N_FOLDS}")

        tr_paths  = [img_paths[i] for i in tr_idx]
        tr_labels = [labels[i]    for i in tr_idx]
        val_paths  = [img_paths[i] for i in val_idx]
        val_labels = [labels[i]    for i in val_idx]

        tr_ds  = ImaginaryDataset(tr_paths,  tr_labels,  TRAIN_TF)
        val_ds = ImaginaryDataset(val_paths, val_labels, VAL_TF)

        sampler    = make_sampler(tr_labels)
        tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = build_model().to(DEVICE)
        cw = class_weights(tr_labels).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

        # Phase 1: frozen backbone
        freeze_backbone(model)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)
        for ep in range(FREEZE_EPOCHS):
            tr_loss, tr_acc, tr_f1 = run_epoch(model, tr_loader, optimizer, criterion, train=True)
            scheduler.step()
            print(f"  [frozen] {ep+1:02d}/{FREEZE_EPOCHS} loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f}")

        # Phase 2: full fine-tune
        unfreeze_all(model)
        optimizer = optim.AdamW(model.parameters(), lr=LR_FINE, weight_decay=5e-4)
        fine_epochs = EPOCHS - FREEZE_EPOCHS
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_epochs)
        for ep in range(fine_epochs):
            tr_loss, tr_acc, tr_f1 = run_epoch(model, tr_loader, optimizer, criterion, train=True)
            scheduler.step()
            print(f"  [full]   {ep+1:02d}/{fine_epochs} loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f}")

        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, None, criterion, train=False)
        print(f"  Val → loss={val_loss:.3f} acc={val_acc:.3f} f1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  *** New best F1={best_val_f1:.4f} — saving")

        # Free MPS memory between folds
        del model, optimizer, scheduler, tr_loader, val_loader, tr_ds, val_ds
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    torch.save(best_model_state, OUT_PATH)
    print(f"\nDone. Best val F1={best_val_f1:.4f} → saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
