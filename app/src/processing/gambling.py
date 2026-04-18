import random
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

CLASSES = ["Healthy", "Diabetes", "Dry Eye Disease", "Multiple Sclerosis", "Primary Open-Angle Glaucoma"]
NUM_CLASSES = len(CLASSES)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# GLCM(10) + FFT(8) + fractal(1) + morphology(9) = 28
HANDCRAFTED_DIM = 28


def get_random_afm_file(data_dir="../../../data/raw/diabetes"):
    files = [f for f in os.listdir(data_dir) if f.endswith(".bmp")]
    return os.path.join(data_dir, random.choice(files))


# ---------------------------------------------------------------------------
# Handcrafted feature extraction
# ---------------------------------------------------------------------------

def _extract_glcm(img_norm: np.ndarray) -> np.ndarray:
    img_uint8 = (img_norm * 255).astype(np.uint8)
    glcm = graycomatrix(
        img_uint8,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    features = []
    for prop in props:
        v = graycoprops(glcm, prop)
        features += [v.mean(), v.std()]
    return np.array(features, dtype=np.float32)  # 10 features


def _extract_fft(img_norm: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.fftshift(np.fft.fft2(img_norm))) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    radii = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2).ravel()
    power_flat = power.ravel()
    bins = np.linspace(0, min(cy, cx), 9)
    ring_power = np.array(
        [power_flat[(radii >= bins[i]) & (radii < bins[i + 1])].mean() for i in range(8)],
        dtype=np.float32,
    )
    return ring_power / (ring_power.sum() + 1e-12)  # 8 features, normalized


def _extract_fractal(img_norm: np.ndarray) -> np.ndarray:
    binary = img_norm > img_norm.mean()
    sizes, counts = [], []
    for s in [2, 4, 8, 16, 32]:
        nh, nw = binary.shape[0] // s, binary.shape[1] // s
        if nh == 0 or nw == 0:
            break
        block = binary[: nh * s, : nw * s].reshape(nh, s, nw, s).any(axis=(1, 3))
        sizes.append(s)
        counts.append(block.sum())
    if len(counts) < 2:
        return np.array([0.0], dtype=np.float32)
    slope = np.polyfit(np.log(sizes), np.log(np.array(counts) + 1e-12), 1)[0]
    return np.array([slope], dtype=np.float32)  # 1 feature


def _extract_morphology(img_norm: np.ndarray) -> np.ndarray:
    binary = img_norm > img_norm.mean()
    skel = skeletonize(binary)
    density = skel.sum() / (skel.size + 1e-12)
    labeled, n_regions = ndimage.label(skel)
    region_sizes = [np.sum(labeled == i) for i in range(1, n_regions + 1)] if n_regions > 0 else [0]
    return np.array(
        [
            density,
            n_regions,
            np.mean(region_sizes),
            np.std(region_sizes),
            np.max(region_sizes),
            img_norm.mean(),
            img_norm.std(),
            img_norm.min(),
            img_norm.max(),
        ],
        dtype=np.float32,
    )  # 9 features


def extract_handcrafted_features(img: np.ndarray) -> np.ndarray:
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return np.concatenate([
        _extract_glcm(img_norm),       # 10
        _extract_fft(img_norm),        # 8
        _extract_fractal(img_norm),    # 1
        _extract_morphology(img_norm), # 9
    ])  # 28 total


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _GaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


_NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
_TO_3CH = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    _TO_3CH,
    _GaussianNoise(std=0.02),  # simulate AFM measurement noise
    _NORMALIZE,
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # masks random patches, reduces overfit
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _TO_3CH,
    _NORMALIZE,
])


class AFMDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        from PIL import Image as PILImage
        img_arr = np.array(PILImage.open(self.image_paths[idx]).convert("L"), dtype=np.float32)
        img_norm = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-12)
        handcrafted = torch.tensor(extract_handcrafted_features(img_norm), dtype=torch.float32)
        image_tensor = self.transform((img_norm * 255).astype(np.uint8))
        return image_tensor, handcrafted, self.labels[idx]


# ---------------------------------------------------------------------------
# Model: Hybrid CNN + Handcrafted features
#
# Architecture:
#   EfficientNet-B0 (pretrained) → AdaptiveAvgPool → Dropout → Linear(1280→128)
#   Handcrafted(28) → Linear(64) → Linear(32)
#   Concat(128+32=160) → Linear(64) → Dropout → Linear(5)
#
# Why EfficientNet-B0:
#   - Smallest EfficientNet variant (5.3M params), minimal overfitting risk
#   - Compound scaling gives strong texture/structure features from ImageNet
#   - Frozen backbone acts as a fixed feature extractor; only 2 last blocks fine-tuned
# ---------------------------------------------------------------------------

class HybridAFMClassifier(nn.Module):
    def __init__(self, handcrafted_dim: int = HANDCRAFTED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn_features = backbone.features  # (B, 1280, 7, 7) at 224px
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.hc_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.cnn_features.parameters():
            param.requires_grad = False
        # Only last block unfrozen — 120 training samples cannot safely tune more
        for param in self.cnn_features[8].parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.cnn_features.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn_head(self.cnn_pool(self.cnn_features(images)))
        hc_out = self.hc_branch(handcrafted)
        return self.classifier(torch.cat([cnn_out, hc_out], dim=1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, hc, labels in loader:
        images, hc, labels = images.to(DEVICE), hc.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images, hc)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1


@torch.no_grad()
def _eval_epoch(model, loader, criterion, return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, hc, labels in loader:
        images, hc, labels = images.to(DEVICE), hc.to(DEVICE), labels.to(DEVICE)
        logits = model(images, hc)
        total_loss += criterion(logits, labels).item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    if return_preds:
        return total_loss / n, acc, f1, all_preds, all_labels
    return total_loss / n, acc, f1


def train_kfold(
    image_paths: list[str],
    labels: list[int],
    n_splits: int = 5,
    epochs_frozen: int = 5,
    epochs_unfrozen: int = 10,
    batch_size: int = 8,
) -> list[dict]:
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels_arr)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        train_ds = AFMDataset(
            [image_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            transform=TRAIN_TRANSFORM,
        )
        val_ds = AFMDataset(
            [image_paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            transform=EVAL_TRANSFORM,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

        model = HybridAFMClassifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        history = {"tr_loss": [], "tr_acc": [], "tr_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}

        # Phase 1: frozen backbone
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_frozen)
        for epoch in range(epochs_frozen):
            tr_loss, tr_acc, tr_f1 = _train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_f1 = _eval_epoch(model, val_loader, criterion)
            scheduler.step()
            for k, v in zip(history, [tr_loss, tr_acc, tr_f1, val_loss, val_acc, val_f1]):
                history[k].append(v)
            print(f"  [P1] {epoch+1:02d}/{epochs_frozen} | tr {tr_loss:.3f}/{tr_acc:.3f}/{tr_f1:.3f} | val {val_loss:.3f}/{val_acc:.3f}/{val_f1:.3f}")

        # Phase 2: unfreeze, low LR
        model.unfreeze_all()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_unfrozen)
        for epoch in range(epochs_unfrozen):
            tr_loss, tr_acc, tr_f1 = _train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_f1 = _eval_epoch(model, val_loader, criterion)
            scheduler.step()
            for k, v in zip(history, [tr_loss, tr_acc, tr_f1, val_loss, val_acc, val_f1]):
                history[k].append(v)
            print(f"  [P2] {epoch+1:02d}/{epochs_unfrozen} | tr {tr_loss:.3f}/{tr_acc:.3f}/{tr_f1:.3f} | val {val_loss:.3f}/{val_acc:.3f}/{val_f1:.3f}")

        _, final_val_acc, final_val_f1, preds, true_labels = _eval_epoch(
            model, val_loader, criterion, return_preds=True
        )
        print(f"  Final val acc: {final_val_acc:.4f} | f1: {final_val_f1:.4f}")
        fold_results.append({
            "fold": fold + 1,
            "val_acc": final_val_acc,
            "val_f1": final_val_f1,
            "model": model,
            "history": history,
            "preds": preds,
            "true_labels": true_labels,
            "val_loader": val_loader,
        })

    mean_acc = np.mean([r["val_acc"] for r in fold_results])
    mean_f1 = np.mean([r["val_f1"] for r in fold_results])
    print(f"\nMean val accuracy ({n_splits}-fold): {mean_acc:.4f} | mean F1: {mean_f1:.4f}")

    plot_training_results(fold_results)

    return fold_results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_results(fold_results: list[dict]):
    n_folds = len(fold_results)
    short_classes = ["Healthy", "Diab.", "DryEye", "MS", "Glaucoma"]

    # --- per-fold loss / acc / f1 curves ---
    _, axes = plt.subplots(n_folds, 3, figsize=(15, 4 * n_folds))
    if n_folds == 1:
        axes = [axes]

    for row, result in enumerate(fold_results):
        h = result["history"]
        total_epochs = len(h["tr_loss"])
        epochs = range(1, total_epochs + 1)
        phase_split = total_epochs // 2 + 0.5  # approximate P1/P2 boundary

        for col, (metric, title) in enumerate([
            ("loss", "Loss"), ("acc", "Accuracy"), ("f1", "Macro F1")
        ]):
            ax = axes[row][col]
            ax.plot(epochs, h[f"tr_{metric}"], label="train", color="steelblue")
            ax.plot(epochs, h[f"val_{metric}"], label="val", color="tomato")
            ax.axvline(x=phase_split, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"Fold {result['fold']} — {title}")
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle("Training curves (dashed line = phase boundary)", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show()
    print("Saved training_curves.png")

    # --- confusion matrix for best fold ---
    best = max(fold_results, key=lambda r: r["val_f1"])
    cm = confusion_matrix(best["true_labels"], best["preds"])

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # normalised heatmap
    ax = axes[0]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_classes, rotation=30, ha="right")
    ax.set_yticklabels(short_classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — Fold {best['fold']} (best F1={best['val_f1']:.3f})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)

    # per-class TP / FP / FN / TN table
    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for c, cls in enumerate(short_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        rows.append([cls, int(tp), int(fp), int(fn), int(tn)])
    table = ax2.table(
        cellText=rows,
        colLabels=["Class", "TP", "FP", "FN", "TN"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2.0)
    ax2.set_title("Per-class TP / FP / FN / TN", pad=20)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.show()
    print("Saved confusion_matrix.png")


def save_model(model: HybridAFMClassifier, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str) -> HybridAFMClassifier:
    model = HybridAFMClassifier()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_dataset_from_dirs(data_root: str) -> tuple[list[str], list[int]]:
    """
    Scans data_root for per-class subdirectories and collects all .bmp files.

    Expected layout:
        data_root/
            ZdraviLudia/      -> label 0  (Healthy)
            Diabetes/         -> label 1
            SucheOko/         -> label 2  (Dry Eye Disease)
            SklerózaMultiplex/ -> label 3 (Multiple Sclerosis)
            PGOV_Glaukom/     -> label 4  (Primary Open-Angle Glaucoma)
    """
    dir_to_label = {
        "ZdraviLudia": 0,
        "Diabetes": 1,
        "SucheOko": 2,
        "SklerózaMultiplex": 3,
        "PGOV_Glaukom": 4,
    }
    image_paths, labels = [], []
    for dir_name, label in dir_to_label.items():
        folder = os.path.join(data_root, dir_name)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found: {folder}")
            continue
        bmps = [f for f in os.listdir(folder) if f.endswith(".bmp")]
        for f in bmps:
            image_paths.append(os.path.join(folder, f))
            labels.append(label)
        print(f"  {dir_name}: {len(bmps)} images (label {label})")
    return image_paths, labels


@torch.no_grad()
def predict(model: HybridAFMClassifier, img: np.ndarray) -> tuple[str, list[float]]:
    model.eval()
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    handcrafted = torch.tensor(extract_handcrafted_features(img_norm)).unsqueeze(0).to(DEVICE)
    image_tensor = EVAL_TRANSFORM((img_norm * 255).astype(np.uint8)).unsqueeze(0).to(DEVICE)
    logits = model(image_tensor, handcrafted)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    label = CLASSES[int(logits.argmax(1).item())]
    return label, probs


# ---------------------------------------------------------------------------
# Entry point  —  run:  python gambling.py [--predict path/to/image.bmp]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or run the Hybrid AFM classifier")
    parser.add_argument("--data", default="../../../data/raw", help="Path to data/raw directory")
    parser.add_argument("--out", default="hybrid_afm.pt", help="Where to save the best-fold model")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs-frozen", type=int, default=4)
    parser.add_argument("--epochs-unfrozen", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--predict", default=None, help="Skip training, run inference on this image")
    args = parser.parse_args()

    if args.predict:
        print(f"Loading model from {args.out} ...")
        model = load_model(args.out)
        label, probs = predict(model, args.predict)
        print(f"\nPrediction: {label}")
        for cls, p in zip(CLASSES, probs):
            print(f"  {cls:<35} {p*100:.1f}%")
    else:
        print(f"Scanning dataset in: {args.data}")
        image_paths, labels = build_dataset_from_dirs(args.data)
        print(f"Total: {len(image_paths)} images, {len(set(labels))} classes\n")

        results = train_kfold(
            image_paths,
            labels,
            n_splits=args.folds,
            epochs_frozen=args.epochs_frozen,
            epochs_unfrozen=args.epochs_unfrozen,
            batch_size=args.batch_size,
        )

        best = max(results, key=lambda r: r["val_acc"])
        print(f"\nBest fold: {best['fold']} (val_acc={best['val_acc']:.4f})")
        save_model(best["model"], args.out)
        print(f"Model saved to {args.out}")
