"""
Evaluate classifier_best.pth on the imaginary dataset and plot metrics.
Usage:  python test_model.py
"""

import os
import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "data/imaginary"
MODEL_PATH  = "app/src/processing/classifier_best.pth"
NUM_CLASSES = 5
BATCH_SIZE  = 32
SEED        = 42

CLASS_NAMES = ["Diabetes", "PGOV_Glaukom", "SklerózaMultiplex", "SucheOko", "ZdraviLudia"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImaginaryDataset(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_paths = img_paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"), dtype=np.uint8)
        return self.transform(img), self.labels[idx]


def load_dataset(data_dir):
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
        labels.append(raw_label - 1)
    return img_paths, labels

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES),
    )
    return model

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar=False,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def plot_per_class_metrics(y_true, y_pred, ax):
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(NUM_CLASSES)
    width = 0.25
    colors = ["#538AC1", "#00B3DB", "#4CAF50"]
    for i, m in enumerate(metrics):
        vals = [report[c][m] for c in CLASS_NAMES]
        ax.bar(x + i * width, vals, width, label=m.capitalize(), color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Per-class Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)


def plot_roc_curves(y_true, y_probs, ax):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ["#538AC1", "#00B3DB", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")


def plot_precision_recall(y_true, y_probs, ax):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ["#538AC1", "#00B3DB", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=color, lw=1.5, label=f"{name} (AP={ap:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")


def plot_class_distribution(y_true, ax):
    counts = Counter(y_true.tolist())
    names  = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    vals   = [counts.get(i, 0) for i in range(NUM_CLASSES)]
    bars   = ax.bar(names, vals, color="#538AC1")
    ax.set_ylabel("Samples", fontsize=10)
    ax.set_title("Class Distribution", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(v),
                ha="center", va="bottom", fontsize=8)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset …")
    img_paths, labels = load_dataset(DATA_DIR)
    print(f"  {len(img_paths)} images | {Counter(labels)}")

    ds     = ImaginaryDataset(img_paths, labels, VAL_TF)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Loading model …")
    model = build_model().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    print("Running inference …")
    y_true, y_pred, y_probs = evaluate(model, loader)

    acc = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, zero_division=0))

    # ---- Plot ---------------------------------------------------------------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), facecolor="#0a141d")
    fig.suptitle(
        f"Model Evaluation  —  Accuracy: {acc:.2%}  |  N={len(y_true)}",
        fontsize=14, fontweight="bold", color="#ddd", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_cm   = fig.add_subplot(gs[0, 0])
    ax_bar  = fig.add_subplot(gs[0, 1:])
    ax_roc  = fig.add_subplot(gs[1, 0])
    ax_pr   = fig.add_subplot(gs[1, 1])
    ax_dist = fig.add_subplot(gs[1, 2])

    for ax in [ax_cm, ax_bar, ax_roc, ax_pr, ax_dist]:
        ax.set_facecolor("#0d1820")
        for spine in ax.spines.values():
            spine.set_edgecolor("#152331")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#ddd")

    plot_confusion_matrix(y_true, y_pred, ax_cm)
    plot_per_class_metrics(y_true, y_pred, ax_bar)
    plot_roc_curves(y_true, y_probs, ax_roc)
    plot_precision_recall(y_true, y_probs, ax_pr)
    plot_class_distribution(y_true, ax_dist)

    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\nSaved → model_evaluation.png")
    plt.show()


if __name__ == "__main__":
    main()
