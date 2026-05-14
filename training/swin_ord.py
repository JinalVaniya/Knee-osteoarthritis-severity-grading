import os
import copy
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights

from sklearn.metrics import (
    f1_score, balanced_accuracy_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# =====================
# Reproducibility
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================
# Config
# =====================
TRAIN_DIR = "/fs1/home/h703296898/final_proj/Dataset/train"
VAL_DIR   = "/fs1/home/h703296898/final_proj/Dataset/val"
TEST_DIR  = "/fs1/home/h703296898/final_proj/Dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 12
MIN_DELTA = 1e-4

SAVE_PATH = "swin_ordinal_test.pth"

# =====================
# Dataset
# =====================
def build_df(root):
    data = []
    for folder in sorted(Path(root).iterdir()):
        if folder.is_dir():
            label = int(folder.name)
            for img in folder.glob("*.png"):
                data.append({"path": str(img), "label": label})
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["path"])
    return df

train_df = build_df(TRAIN_DIR)
val_df   = build_df(VAL_DIR)
test_df  = build_df(TEST_DIR)

classes = sorted(train_df.label.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)
num_ord = num_classes - 1  # CHANGED: ordinal output size

# =====================
# Dataset class
# =====================
class KneeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = torch.tensor(
            [class_to_idx[v] for v in self.df["label"].values],
            dtype=torch.long,
        )
        # CHANGED: ordinal targets instead of multiclass labels
        self.labels_ord = torch.tensor(
            (self.labels.numpy()[:, None] > np.arange(num_ord)),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        label_ord = self.labels_ord[idx]
        return img, label, label_ord

# =====================
# SAME TRANSFORMS AS RESNET
# =====================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = KneeDataset(train_df, train_tf)
val_ds   = KneeDataset(val_df, val_tf)
test_ds  = KneeDataset(test_df, val_tf)

# =====================
# Class weights
# =====================
y = train_df.label.map(class_to_idx).values
cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
cw = torch.tensor(cw, dtype=torch.float32)

cw[0] *= 1.20
cw[1] *= 1.20
cw[2] *= 1.25
cw[3] *= 1.05

# =====================
# Sampler
# =====================
sample_weights = [cw[class_to_idx[l]].item() for l in train_df.label]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =====================
# Model (Swin Tiny)
# =====================
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

in_features = model.head.in_features
# CHANGED: ordinal head outputs num_classes-1 logits
model.head = nn.Sequential(
    nn.LayerNorm(in_features),
    nn.Dropout(0.3),
    nn.Linear(in_features, num_ord)
)

# =====================
# SAME FREEZE STRATEGY AS RESNET
# =====================
def freeze(model, flag=True):
    for name, p in model.named_parameters():
        if name.startswith("head"):
            p.requires_grad = True
        else:
            p.requires_grad = not flag

freeze(model, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
cw = cw.to(device)

# =====================
# CHANGED: Ordinal loss instead of CrossEntropyLoss
# =====================
pos_weight = torch.ones(num_ord, device=device)
thresholds = torch.linspace(0.45, 0.60, steps=num_ord).to(device)

def ordinal_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )

    mono = 0.0
    for i in range(1, num_ord):
        mono = mono + torch.relu(logits[:, i] - logits[:, i - 1]).mean()

    return bce + 0.05 * mono

optimizer = optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if not n.startswith("head")], "lr": 3e-5},
    {"params": model.head.parameters(), "lr": 3e-4}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =====================
# Training
# =====================
best_f1 = -1
best_state = None
bad_epochs = 0

for epoch in range(EPOCHS):
    start_time = time.time()

    if epoch == 5:
        freeze(model, False)

    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for x, yb, yb_ord in train_loader:
        x, yb, yb_ord = x.to(device), yb.to(device), yb_ord.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = ordinal_loss(out, yb_ord)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        pred = (torch.sigmoid(out) > thresholds).sum(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    train_loss /= total
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    preds, labels, val_probs = [], [], []

    with torch.no_grad():
        for x, yb, yb_ord in val_loader:
            x, yb, yb_ord = x.to(device), yb.to(device), yb_ord.to(device)
            out = model(x)
            loss = ordinal_loss(out, yb_ord)

            val_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(out).cpu().numpy()

            preds.extend((probs > thresholds.cpu().numpy()).sum(axis=1))
            val_probs.extend(probs)
            labels.extend(yb.cpu().numpy())

            pred = torch.tensor(preds[-len(yb):], device=device)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    val_loss /= total
    val_acc = correct / total

    f1 = f1_score(labels, preds, average="macro")
    bal = balanced_accuracy_score(labels, preds)
    rec = recall_score(labels, preds, average=None)

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
        f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
        f"F1 {f1:.4f} BalAcc {bal:.4f} | "
        f"Recall {np.round(rec,2)} | "
        f"{elapsed:.3f}s"
    )

    if f1 > best_f1 + MIN_DELTA:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping triggered")
            break

    scheduler.step()

# =====================
# Final evaluation
# =====================
model.load_state_dict(best_state)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved final model to {SAVE_PATH}")

# =====================
# TEST SET
# =====================
print("\n================ TEST SET ================\n")

test_preds, test_labels, test_probs = [], [], []

with torch.no_grad():
    for x, yb, yb_ord in test_loader:
        x = x.to(device)
        out = model(x)
        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > thresholds.cpu().numpy()).sum(axis=1)

        test_probs.extend(probs)
        test_preds.extend(preds)
        test_labels.extend(yb.numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_probs = np.array(test_probs)

print("TEST F1:", f1_score(test_labels, test_preds, average="macro"))
print("TEST Balanced:", balanced_accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds))
print(confusion_matrix(test_labels, test_preds))

# Save outputs
np.save("test_swin_ordinal_probs.npy", test_probs)
pd.DataFrame({"true": test_labels, "pred": test_preds}).to_csv("test_swin_ordinal_preds.csv", index=False)

# =====================
# TEST ROC Curve (ORDINAL VERSION)
# =====================
plt.figure(figsize=(8, 6))
for i in range(num_ord):
    y_true_bin = (test_labels > i).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"KL > {i} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Ordinal ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("test_roc_curve_swin_ordinal.png")
plt.close()

# =====================
# TEST Distribution Plot
# =====================
plt.figure(figsize=(8, 5))
sns.histplot(test_labels, color="blue", label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(test_preds, color="red", label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("TEST Actual vs Predicted Distribution")
plt.savefig("test_distribution_swin_ordinal.png")
plt.close()
