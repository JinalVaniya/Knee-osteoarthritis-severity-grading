import os
import copy
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# =====================
# Reproducibility
# =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# =====================
# Config
# =====================
TRAIN_DIR = "/fs1/home/h703296898/final_proj/Dataset/train"
VAL_DIR   = "/fs1/home/h703296898/final_proj/Dataset/val"
TEST_DIR  = "/fs1/home/h703296898/final_proj/Dataset/test"
META_DIR  = "/fs1/home/h703296898/final_proj/metadata/"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 12
MIN_DELTA = 1e-4
SAVE_PATH = "swin_meta_ordinal.pth"

# =====================
# Build image dataframe
# =====================
def build_df(root: str) -> pd.DataFrame:
    data = []
    for folder in sorted(Path(root).iterdir()):
        if folder.is_dir():
            label = int(folder.name)
            for img in folder.glob("*.png"):
                fname = img.name
                pid = fname[:-5]
                side = 1 if fname.endswith("L.png") else 2

                data.append({
                    "path": str(img),
                    "KL_Grade": label,
                    "ID": pid,
                    "SIDE": side,
                })

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["path"])
    return df

train_df = build_df(TRAIN_DIR)
val_df   = build_df(VAL_DIR)
test_df  = build_df(TEST_DIR)

# =====================
# Data leakage check
# =====================
train_ids = set(train_df["ID"])
val_ids   = set(val_df["ID"])
test_ids  = set(test_df["ID"])

print("Overlap patients between train and val:", len(train_ids.intersection(val_ids)))
print("Overlap patients between train and test:", len(train_ids.intersection(test_ids)))
print("Example overlapping IDs (train/val):", list(train_ids.intersection(val_ids))[:10])
print("Example overlapping IDs (train/test):", list(train_ids.intersection(test_ids))[:10])

# =====================
# Metadata loading
# =====================
def get_master_clinical(metadata_path: str) -> pd.DataFrame:
    df_clin = pd.read_sas(os.path.join(metadata_path, "allclinical00.sas7bdat"), encoding="latin1")
    df_enrol = pd.read_sas(os.path.join(metadata_path, "enrollees.sas7bdat"), encoding="latin1")
    df_radio = pd.read_sas(os.path.join(metadata_path, "kxr_sq_bu00.sas7bdat"), encoding="latin1")

    for df in (df_clin, df_enrol, df_radio):
        df["ID"] = df["ID"].astype(str).str.replace(".0", "", regex=False)

    # match metadata SIDE to image SIDE
    df_radio["SIDE"] = df_radio["SIDE"].map({1: 2, 2: 1})

    master = pd.merge(
        df_clin[["ID", "V00AGE", "P01BMI", "V00WOMKPR", "V00WOMKPL", "P01WEIGHT", "P01HEIGHT"]],
        df_enrol[["ID", "P02SEX", "P02RACE", "P02HISP"]],
        on="ID",
    )

    # duplicate patient rows to knee-level rows
    left = master.copy(); left["SIDE"] = 1
    right = master.copy(); right["SIDE"] = 2
    master = pd.concat([left, right], axis=0)

    master = pd.merge(
        master,
        df_radio[[
            "ID", "SIDE",
            "V00XRJSM", "V00XRJSL",
            "V00XROSFM", "V00XROSTL",
            "V00XRSCFM",
        ]],
        on=["ID", "SIDE"],
    )

    master = master.rename(columns={
        "V00AGE": "AGE",
        "P01BMI": "BMI",
        "P01WEIGHT": "WEIGHT",
        "P01HEIGHT": "HEIGHT",
        "V00WOMKPL": "KNEE_PAIN_LEFT",
        "V00WOMKPR": "KNEE_PAIN_RIGHT",
        "P02SEX": "GENDER",
        "P02RACE": "RACE",
        "P02HISP": "HISP",
        "V00XRJSM": "JSN_MEDIAL",
        "V00XRJSL": "JSN_LATERAL",
        "V00XROSFM": "OSTEO_FEMUR_MEDIAL",
        "V00XROSTL": "OSTEO_TIBIA_LATERAL",
        "V00XRSCFM": "SCLEROSIS_FEMUR_MEDIAL",
    })

    master["GENDER"] = master["GENDER"].replace(2, 0)
    master["OVERALL_KNEE_PAIN"] = (master["KNEE_PAIN_LEFT"] + master["KNEE_PAIN_RIGHT"]) / 2
    return master

print("Loading metadata...")
meta_df = get_master_clinical(META_DIR)

# =====================
# Merge image + metadata
# =====================
train_df = train_df.merge(meta_df, on=["ID", "SIDE"])
val_df   = val_df.merge(meta_df, on=["ID", "SIDE"])
test_df  = test_df.merge(meta_df, on=["ID", "SIDE"])

# =====================
# Metadata preprocessing
# =====================
meta_cols = [
    "AGE", "BMI", "KNEE_PAIN_RIGHT", "KNEE_PAIN_LEFT",
    "WEIGHT", "HEIGHT", "OVERALL_KNEE_PAIN",
    "JSN_MEDIAL", "JSN_LATERAL",
    "OSTEO_FEMUR_MEDIAL", "OSTEO_TIBIA_LATERAL",
    "SCLEROSIS_FEMUR_MEDIAL",
]

scaler = StandardScaler()
train_meta = scaler.fit_transform(train_df[meta_cols].fillna(train_df[meta_cols].median()))
val_meta   = scaler.transform(val_df[meta_cols].fillna(train_df[meta_cols].median()))
test_meta  = scaler.transform(test_df[meta_cols].fillna(train_df[meta_cols].median()))

# =====================
# Ordinal label setup
# =====================
# Original multiclass labels: 0,1,2,3,4
# Ordinal targets become 4 binary columns:
# [y > 0, y > 1, y > 2, y > 3]
y_train = train_df["KL_Grade"].values.astype(int)
y_val   = val_df["KL_Grade"].values.astype(int)
y_test  = test_df["KL_Grade"].values.astype(int)

num_classes = len(np.unique(y_train))
num_ord = num_classes - 1

y_train_ord = (y_train[:, None] > np.arange(num_ord)).astype(np.float32)
y_val_ord   = (y_val[:, None] > np.arange(num_ord)).astype(np.float32)
y_test_ord  = (y_test[:, None] > np.arange(num_ord)).astype(np.float32)

# =====================
# Dataset
# =====================
class KneeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, meta: np.ndarray, y_ord: np.ndarray, y_cls: np.ndarray, transform):
        self.df = df.reset_index(drop=True)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.y_ord = torch.tensor(y_ord, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        img = self.transform(img)
        return img, self.meta[idx], self.y_ord[idx], self.y_cls[idx]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = KneeDataset(train_df, train_meta, y_train_ord, y_train, train_tf)
val_ds   = KneeDataset(val_df, val_meta, y_val_ord, y_val, val_tf)
test_ds  = KneeDataset(test_df, test_meta, y_test_ord, y_test, val_tf)

# =====================
# Sampler / class weights
# =====================
# Keep the original balanced sampling idea, but compute it from multiclass labels.
cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
cw = torch.tensor(cw, dtype=torch.float32)
sample_weights = [cw[label].item() for label in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =====================
# Model
# =====================
class MetaEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class SwinMetaOrdinal(nn.Module):
    def __init__(self, meta_dim: int, num_ord: int):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.meta = MetaEncoder(meta_dim)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_ord),
        )

    def forward(self, x, meta):
        img_feat = self.backbone(x)
        meta_feat = self.meta(meta)
        return self.fc(torch.cat([img_feat, meta_feat], dim=1))

# =====================
# Ordinal loss
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinMetaOrdinal(train_meta.shape[1], num_ord).to(device)

# Data-driven positive weights for each ordinal boundary
pos_counts = y_train_ord.sum(axis=0)
neg_counts = len(y_train_ord) - pos_counts
pos_weight = torch.tensor((neg_counts / np.clip(pos_counts, 1, None)), dtype=torch.float32).to(device)

thresholds = torch.tensor([0.45, 0.50, 0.55, 0.60], device=device)[:num_ord]

def ordinal_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    # Monotonicity penalty: later ordinal logits should not exceed earlier ones.
    mono = torch.relu(logits[:, 1:] - logits[:, :-1]).mean() if logits.shape[1] > 1 else 0.0
    return bce + 0.05 * mono

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# =====================
# Training
# =====================
best_f1 = -1
best_state = None
pat = 0

for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for x, meta, y_ord, y_cls in train_loader:
        x = x.to(device)
        meta = meta.to(device)
        y_ord = y_ord.to(device)
        y_cls = y_cls.to(device)

        optimizer.zero_grad()
        out = model(x, meta)
        loss = ordinal_loss(out, y_ord)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        pred = (torch.sigmoid(out) > thresholds).sum(1)
        correct += (pred == y_cls).sum().item()
        total += y_cls.size(0)

    train_loss /= total
    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for x, meta, y_ord, y_cls in val_loader:
            x = x.to(device)
            meta = meta.to(device)
            y_ord = y_ord.to(device)
            y_cls = y_cls.to(device)

            out = model(x, meta)
            loss = ordinal_loss(out, y_ord)
            val_loss += loss.item() * x.size(0)

            preds.extend((torch.sigmoid(out) > thresholds).sum(1).cpu().numpy())
            labels.extend(y_cls.cpu().numpy())

    val_loss /= len(val_ds)
    val_acc = np.mean(np.array(preds) == np.array(labels))
    f1 = f1_score(labels, preds, average="macro")
    bal = balanced_accuracy_score(labels, preds)

    print(
        f"Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
        f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | F1 {f1:.4f} BalAcc {bal:.4f} | {time.time()-start:.1f}s"
    )

    if f1 > best_f1 + MIN_DELTA:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping")
            break

# =====================
# Final validation evaluation
# =====================
model.load_state_dict(best_state)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved final model to {SAVE_PATH}")

model.eval()
val_preds, val_labels, val_probs = [], [], []
with torch.no_grad():
    for x, meta, y_ord, y_cls in val_loader:
        x = x.to(device)
        meta = meta.to(device)
        out = model(x, meta)
        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > thresholds.cpu().numpy()).sum(axis=1)

        val_probs.extend(probs)
        val_preds.extend(preds)
        val_labels.extend(y_cls.numpy())

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)
val_probs = np.array(val_probs)

print("\n================ VALIDATION RESULTS ================\n")
print("VAL F1:", f1_score(val_labels, val_preds, average="macro"))
print("VAL Balanced:", balanced_accuracy_score(val_labels, val_preds))
print(classification_report(val_labels, val_preds))
print(confusion_matrix(val_labels, val_preds))

np.save("val_swin_meta_ordinal_probs.npy", val_probs)
pd.DataFrame({"true": val_labels, "pred": val_preds}).to_csv("val_swin_meta_ordinal.csv", index=False)

# =====================
# Validation ROC Curve (Ordinal)
# =====================
y_true_bin_val = np.array([val_labels > i for i in range(num_ord)]).T
plt.figure(figsize=(8, 6))
for i in range(num_ord):
    fpr, tpr, _ = roc_curve(y_true_bin_val[:, i], val_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"KL > {i} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Validation Ordinal ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("val_roc_curve_swin_meta_ordinal.png")
plt.close()

# =====================
# Validation distribution plot
# =====================
plt.figure(figsize=(8, 5))
sns.histplot(val_labels, label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(val_preds, label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("VALIDATION Actual vs Predicted Distribution")
plt.savefig("val_distribution_swin_meta_ordinal.png")
plt.close()

# =====================
# Test evaluation
# =====================
print("\n================ TEST RESULTS ================\n")

test_preds, test_labels, test_probs = [], [], []
with torch.no_grad():
    for x, meta, y_ord, y_cls in test_loader:
        x = x.to(device)
        meta = meta.to(device)
        out = model(x, meta)
        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > thresholds.cpu().numpy()).sum(axis=1)

        test_probs.extend(probs)
        test_preds.extend(preds)
        test_labels.extend(y_cls.numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_probs = np.array(test_probs)

print("TEST F1:", f1_score(test_labels, test_preds, average="macro"))
print("TEST Balanced:", balanced_accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds))
print(confusion_matrix(test_labels, test_preds))

np.save("test_swin_meta_ordinal_probs.npy", test_probs)
pd.DataFrame({"true": test_labels, "pred": test_preds}).to_csv("test_swin_meta_ordinal.csv", index=False)

# =====================
# Test ROC Curve (Ordinal)
# =====================
y_true_bin_test = np.array([test_labels > i for i in range(num_ord)]).T
plt.figure(figsize=(8, 6))
for i in range(num_ord):
    fpr, tpr, _ = roc_curve(y_true_bin_test[:, i], test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"KL > {i} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Test Ordinal ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("test_roc_curve_swin_meta_ordinal.png")
plt.close()

# =====================
# Test distribution plot
# =====================
plt.figure(figsize=(8, 5))
sns.histplot(test_labels, label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(test_preds, label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("TEST Actual vs Predicted Distribution")
plt.savefig("test_distribution_swin_meta_ordinal.png")
plt.close()

