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
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# =====================
# Reproducibility
# =====================
def set_seed(seed=42):
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
META_DIR  = "/fs1/home/h703296898/final_proj/metadata/"
TEST_DIR  = "/fs1/home/h703296898/final_proj/Dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 12

SAVE_PATH = f"swin_meta_multi.pth"
# =====================
# Build image dataframe (SIDE FROM IMAGE)
# =====================
def build_df(root):
    data = []
    for folder in sorted(Path(root).iterdir()):
        if folder.is_dir():
            label = int(folder.name)
            for img in folder.glob("*.png"):
                fname = img.name
                pid = fname[:-5]

                # YOUR ORIGINAL IMAGE SIDE (unchanged)
                side = 1 if fname.endswith("L.png") else 2

                data.append({
                    "path": str(img),
                    "KL_Grade": label,
                    "ID": pid,
                    "SIDE": side
                })

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["path"])
    return df

train_df = build_df(TRAIN_DIR)
val_df   = build_df(VAL_DIR)
test_df = build_df(TEST_DIR)

# =====================
# 🔍 CHECK DATA LEAKAGE (patient overlap)
# =====================
train_ids = set(train_df["ID"])
test_ids   = set(test_df["ID"])

overlap = train_ids.intersection(test_ids)

print("Overlap patients:", len(overlap))

# Optional: inspect a few IDs
print("Example overlapping IDs:", list(overlap)[:10])

# =====================
# Load metadata
# =====================
def get_master_clinical(metadata_path):

    df_clin = pd.read_sas(os.path.join(metadata_path,"allclinical00.sas7bdat"), encoding="latin1")
    df_enrol = pd.read_sas(os.path.join(metadata_path,"enrollees.sas7bdat"), encoding="latin1")
    df_radio = pd.read_sas(os.path.join(metadata_path,"kxr_sq_bu00.sas7bdat"), encoding="latin1")

    # ID fix
    for df in [df_clin, df_enrol, df_radio]:
        df["ID"] = df["ID"].astype(str).str.replace(".0","",regex=False)

    # 🔥 FIX: flip metadata SIDE to match image
    df_radio["SIDE"] = df_radio["SIDE"].map({1: 2, 2: 1})

    # Merge clinical + enrollee
    master = pd.merge(
        df_clin[["ID","V00AGE","P01BMI","V00WOMKPR","V00WOMKPL","P01WEIGHT","P01HEIGHT"]],
        df_enrol[["ID","P02SEX","P02RACE","P02HISP"]],
        on="ID"
    )

# =====================
# 🔥 ADD SIDE DUPLICATION
# =====================
    master_left = master.copy()
    master_left["SIDE"] = 1
    master_right = master.copy()
    master_right["SIDE"] = 2
    master = pd.concat([master_left, master_right], axis=0)

    # Merge radiographic (WITH SIDE)
    master = pd.merge(
        master,
        df_radio[[
            "ID","SIDE",
            "V00XRJSM","V00XRJSL",
            "V00XROSFM","V00XROSTL",
            "V00XRSCFM"
        ]],
        on=["ID","SIDE"]
    )

    # Rename
    master = master.rename(columns={
        "V00AGE":"AGE","P01BMI":"BMI",
        "P01WEIGHT":"WEIGHT","P01HEIGHT":"HEIGHT",
        "V00WOMKPL":"KNEE_PAIN_LEFT","V00WOMKPR":"KNEE_PAIN_RIGHT",
        "P02SEX":"GENDER","P02RACE":"RACE","P02HISP":"HISP",
        "V00XRJSM":"JSN_MEDIAL","V00XRJSL":"JSN_LATERAL",
        "V00XROSFM":"OSTEO_FEMUR_MEDIAL","V00XROSTL":"OSTEO_TIBIA_LATERAL",
        "V00XRSCFM":"SCLEROSIS_FEMUR_MEDIAL"
    })

    master["GENDER"] = master["GENDER"].replace(2,0)

    master["OVERALL_KNEE_PAIN"] = (
        master["KNEE_PAIN_LEFT"] + master["KNEE_PAIN_RIGHT"]
    ) / 2

    return master

print("Loading metadata...")
meta_df = get_master_clinical(META_DIR)

# =====================
# Merge image + metadata
# =====================
train_df = train_df.merge(meta_df, on=["ID","SIDE"])
val_df   = val_df.merge(meta_df, on=["ID","SIDE"])
test_df   = test_df.merge(meta_df, on=["ID","SIDE"])

# =====================
# Metadata preprocessing
# =====================
meta_cols = [
    "AGE","BMI","KNEE_PAIN_RIGHT","KNEE_PAIN_LEFT",
    "WEIGHT","HEIGHT","OVERALL_KNEE_PAIN",
    "JSN_MEDIAL","JSN_LATERAL",
    "OSTEO_FEMUR_MEDIAL","OSTEO_TIBIA_LATERAL",
    "SCLEROSIS_FEMUR_MEDIAL"
]

scaler = StandardScaler()
train_meta = scaler.fit_transform(train_df[meta_cols].fillna(train_df[meta_cols].median()))
val_meta   = scaler.transform(val_df[meta_cols].fillna(train_df[meta_cols].median()))
test_meta   = scaler.transform(test_df[meta_cols].fillna(train_df[meta_cols].median()))

# =====================
# Dataset
# =====================
class KneeDataset(Dataset):
    def __init__(self, df, meta, transform):
        self.df = df.reset_index(drop=True)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row.KL_Grade, dtype=torch.long)
        return img, self.meta[idx], label

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = KneeDataset(train_df, train_meta, train_tf)
val_ds   = KneeDataset(val_df, val_meta, val_tf)
test_ds   = KneeDataset(test_df, test_meta, test_tf)
# =====================
# Class weights
# =====================
y = train_df.KL_Grade.values
num_classes = len(np.unique(y))

cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
cw = torch.tensor(cw, dtype=torch.float32)

sample_weights = [cw[l].item() for l in y]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader   = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# =====================
# Model
# =====================
class MetaEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
    def forward(self,x):
        return self.net(x)

class SwinMeta(nn.Module):
    def __init__(self, num_classes, meta_dim):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.meta = MetaEncoder(meta_dim)

        self.fc = nn.Sequential(
            nn.Linear(feat_dim+128,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256,num_classes)
        )

    def forward(self,x,meta):
        img_feat = self.backbone(x)
        meta_feat = self.meta(meta)
        return self.fc(torch.cat([img_feat,meta_feat],dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinMeta(num_classes, train_meta.shape[1]).to(device)
cw = cw.to(device)

# =====================
# Loss & optimizer
# =====================
criterion = nn.CrossEntropyLoss(weight=cw)

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
    train_loss, correct, total = 0,0,0

    for x,meta,yb in train_loader:
        x,meta,yb = x.to(device),meta.to(device),yb.to(device)

        optimizer.zero_grad()
        out = model(x,meta)
        loss = criterion(out,yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*x.size(0)
        correct += (out.argmax(1)==yb).sum().item()
        total += yb.size(0)

    train_loss/=total
    train_acc=correct/total

    model.eval()
    val_loss,correct,total = 0,0,0
    preds,labels=[],[]

    with torch.no_grad():
        for x,meta,yb in val_loader:
            x,meta,yb = x.to(device),meta.to(device),yb.to(device)

            out = model(x,meta)
            loss = criterion(out,yb)

            val_loss += loss.item()*x.size(0)
            correct += (out.argmax(1)==yb).sum().item()
            total += yb.size(0)

            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())

    val_loss/=total
    val_acc=correct/total

    f1 = f1_score(labels,preds,average="macro")
    bal = balanced_accuracy_score(labels,preds)

    print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | F1 {f1:.4f} BalAcc {bal:.4f} | {time.time()-start:.2f}s")

    if f1 > best_f1:
        best_f1=f1
        best_state=copy.deepcopy(model.state_dict())
        pat=0
    else:
        pat+=1
        if pat>=PATIENCE:
            print("Early stopping")
            break

# =====================
# Final
# =====================
model.load_state_dict(best_state)
# SAVE FINAL MODEL
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved final model to {SAVE_PATH}")

model.eval()
test_preds, test_labels, test_probs = [], [], []

with torch.no_grad():
    for x, meta, y in test_loader:
        x, meta = x.to(device), meta.to(device)

        out = model(x, meta)
        probs = torch.softmax(out, dim=1).cpu().numpy()

        test_probs.extend(probs)
        test_preds.extend(np.argmax(probs, axis=1))
        test_labels.extend(y.numpy())


test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_probs = np.array(test_probs)

print("\n================ TEST RESULTS ================\n")
print("TEST F1:", f1_score(test_labels, test_preds, average="macro"))
print("TEST Balanced:", balanced_accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds))
print(confusion_matrix(test_labels, test_preds))

np.save("test_swin_meta_multi.npy", np.array(test_probs))
pd.DataFrame({"true": test_labels, "pred": test_preds}).to_csv("test_swin_meta_multi.csv", index=False)

# =====================
# TEST ROC Curve (ORDINAL VERSION)
# =====================
y_true_bin = label_binarize(test_labels, classes=np.arange(num_classes))

plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"KL{i} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("test_roc_curve_swin_meta_multi.png")
plt.close()

# =====================
# TEST Distribution Plot
# =====================
plt.figure(figsize=(8,5))
sns.histplot(test_labels, color="blue", label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(test_preds, color="red", label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("TEST Actual vs Predicted Distribution")
plt.savefig("test_distribution_swin_meta_multi.png")
plt.close()

