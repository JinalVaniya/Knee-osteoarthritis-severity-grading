import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =====================
# Reproducibility
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda")

# =====================
# Config
# =====================
TRAIN_DIR = "/fs1/home/h703296898/final_proj/Dataset/train"
VAL_DIR   = "/fs1/home/h703296898/final_proj/Dataset/val"
META_DIR  = "/fs1/home/h703296898/final_proj/metadata/"
TEST_DIR  = "/fs1/home/h703296898/final_proj/Dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 8

cat_cols = ["SIDE","GENDER","RACE","HISP"]

num_cols = [
    "AGE","BMI","KNEE_PAIN_RIGHT","KNEE_PAIN_LEFT",
    "WEIGHT","HEIGHT","OVERALL_KNEE_PAIN",
    "JSN_MEDIAL","JSN_LATERAL",
    "OSTEO_FEMUR_MEDIAL","OSTEO_TIBIA_LATERAL",
    "SCLEROSIS_FEMUR_MEDIAL"
]

SAVE_PATH = f"img_meta1.pth"
# =====================
# Metadata
# =====================
def get_master_clinical(path):

    df_clin = pd.read_sas(os.path.join(path,"allclinical00.sas7bdat"))
    df_enrol = pd.read_sas(os.path.join(path,"enrollees.sas7bdat"))
    df_radio = pd.read_sas(os.path.join(path,"kxr_sq_bu00.sas7bdat"))

    for df in [df_clin, df_enrol, df_radio]:
        df["ID"] = df["ID"].astype(str).str.replace(".0","",regex=False)

    df_radio["SIDE"] = df_radio["SIDE"].map({1:2, 2:1})

    master = pd.merge(
        df_clin[["ID","V00AGE","P01BMI","V00WOMKPR","V00WOMKPL","P01WEIGHT","P01HEIGHT"]],
        df_enrol[["ID","P02SEX","P02RACE","P02HISP"]],
        on="ID"
    )

    left = master.copy(); left["SIDE"] = 1
    right = master.copy(); right["SIDE"] = 2
    master = pd.concat([left,right])

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
# Image dataframe
# =====================
def build_df(root):
    data = []
    for folder in sorted(Path(root).iterdir()):
        for img in folder.glob("*.png"):
            pid = img.name[:-5]
            side = 1 if img.name.endswith("L.png") else 2

            data.append({
                "image_path": str(img),
                "ID": pid,
                "SIDE": side,
                "KL_Grade": int(folder.name)
            })

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["image_path"])
    return df

train_df = build_df(TRAIN_DIR)
val_df = build_df(VAL_DIR)
test_df = build_df(TEST_DIR)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# =====================
# 🔍 CHECK DATA LEAKAGE (patient overlap)
# =====================
train_ids = set(train_df["ID"])
test_ids = set(test_df["ID"])

overlap = train_ids.intersection(test_ids)
print("Overlap patients:", len(overlap))

# Optional: inspect a few IDs
print("Example overlapping IDs:", list(overlap)[:10])

# =====================
# Merge
# =====================
train_df = train_df.merge(meta_df, on=["ID","SIDE"])
val_df = val_df.merge(meta_df, on=["ID","SIDE"])
test_df = test_df.merge(meta_df, on=["ID","SIDE"])

print("Train after merge:", len(train_df))
print("Val after merge:", len(val_df))
print("Test after merge:", len(test_df))


# =====================
# Scaling
# =====================
scaler_meta = MinMaxScaler()

X_train_num = scaler_meta.fit_transform(train_df[num_cols].fillna(train_df[num_cols].median()))
X_val_num   = scaler_meta.transform(val_df[num_cols].fillna(train_df[num_cols].median()))
X_test_num   = scaler_meta.transform(test_df[num_cols].fillna(train_df[num_cols].median()))

X_train_cat = train_df[cat_cols].fillna(0).astype(int).values
X_val_cat   = val_df[cat_cols].fillna(0).astype(int).values
X_test_cat   = test_df[cat_cols].fillna(0).astype(int).values

X_train_meta = np.concatenate([X_train_cat, X_train_num], axis=1)
X_val_meta   = np.concatenate([X_val_cat, X_val_num], axis=1)
X_test_meta   = np.concatenate([X_test_cat, X_test_num], axis=1)

y_train = train_df["KL_Grade"].values
y_val   = val_df["KL_Grade"].values
y_test   = test_df["KL_Grade"].values

num_classes = len(np.unique(y_train))

# =====================
# Dataset
# =====================
class DatasetStrip(Dataset):
    def __init__(self, df, meta, y, transform):
        self.df = df.reset_index(drop=True)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

        self.meta_imgs = []
        for m in meta:
            strip = np.tile(m, (16,1))   # 🔥 reduced height
            strip = (strip - strip.min()) / (strip.max() - strip.min() + 1e-8)
            strip = np.uint8(strip * 255)

            img = Image.fromarray(strip).resize((IMG_SIZE,16)).convert("RGB")
            self.meta_imgs.append(img)

    def __getitem__(self, idx):
        with Image.open(self.df.iloc[idx]["image_path"]) as img:
            img = img.convert("RGB")

        meta_img = self.meta_imgs[idx]

        combined = Image.new("RGB",(IMG_SIZE,IMG_SIZE+16))
        combined.paste(img.resize((IMG_SIZE,IMG_SIZE)),(0,0))
        combined.paste(meta_img,(0,IMG_SIZE))

        return self.transform(combined), self.y[idx]

    def __len__(self):
        return len(self.df)

# =====================
# Transforms
# =====================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE+16)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE+16)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE+16)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
# =====================
# DataLoaders
# =====================
train_loader = DataLoader(
    DatasetStrip(train_df,X_train_meta,y_train,train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)

val_loader = DataLoader(
    DatasetStrip(val_df,X_val_meta,y_val,val_tf),
    batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True
)

test_loader = DataLoader(
    DatasetStrip(test_df,X_test_meta,y_test,test_tf),
    batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True
)

# =====================
# Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# =====================
# Training
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)

best_f1 = 0
pat = 0

print("Entering training loop...", flush=True)

for epoch in range(MAX_EPOCHS):
    start = time.time()

    model.train()
    train_loss, correct, total = 0,0,0

    for x,y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out,y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

    train_loss /= total
    train_acc = correct/total

    model.eval()
    preds,labels=[],[]
    val_loss,correct,total = 0,0,0

    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out,y)

            val_loss += loss.item()*x.size(0)
            pred = out.argmax(1)

            correct += (pred==y).sum().item()
            total += y.size(0)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    val_loss /= total
    val_acc = correct/total

    f1 = f1_score(labels,preds,average="macro")
    bal = balanced_accuracy_score(labels,preds)

    print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | F1 {f1:.4f} BalAcc {bal:.4f} | {time.time()-start:.1f}s")

    if f1 > best_f1:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
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
    for x, y in test_loader:
        x = x.to(device)

        out = model(x)
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

# Save outputs
np.save("test_img_meta1.npy", test_probs)
pd.DataFrame({"true": test_labels, "pred": test_preds}).to_csv("test_img_meta1.csv", index=False)

# =====================
# TEST ROC Curve (Multiclass)
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
plt.savefig("test_roc_curve_img_meta1.png")
plt.close()

# =====================
# TEST Distribution Plot
# =====================

plt.figure(figsize=(8,5))
sns.histplot(test_labels, color="blue", label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(test_preds, color="red", label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("TEST Actual vs Predicted Distribution")
plt.savefig("Distribution Plot img_meta1.png")
plt.close()
