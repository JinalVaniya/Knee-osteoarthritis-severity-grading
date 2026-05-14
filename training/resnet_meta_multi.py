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
from sklearn.preprocessing import RobustScaler
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Config
# =====================
TRAIN_DIR = "/fs1/home/h703296898/final_proj/Dataset/train"
VAL_DIR   = "/fs1/home/h703296898/final_proj/Dataset/val"
META_DIR  = "/fs1/home/h703296898/final_proj/metadata/"
TEST_DIR  = "/fs1/home/h703296898/final_proj/Dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 10

cat_cols = ["SIDE","GENDER","RACE","HISP"]
num_cols = [
    "AGE","BMI","KNEE_PAIN_RIGHT","KNEE_PAIN_LEFT",
    "WEIGHT","HEIGHT","OVERALL_KNEE_PAIN",
    "JSN_MEDIAL","JSN_LATERAL",
    "OSTEO_FEMUR_MEDIAL","OSTEO_TIBIA_LATERAL",
    "SCLEROSIS_FEMUR_MEDIAL"
]

SAVE_PATH = f"img_meta_resnet.pth"
# =====================
# Build image dataframe
# =====================
def build_df(root):
    data = []
    for folder in sorted(Path(root).iterdir()):
        if folder.is_dir():
            label = int(folder.name)
            for img in folder.glob("*.png"):
                pid = img.name[:-5]
                side = 1 if img.name.endswith("L.png") else 2
                data.append({
                    "image_path": str(img),
                    "ID": pid,
                    "SIDE": side,
                    "KL_Grade": label
                })
    return pd.DataFrame(data)

train_df = build_df(TRAIN_DIR)
val_df   = build_df(VAL_DIR)
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
# Metadata function (CORRECT)
# =====================
def get_master_clinical(path):

    df_clin = pd.read_sas(os.path.join(path,"allclinical00.sas7bdat"))
    df_enrol = pd.read_sas(os.path.join(path,"enrollees.sas7bdat"))
    df_radio = pd.read_sas(os.path.join(path,"kxr_sq_bu00.sas7bdat"))

    for df in [df_clin, df_enrol, df_radio]:
        df["ID"] = df["ID"].astype(str).str.replace(".0","",regex=False)

    # 🔥 FIX SIDE mismatch
    df_radio["SIDE"] = df_radio["SIDE"].map({1:2, 2:1})

    master = pd.merge(
        df_clin[["ID","V00AGE","P01BMI","V00WOMKPR","V00WOMKPL","P01WEIGHT","P01HEIGHT"]],
        df_enrol[["ID","P02SEX","P02RACE","P02HISP"]],
        on="ID"
    )

    # duplicate patient → knee-level
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
        "V00WOMKPL":"KNEE_PAIN_LEFT","V00WOMKPR":"KNEE_PAIN_RIGHT",
        "P01WEIGHT":"WEIGHT","P01HEIGHT":"HEIGHT",
        "P02SEX":"GENDER","P02RACE":"RACE","P02HISP":"HISP",
        "V00XRJSM":"JSN_MEDIAL","V00XRJSL":"JSN_LATERAL",
        "V00XROSFM":"OSTEO_FEMUR_MEDIAL","V00XROSTL":"OSTEO_TIBIA_LATERAL",
        "V00XRSCFM":"SCLEROSIS_FEMUR_MEDIAL"
    })

    master["GENDER"] = master["GENDER"].replace(2,0)
    master["OVERALL_KNEE_PAIN"] = (
        master["KNEE_PAIN_LEFT"] + master["KNEE_PAIN_RIGHT"]
    ) / 2

    print("Metadata rows after merge:", len(master))
    return master

meta_df = get_master_clinical(META_DIR)

# =====================
# Merge
# =====================
train_df = train_df.merge(meta_df, on=["ID","SIDE"])
val_df   = val_df.merge(meta_df, on=["ID","SIDE"])
test_df   = test_df.merge(meta_df, on=["ID","SIDE"])

print("Train after merge:", len(train_df))
print("Val after merge:", len(val_df))
print("Test after merge:", len(test_df))
# =====================
# Preprocessing
# =====================
scaler = RobustScaler()
X_train_num = scaler.fit_transform(train_df[num_cols].fillna(train_df[num_cols].median()))
X_val_num = scaler.transform(val_df[num_cols].fillna(train_df[num_cols].median()))
X_test_num = scaler.transform(test_df[num_cols].fillna(train_df[num_cols].median()))

X_train_cat = train_df[cat_cols].fillna(0).astype(int).values
X_val_cat = val_df[cat_cols].fillna(0).astype(int).values
X_test_cat = test_df[cat_cols].fillna(0).astype(int).values

y_train = train_df["KL_Grade"].values
y_val = val_df["KL_Grade"].values
y_test = test_df["KL_Grade"].values

num_classes = len(np.unique(y_train))
num_ord = num_classes - 1

# =====================
# Dataset
# =====================
class DatasetMM(Dataset):
    def __init__(self, df, X_cat, X_num, y, transform):
        self.df = df.reset_index(drop=True)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.y_ord = torch.tensor((y[:,None] > np.arange(num_ord)), dtype=torch.float32)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        img = self.transform(img)
        return img, self.X_cat[idx], self.X_num[idx], self.y[idx], self.y_ord[idx]

    def __len__(self):
        return len(self.df)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_loader = DataLoader(DatasetMM(train_df,X_train_cat,X_train_num,y_train,train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_loader = DataLoader(DatasetMM(val_df,X_val_cat,X_val_num,y_val,val_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = DataLoader(DatasetMM(test_df,X_test_cat,X_test_num,y_test,test_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# =====================
# Model (FIXED)
# =====================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.img = nn.Sequential(*list(resnet.children())[:-1])
        self.img_fc = nn.Linear(512,256)

        # 🔥 categorical embeddings
        self.emb = nn.ModuleList([nn.Embedding(10,8) for _ in cat_cols])

        self.meta = nn.Sequential(
            nn.Linear(4*8 + len(num_cols),256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_ord)
        )

    def forward(self,x,cat,num):
        img = self.img(x).flatten(1)
        img = self.img_fc(img)

        emb = torch.cat([self.emb[i](cat[:,i]) for i in range(len(cat_cols))],dim=1)
        meta = torch.cat([emb,num],dim=1)
        meta = self.meta(meta)

        return self.fc(torch.cat([img,meta],dim=1))

model = Model().to(device)

# =====================
# Ordinal loss
# =====================
pos_weight = torch.tensor([1.2,1.4,1.3,1.2]).to(device)

def ordinal_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    mono = torch.relu(logits[:,1]-logits[:,0]).mean()
    mono += torch.relu(logits[:,2]-logits[:,1]).mean()
    mono += torch.relu(logits[:,3]-logits[:,2]).mean()
    return bce + 0.05 * mono

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
thresholds = torch.tensor([0.45,0.50,0.55,0.60]).to(device)

# =====================
# Training
# =====================
best_f1 = 0
pat = 0

print("Entering training loop...", flush=True)

for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    train_loss, correct, total = 0,0,0

    for x,cat,num,y,y_ord in train_loader:
        x,cat,num,y_ord = x.to(device),cat.to(device),num.to(device),y_ord.to(device)

        optimizer.zero_grad()
        out = model(x,cat,num)
        loss = ordinal_loss(out,y_ord)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*x.size(0)
        pred = (torch.sigmoid(out)>thresholds).sum(1)
        correct += (pred.cpu()==y).sum().item()
        total += y.size(0)

    train_loss/=total
    train_acc=correct/total

    model.eval()
    preds,labels=[],[]
    val_loss=0

    with torch.no_grad():
        for x,cat,num,y,y_ord in val_loader:
            x,cat,num = x.to(device),cat.to(device),num.to(device)
            out=model(x,cat,num)
            loss=ordinal_loss(out,y_ord.to(device))

            val_loss+=loss.item()*x.size(0)
            preds.extend((torch.sigmoid(out)>thresholds).sum(1).cpu().numpy())
            labels.extend(y.numpy())

    val_loss/=len(labels)
    val_acc=np.mean(np.array(preds)==np.array(labels))
    f1=f1_score(labels,preds,average="macro")
    bal=balanced_accuracy_score(labels,preds)

    print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | F1 {f1:.4f} BalAcc {bal:.4f} | {time.time()-start:.1f}s", flush=True)

    if f1>best_f1:
        best_f1=f1
        best_state=copy.deepcopy(model.state_dict())
        pat=0
    else:
        pat+=1
        if pat>=PATIENCE:
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
    for x, cat, num,  y, _  in test_loader:
        x, cat, num = x.to(device), cat.to(device), num.to(device)

        out = model(x, cat,  num)
        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > thresholds.cpu().numpy()).sum(axis=1)

        test_probs.extend(probs)
        test_preds.extend(preds)
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
np.save("test_img_meta_resnet.npy", test_probs)
pd.DataFrame({"true": test_labels, "pred": test_preds}).to_csv("test_img_meta_resnet.csv", index=False)

# =====================
# TEST ROC Curve (Multiclass)
# =====================

plt.figure(figsize=(8,6))
for i in range(num_ord):
    y_true_bin = (test_labels > i).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"KL > {i} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("test_roc_curve_img_meta_resnet.png")
plt.close()

# =====================
# TEST Distribution Plot
# =====================


plt.figure(figsize=(8,5))
sns.histplot(test_labels, color="blue", label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(test_preds, color="red", label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("TEST Actual vs Predicted Distribution")
plt.savefig("test_distribution_img_meta_resnet.png")
plt.close()
