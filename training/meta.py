import os
import copy
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler

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
META_DIR  = "/fs1/home/h703296898/final_proj/metadata/"
TEST_DIR  = "/fs1/home/h703296898/final_proj/Dataset/test"

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

SAVE_PATH = f"meta.pth"
# =====================
# Build DF (labels only)
# =====================
def build_df(root):
    data = []
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if os.path.isdir(folder_path):
            label = int(folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".png"):
                    pid = fname[:-5]
                    side = 1 if fname.endswith("L.png") else 2
                    data.append({
                        "ID": pid,
                        "SIDE": side,
                        "KL_Grade": label
                    })
    return pd.DataFrame(data)

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
def get_master_clinical(path):

    df_clin = pd.read_sas(os.path.join(path,"allclinical00.sas7bdat"))
    df_enrol = pd.read_sas(os.path.join(path,"enrollees.sas7bdat"))
    df_radio = pd.read_sas(os.path.join(path,"kxr_sq_bu00.sas7bdat"))

    for df in [df_clin, df_enrol, df_radio]:
        df["ID"] = df["ID"].astype(str).str.replace(".0","",regex=False)

    # flip SIDE
    df_radio["SIDE"] = df_radio["SIDE"].map({1:2, 2:1})

    master = pd.merge(
        df_clin[["ID","V00AGE","P01BMI","V00WOMKPR","V00WOMKPL","P01WEIGHT","P01HEIGHT"]],
        df_enrol[["ID","P02SEX","P02RACE","P02HISP"]],
        on="ID"
    )

    # duplicate to knee level
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

    return master

meta_df = get_master_clinical(META_DIR)

# =====================
# Merge
# =====================
train_df = train_df.merge(meta_df, on=["ID","SIDE"])
val_df   = val_df.merge(meta_df, on=["ID","SIDE"])
test_df = test_df.merge(meta_df, on = ["ID","SIDE"])

# =====================
# Preprocessing
# =====================
scaler = RobustScaler()

X_train_num = scaler.fit_transform(train_df[num_cols].fillna(train_df[num_cols].median()))
X_val_num = scaler.transform(val_df[num_cols].fillna(train_df[num_cols].median()))
X_test_num = scaler.transform(test_df[num_cols].fillna(train_df[num_cols].median()))

X_train_cat = train_df[cat_cols].fillna(0).astype(int).values
X_val_cat   = val_df[cat_cols].fillna(0).astype(int).values
X_test_cat   = test_df[cat_cols].fillna(0).astype(int).values

y_train = train_df["KL_Grade"].values
y_val   = val_df["KL_Grade"].values
y_test   = test_df["KL_Grade"].values

num_classes = len(np.unique(y_train))
num_ord = num_classes - 1

# =====================
# Convert to tensors
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_cat = torch.tensor(X_train_cat, dtype=torch.float32).to(device)
X_train_num = torch.tensor(X_train_num, dtype=torch.float32).to(device)


X_val_cat   = torch.tensor(X_val_cat, dtype=torch.float32).to(device)
X_val_num   = torch.tensor(X_val_num, dtype=torch.float32).to(device)

X_test_cat   = torch.tensor(X_test_cat, dtype=torch.float32).to(device)
X_test_num   = torch.tensor(X_test_num, dtype=torch.float32).to(device)

y_train_t = torch.tensor(y_train).to(device)
y_val_t   = torch.tensor(y_val).to(device)
y_test_t   = torch.tensor(y_test).to(device)

y_train_ord = torch.tensor((y_train[:,None] > np.arange(num_ord)), dtype=torch.float32).to(device)
y_val_ord   = torch.tensor((y_val[:,None] > np.arange(num_ord)), dtype=torch.float32).to(device)
y_test_ord   = torch.tensor((y_test[:,None] > np.arange(num_ord)), dtype=torch.float32).to(device)

# =====================
# Model
# =====================
class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(cat_cols)+len(num_cols),256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128,num_ord)
        )

    def forward(self, cat, num):
        x = torch.cat([cat,num],dim=1)
        return self.net(x)

model = MetaModel().to(device)

# =====================
# Ordinal loss
# =====================
pos_weight = torch.tensor([1.2,1.4,1.3,1.2]).to(device)

def ordinal_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    mono = torch.relu(logits[:,1] - logits[:,0]).mean()
    mono += torch.relu(logits[:,2] - logits[:,1]).mean()
    mono += torch.relu(logits[:,3] - logits[:,2]).mean()

    return bce + 0.05 * mono

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
thresholds = torch.tensor([0.45,0.50,0.55,0.60]).to(device)

# =====================
# Training
# =====================
best_f1 = 0
pat = 0

for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    optimizer.zero_grad()

    out = model(X_train_cat, X_train_num)
    loss = ordinal_loss(out, y_train_ord)

    loss.backward()
    optimizer.step()

    train_pred = (torch.sigmoid(out)>thresholds).sum(1)
    train_acc = (train_pred == y_train_t).float().mean().item()

    model.eval()
    with torch.no_grad():
        out = model(X_val_cat, X_val_num)
        val_loss = ordinal_loss(out, y_val_ord)

        preds = (torch.sigmoid(out)>thresholds).sum(1).cpu().numpy()

    val_acc = np.mean(preds==y_val)
    f1 = f1_score(y_val, preds, average="macro")
    bal = balanced_accuracy_score(y_val, preds)

    print(f"Epoch {epoch+1} | Train Loss {loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | F1 {f1:.4f} BalAcc {bal:.4f} | {time.time()-start:.1f}s")

    if f1 > best_f1:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            break

# =====================
# Final
# =====================
model.load_state_dict(best_state)


# SAVE FINAL MODEL
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved final model to {SAVE_PATH}")

# =====================
# TEST SET
# =====================
model.eval()
print("\n================ TEST SET ================\n")


with torch.no_grad():
    out = model(X_test_cat, X_test_num)
    probs = torch.sigmoid(out).cpu().numpy()
    preds = (probs > thresholds.cpu().numpy()).sum(axis=1)

test_preds = preds
test_labels = y_test
test_probs = probs

print("TEST F1:", f1_score(test_labels, test_preds, average="macro"))
print("TEST Balanced:", balanced_accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds))
print(confusion_matrix(test_labels, test_preds))

np.save("test_meta.npy", np.array(test_probs))
pd.DataFrame({"true": test_labels, "pred": test_preds}).to_csv("test_meta.csv", index=False)

# =====================
# TEST ROC Curve (ORDINAL VERSION)
# =====================
y_score_test = np.array(test_probs)   # shape: (N, num_ord)

plt.figure(figsize=(8,6))

for i in range(num_ord):
    # Binary label: is class > i
    y_true_bin = (np.array(test_labels) > i).astype(int)

    fpr, tpr, _ = roc_curve(y_true_bin, y_score_test[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"KL > {i} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("Ordinal ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("test_roc_curve_meta.png")
plt.close()

# =====================
# TEST Distribution Plot
# =====================
plt.figure(figsize=(8,5))
sns.histplot(test_labels, color="blue", label="Actual", bins=5, stat="probability", alpha=0.5)
sns.histplot(test_preds, color="red", label="Predicted", bins=5, stat="probability", alpha=0.5)
plt.legend()
plt.title("TEST Actual vs Predicted Distribution")
plt.savefig("test_distribution_meta.png")
plt.close()

