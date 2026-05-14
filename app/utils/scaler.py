# save_scaler.py
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_DIR = "/fs1/home/h703296898/final_proj/Dataset/train"
VAL_DIR   = "/fs1/home/h703296898/final_proj/Dataset/val"
META_DIR  = "/fs1/home/h703296898/final_proj/metadata/"

meta_cols = [
    "AGE","BMI","KNEE_PAIN_RIGHT","KNEE_PAIN_LEFT",
    "WEIGHT","HEIGHT","OVERALL_KNEE_PAIN",
    "JSN_MEDIAL","JSN_LATERAL",
    "OSTEO_FEMUR_MEDIAL","OSTEO_TIBIA_LATERAL",
    "SCLEROSIS_FEMUR_MEDIAL"
]

def get_master_clinical(metadata_path):
    df_clin = pd.read_sas(os.path.join(metadata_path, "allclinical00.sas7bdat"), encoding="latin1")
    df_enrol = pd.read_sas(os.path.join(metadata_path, "enrollees.sas7bdat"), encoding="latin1")
    df_radio = pd.read_sas(os.path.join(metadata_path, "kxr_sq_bu00.sas7bdat"), encoding="latin1")

    for df in [df_clin, df_enrol, df_radio]:
        df["ID"] = df["ID"].astype(str).str.replace(".0", "", regex=False)

    df_radio["SIDE"] = df_radio["SIDE"].map({1: 2, 2: 1})

    master = pd.merge(
        df_clin[["ID","V00AGE","P01BMI","V00WOMKPR","V00WOMKPL","P01WEIGHT","P01HEIGHT"]],
        df_enrol[["ID","P02SEX","P02RACE","P02HISP"]],
        on="ID"
    )

    left = master.copy(); left["SIDE"] = 1
    right = master.copy(); right["SIDE"] = 2
    master = pd.concat([left, right], axis=0)

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
    master["GENDER"] = master["GENDER"].replace(2, 0)
    master["OVERALL_KNEE_PAIN"] = (master["KNEE_PAIN_LEFT"] + master["KNEE_PAIN_RIGHT"]) / 2
    return master

meta_df = get_master_clinical(META_DIR)

# build your train set the same way as in training
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
                    data.append({"ID": pid, "SIDE": side, "KL_Grade": label})
    return pd.DataFrame(data)

train_df = build_df(TRAIN_DIR)
train_df = train_df.merge(meta_df, on=["ID","SIDE"])

scaler = StandardScaler()
scaler.fit(train_df[meta_cols].fillna(train_df[meta_cols].median()))

with open("swin_meta_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved swin_meta_scaler.pkl")