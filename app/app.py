import os
from pathlib import Path
import re

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from utils.predict import load_model, load_scaler, predict_knee_oa
from utils.risk import get_risk_result
from utils.advice import generate_advice_llm, advice_for_streamlit, build_disclaimer
from pathlib import Path
import gdown

MODEL_DIR = BASE_DIR / "models"

IMG_MODEL_PATH = MODEL_DIR / "swin_ord.pth"
META_MODEL_PATH = MODEL_DIR / "swin_meta_ord.pth"

# Google Drive file IDs
IMG_FILE_ID = "13m_4rJtP17jkflO5rBJuAgURY-NZrVpA"
META_FILE_ID = "1Pt6LN2sIhT_xD9Owcle0QE5fbRuacDpm"

MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Knee OA Assistant", page_icon="🦵", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

IMG_MODEL_PATH = MODEL_DIR / "swin_ord.pth"
META_MODEL_PATH = MODEL_DIR / "swin_meta_ord.pth"
META_SCALER_PATH = ARTIFACT_DIR / "swin_meta_scaler.pkl"
LOOKUP_PATH = ARTIFACT_DIR / "patient_metadata.csv"

if not IMG_MODEL_PATH.exists():
    st.error(f"Missing model file: {IMG_MODEL_PATH}")
    st.stop()

st.markdown(
    """
    <style>
        .stApp {
            background: background: linear-gradient( 180deg,#eef3f8 0%, #e6ecf3 50%,#eef3f8 100%);
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] {
            background: #2F2F2F;
        }
        [data-testid="stMetric"] {
            background: #cbd5e1;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 0.6rem 0.9rem;
            box-shadow: 0 2px 10px rgba(37, 99, 235, 0.06);
        }
        .section-title {
            margin-top: 0.35rem;
            margin-bottom: 0.45rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .card {
            background: #cbd5e1;
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.75rem;
        }
        .card-label {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .card-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
        }
        .summary-note {
            color: #b91c1c;
            font-weight: 800;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
            margin: 0.35rem 0 0.75rem 0;
        }
        .subtle-note {
            color: #334155;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
            margin-top: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🦵 Knee Osteoarthritis Checker")
st.caption("Image-only and image+metadata prediction using best Swin-based models.")

MODEL_CHOICES = {
    "Image only (Swin)": "swin_ord",
    "Image + Metadata (Swin + Metadata)": "swin_meta_ord",
}


@st.cache_resource
def cached_model(model_name: str, weights_path: str):
    return load_model(model_name, weights_path)


@st.cache_resource
def cached_scaler(scaler_path: str):
    return load_scaler(scaler_path)


@st.cache_data
def load_patient_lookup(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "ID" in df.columns:
        df["ID"] = (
            df["ID"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
    if "SIDE" in df.columns:
        df["SIDE"] = pd.to_numeric(df["SIDE"], errors="coerce")
    return df


def infer_patient_id_and_side_from_filename(filename: str):
    stem = Path(filename).stem.strip()
    stem_upper = stem.upper()

    # Case 1: 12345L / 12345R (no separator)
    m = re.match(r"^(.*?)([LR])$", stem_upper)
    if m:
        patient_id = m.group(1).strip()
        side_token = m.group(2)
        return patient_id, 1 if side_token == "L" else 2   # L=1, R=2 (training)

    # Case 2: 12345_L / 12345-R / 12345_1 / 12345-2
    m = re.match(r"^(.*?)[_\- ]([LR12])$", stem_upper)
    if m:
        patient_id = m.group(1).strip()
        side_token = m.group(2)

        # IMPORTANT FIX:
        # _1 = RIGHT → SIDE=2
        # _2 = LEFT  → SIDE=1
        if side_token == "1":
            return patient_id, 2   # RIGHT
        if side_token == "2":
            return patient_id, 1   # LEFT

        if side_token == "L":
            return patient_id, 1
        if side_token == "R":
            return patient_id, 2

    # Case 3: strict fallback for _1 / _2
    m = re.match(r"^(.*?)[_\-](1|2)$", stem)
    if m:
        patient_id = m.group(1).strip()
        side_token = m.group(2)
        return patient_id, 2 if side_token == "1" else 1   # FIXED

    # Final fallback
    if "_" in stem:
        left, right = stem.rsplit("_", 1)
        right = right.upper()

        if right == "1":
            return left.strip(), 2   # RIGHT
        if right == "2":
            return left.strip(), 1   # LEFT
        if right == "L":
            return left.strip(), 1
        if right == "R":
            return left.strip(), 2

    raise ValueError(
        "Could not infer patient ID and knee side from filename. "
        "Supported formats: 12345L, 12345R, 12345_L, 12345_R, 12345_1, 12345_2"
    )

def lookup_exact_metadata(df: pd.DataFrame, patient_id: str, side_int: int):
    if df is None or df.empty:
        return None

    if "ID" not in df.columns or "SIDE" not in df.columns:
        return None

    patient_id = str(patient_id).replace(".0", "").strip()

    df2 = df.copy()
    df2["ID"] = (
        df2["ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    df2["SIDE"] = pd.to_numeric(df2["SIDE"], errors="coerce")

    needed_cols = [
        "AGE",
        "GENDER",
        "BMI",
        "KNEE_PAIN_RIGHT",
        "KNEE_PAIN_LEFT",
        "WEIGHT",
        "HEIGHT",
        "OVERALL_KNEE_PAIN",
        "JSN_MEDIAL",
        "JSN_LATERAL",
        "OSTEO_FEMUR_MEDIAL",
        "OSTEO_TIBIA_LATERAL",
        "SCLEROSIS_FEMUR_MEDIAL",
    ]

    rows = df2[df2["ID"] == patient_id]
    if rows.empty:
        return None

    exact = rows[rows["SIDE"] == int(side_int)]
    if not exact.empty:
        row = exact.iloc[0]
    elif len(rows) == 1:
        row = rows.iloc[0]
    else:
        return None

    out = {}
    for col in needed_cols:
        if col in row.index:
            val = row[col]
            out[col] = float(val) if pd.notna(val) else np.nan
        else:
            out[col] = np.nan

    return out


def risk_badge(risk_band: str) -> str:
    if risk_band == "Low Risk":
        return "🟢 Low Risk"
    if risk_band == "Medium Risk":
        return "🟡 Medium Risk"
    if risk_band == "High Risk":
        return "🔴 High Risk"
    return "⚪ Unknown"


def calc_bmi(weight_kg: float, height_cm: float) -> float:
    try:
        height_m = float(height_cm) / 100.0
        if height_m <= 0:
            return 0.0
        return round(float(weight_kg) / (height_m * height_m), 1)
    except Exception:
        return 0.0


def section_title(text: str, color: str = "#2563eb") -> None:
    st.markdown(
        f"<h3 class='section-title' style='color:{color};'>{text}</h3>",
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, accent: str = "#2563eb") -> None:
    st.markdown(
        f"""
        <div class="card" style="border-left: 6px solid {accent};">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_cards(metadata: dict | None) -> None:
    st.sidebar.subheader("Metadata")
    if not metadata:
        st.sidebar.caption(
            "Metadata will be auto-filled from patient metadata using the uploaded image."
        )
        return

    fields = [
        ("AGE", "Age"),
        ("GENDER", "Gender"),
        ("BMI", "BMI"),
        ("KNEE_PAIN_RIGHT", "Right knee pain"),
        ("KNEE_PAIN_LEFT", "Left knee pain"),
        ("WEIGHT", "Weight (kg)"),
        ("HEIGHT", "Height (cm)"),
        ("OVERALL_KNEE_PAIN", "Overall knee pain"),
        ("JSN_MEDIAL", "JSN medial"),
        ("JSN_LATERAL", "JSN lateral"),
        ("OSTEO_FEMUR_MEDIAL", "Osteo femur medial"),
        ("OSTEO_TIBIA_LATERAL", "Osteo tibia lateral"),
        ("SCLEROSIS_FEMUR_MEDIAL", "Sclerosis femur medial"),
    ]

    st.sidebar.caption("Values loaded from patient metadata.")
    for key, label in fields:
        value = metadata.get(key, "—")
        if key == "GENDER":
            if isinstance(value, (int, float, np.integer, np.floating)):
                value = "Male" if float(value) == 1 else "Female"
            else:
                value = str(value)
        elif isinstance(value, (float, np.floating)):
            value = f"{value:.1f}"
        elif isinstance(value, (int, np.integer)):
            value = str(int(value))
        st.sidebar.markdown(
            f"""
            <div class="card" style="margin-bottom:0.55rem; padding:0.9rem 0.95rem;">
                <div class="card-label" style="font-size:1.05rem;">{label}</div>
                <div class="card-value" style="font-size:1.6rem;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.sidebar.header("Prediction mode")
mode_label = st.sidebar.selectbox("Choose model", list(MODEL_CHOICES.keys()))
model_name = MODEL_CHOICES[mode_label]
use_metadata = model_name == "swin_meta_ord"

# UPDATED: manual metadata entry removed; metadata now comes from the CSV lookup only.
st.sidebar.header("Metadata")
if use_metadata:
    st.sidebar.caption("Metadata will be auto-filled from patient metadata using the uploaded image.")
else:
    st.sidebar.caption("This mode uses image only.")

uploaded = st.file_uploader("Upload a knee X-ray", type=["png", "jpg", "jpeg"])
run = st.button("Predict")

if run:
    if uploaded is None:
        st.error("Please upload an X-ray image first.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded X-ray", width=400)

    metadata_lookup = load_patient_lookup(str(LOOKUP_PATH))
    try:
        patient_id_guess, side = infer_patient_id_and_side_from_filename(uploaded.name)
    except ValueError as e:
        if use_metadata:
            st.error(str(e))
            st.stop()
        patient_id_guess, side = Path(uploaded.name).stem, None

    #st.sidebar.caption(f"Parsed ID: {patient_id_guess}")
    #st.sidebar.caption(f"Parsed SIDE: {side if side is not None else '—'}")

    # UPDATED: infer side from the filename instead of asking the user.
    if use_metadata and side is None:
        st.error(
            "Could not infer the knee side from the uploaded filename. "
            "Use a filename ending in L or R (for example: 12345L.png, 12345R.png, 12345_1.png, or 12345_2.png)."
        )
        st.stop()

    exact_metadata = lookup_exact_metadata(metadata_lookup, patient_id_guess, side) if use_metadata else None

    if use_metadata and metadata_lookup is not None:
        id_only_rows = metadata_lookup[
            metadata_lookup["ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip() == str(patient_id_guess).replace(".0", "").strip()
        ] if "ID" in metadata_lookup.columns else pd.DataFrame()
        id_side_rows = metadata_lookup[
            (metadata_lookup["ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip() == str(patient_id_guess).replace(".0", "").strip()) &
            (pd.to_numeric(metadata_lookup["SIDE"], errors="coerce") == int(side))
        ] if ("ID" in metadata_lookup.columns and "SIDE" in metadata_lookup.columns and side is not None) else pd.DataFrame()
        #st.sidebar.caption(f"Rows with ID only: {len(id_only_rows)}")
        #st.sidebar.caption(f"Rows with ID + SIDE: {len(id_side_rows)}")

    # UPDATED: force automatic lookup for multimodal mode.
    if use_metadata and exact_metadata is None:
        st.error(
            f"No matching metadata row was found for ID={patient_id_guess}, SIDE={side}. "
            "Please confirm the filename matches the CSV patient ID format."
        )
        st.stop()

    metadata = exact_metadata if use_metadata else {}

    context_age = exact_metadata["AGE"] if exact_metadata is not None else None
    context_bmi = exact_metadata["BMI"] if exact_metadata is not None else None
    context_pain = exact_metadata["OVERALL_KNEE_PAIN"] if exact_metadata is not None else None

    if model_name == "swin_ord":
        model = cached_model("swin_ord", str(IMG_MODEL_PATH))
        result = predict_knee_oa(
            model=model,
            model_name="swin_ord",
            image=image,
            metadata={},
            scaler=None,
        )
    else:
        model = cached_model("swin_meta_ord", str(META_MODEL_PATH))
        scaler = cached_scaler(str(META_SCALER_PATH))
        if scaler is None:
            st.warning(
                "Metadata scaler file was not found. The multimodal model will run with raw metadata, "
                "which may reduce accuracy."
            )
        result = predict_knee_oa(
            model=model,
            model_name="swin_meta_ord",
            image=image,
            metadata=metadata,
            scaler=scaler,
        )

    risk = get_risk_result(
        kl_grade=result["kl_grade"],
        age=context_age,
        bmi=context_bmi,
        pain=context_pain,
        jsn_medial=exact_metadata["JSN_MEDIAL"] if exact_metadata is not None else None,
        jsn_lateral=exact_metadata["JSN_LATERAL"] if exact_metadata is not None else None,
        osteo_femur_medial=exact_metadata["OSTEO_FEMUR_MEDIAL"] if exact_metadata is not None else None,
        osteo_tibia_lateral=exact_metadata["OSTEO_TIBIA_LATERAL"] if exact_metadata is not None else None,
        sclerosis_femur_medial=exact_metadata["SCLEROSIS_FEMUR_MEDIAL"] if exact_metadata is not None else None,
    )

    advice = generate_advice_llm(
        kl_grade=result["kl_grade"],
        age=context_age,
        bmi=context_bmi,
        pain=context_pain,
        risk_band=risk["final_band"],
        stage_label=risk["stage_label"],
        extra_metadata=metadata if use_metadata else {},
        provider=os.getenv("LLM_PROVIDER", "groq"),
        return_sections=True,
    )

    sections = advice_for_streamlit(advice)

    render_metadata_cards(exact_metadata if use_metadata else None)

    if use_metadata and exact_metadata is not None:
        st.sidebar.markdown(
            f"""
            <div class="card" style="margin-top:0.25rem; margin-bottom:0.65rem;">
                <div class="card-label" style="font-size:1.05rem;">Loaded from CSV</div>
                <div class="card-value" style="font-size:1.15rem; font-weight:700;">ID={patient_id_guess}, SIDE={side}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        section_title("Prediction", "#1d4ed8")
        stat_card("KL Grade", str(result["kl_grade"]), "#2563eb")
        stat_card("Confidence", f'{result["confidence"] * 100:.1f}%', "#7c3aed")
        stat_card("Risk", risk_badge(risk["final_band"]), "#dc2626")

        st.markdown(f"<div style='font-size:1.2rem; font-weight:700; margin-top:0.25rem;'><b>Stage:</b> {result['stage_label']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.2rem; font-weight:700; margin-top:0.1rem;'><b>Base risk:</b> {risk['base_band']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.2rem; font-weight:700; margin-top:0.1rem;'><b>Final risk:</b> {risk['final_band']}</div>", unsafe_allow_html=True)
        st.caption(risk["summary"])

    st.divider()
    st.subheader("Personalized guidance")

    st.markdown(
        "<div class='summary-note'>This is a general education summary, not a medical diagnosis.</div>",
        unsafe_allow_html=True,
    )

    section_title("Summary", "#b91c1c")
    for item in sections.get("Summary", []):
        st.write(f"- {item}")

    left, right = st.columns(2)

    with left:
        section_title("Exercise", "#0f766e")
        for item in sections.get("Exercise", []):
            st.write(f"- {item}")

        section_title("Diet", "#ca8a04")
        for item in sections.get("Diet", []):
            st.write(f"- {item}")

    with right:
        section_title("Avoid", "#c2410c")
        for item in sections.get("Avoid", []):
            st.write(f"- {item}")

        section_title("When to seek medical help", "#6b7280")
        for item in sections.get("When to seek medical help", []):
            st.write(f"- {item}")

    with st.expander("Model output details"):
        st.json(result)

    with st.expander("Risk inputs"):
        st.json(risk)

    st.warning(build_disclaimer())
