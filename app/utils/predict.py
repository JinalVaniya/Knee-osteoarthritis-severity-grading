from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import swin_t


IMG_SIZE = 224
NUM_CLASSES = 5
NUM_ORD = NUM_CLASSES - 1
META_DIM = 12
DEFAULT_THRESHOLDS = np.array([0.45, 0.50, 0.55, 0.60], dtype=np.float32)

# Keep the inference schema locked to the fields used during training.
TRAIN_META_KEYS = [
    "AGE",
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


@dataclass
class PredictionResult:
    model_name: str
    kl_grade: int
    confidence: float
    raw_output: Any
    probabilities: Dict[str, float]
    stage_label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "kl_grade": self.kl_grade,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
            "probabilities": self.probabilities,
            "stage_label": self.stage_label,
        }


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(str(image)).convert("RGB")


def kl_to_stage_label(kl_grade: int) -> str:
    mapping = {
        0: "No OA",
        1: "Doubtful / Very Early OA",
        2: "Mild OA",
        3: "Moderate OA",
        4: "Severe OA",
    }
    return mapping.get(int(kl_grade), "Unknown")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        val = float(v)
        if not np.isfinite(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def normalize_side(side: Any) -> int:
    """
    Normalize a side label to the training convention:
      1 = left knee
      2 = right knee

    Accepts: 1/2, "1"/"2", "L"/"R", "Left"/"Right".
    """
    if side is None:
        raise ValueError("SIDE is required for metadata lookup.")

    if isinstance(side, str):
        s = side.strip().upper()
        if s in {"1", "L", "LEFT"}:
            return 1
        if s in {"2", "R", "RIGHT"}:
            return 2
        raise ValueError(f"Unrecognized SIDE value: {side}")

    try:
        side_int = int(side)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Unrecognized SIDE value: {side}") from e

    if side_int in (1, 2):
        return side_int

    raise ValueError(f"Unrecognized SIDE value: {side}")


def infer_side_from_filename(image_path: Union[str, Path]) -> Optional[int]:
    """
    Only use this if you truly do not have a side value in CSV/metadata.

    Training used filename suffix L/R. If the filename is ambiguous,
    return None instead of guessing.
    """
    name = Path(str(image_path)).name.upper()

    if name.endswith("L.PNG") or name.endswith("_L.PNG") or name.endswith("-L.PNG"):
        return 1
    if name.endswith("R.PNG") or name.endswith("_R.PNG") or name.endswith("-R.PNG"):
        return 2

    return None


def select_metadata_row(
    metadata_table: pd.DataFrame,
    patient_id: Any,
    side: Any,
) -> Dict[str, Any]:
    """
    Select the exact row used by the training merge: (ID, SIDE).

    This is the correct place to resolve side, because the Swin+Meta model
    does NOT take SIDE as an input feature; SIDE is only used to choose the
    correct metadata row.
    """
    if not isinstance(metadata_table, pd.DataFrame):
        raise TypeError("metadata_table must be a pandas DataFrame.")

    if "ID" not in metadata_table.columns or "SIDE" not in metadata_table.columns:
        raise ValueError("metadata_table must contain 'ID' and 'SIDE' columns.")

    pid = str(patient_id).strip()
    side_n = normalize_side(side)

    matches = metadata_table[
        (metadata_table["ID"].astype(str).str.strip() == pid)
        & (metadata_table["SIDE"].apply(normalize_side) == side_n)
    ]

    if matches.empty:
        raise LookupError(f"No metadata row found for ID={pid}, SIDE={side_n}.")

    if len(matches) > 1:
        warnings.warn(f"Multiple metadata rows found for ID={pid}, SIDE={side_n}; using the first match.")

    return matches.iloc[0].to_dict()


def load_scaler(scaler_path: Optional[Union[str, Path]]) -> Any:
    if scaler_path is None:
        return None

    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def make_image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def prepare_swin_metadata(metadata: Dict[str, Any], scaler: Any) -> np.ndarray:
    """
    Prepare the exact 12 metadata features used during training.

    IMPORTANT:
    - The fitted scaler is required.
    - Do not use raw metadata here, because the ordinal Swin+Meta model
      was trained on scaled features.
    """
    if scaler is None:
        raise ValueError(
            "A fitted scaler is required for Swin+Meta ordinal inference. "
            "Load the training .pkl scaler and pass it into predict_swin_meta_ord."
        )

    if not isinstance(metadata, dict):
        metadata = dict(metadata)

    extra_keys = sorted(set(metadata.keys()) - set(TRAIN_META_KEYS))
    if extra_keys:
        warnings.warn(
            "Ignoring metadata keys not used by the trained Swin meta ordinal model: "
            + ", ".join(extra_keys)
        )

    arr = np.array(
        [[
            _safe_float(metadata.get("AGE", 0)),
            _safe_float(metadata.get("BMI", 0)),
            _safe_float(metadata.get("KNEE_PAIN_RIGHT", 0)),
            _safe_float(metadata.get("KNEE_PAIN_LEFT", 0)),
            _safe_float(metadata.get("WEIGHT", 0)),
            _safe_float(metadata.get("HEIGHT", 0)),
            _safe_float(metadata.get("OVERALL_KNEE_PAIN", 0)),
            _safe_float(metadata.get("JSN_MEDIAL", 0)),
            _safe_float(metadata.get("JSN_LATERAL", 0)),
            _safe_float(metadata.get("OSTEO_FEMUR_MEDIAL", 0)),
            _safe_float(metadata.get("OSTEO_TIBIA_LATERAL", 0)),
            _safe_float(metadata.get("SCLEROSIS_FEMUR_MEDIAL", 0)),
        ]],
        dtype=np.float32,
    )

    try:
        arr = scaler.transform(arr)
    except Exception as e:
        raise ValueError(f"Scaler transform failed: {e}") from e

    return arr


def _maybe_strip_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    prefixes = ("module.", "model.")
    for prefix in prefixes:
        if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return _maybe_strip_prefix(ckpt[key])
        return _maybe_strip_prefix(ckpt)
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _ordinal_to_class_probs(boundary_probs: np.ndarray) -> np.ndarray:
    """
    Convert ordinal boundary probabilities P(y > k) into class probabilities.
    boundary_probs shape: (num_ord,)
    returns shape: (NUM_CLASSES,)
    """
    bp = np.clip(np.asarray(boundary_probs, dtype=np.float32), 0.0, 1.0)
    if bp.shape[0] != NUM_ORD:
        raise ValueError(f"Expected {NUM_ORD} ordinal probabilities, got {bp.shape[0]}")

    class_probs = np.empty(NUM_CLASSES, dtype=np.float32)
    class_probs[0] = 1.0 - bp[0]
    for i in range(1, NUM_ORD):
        class_probs[i] = bp[i - 1] - bp[i]
    class_probs[NUM_ORD] = bp[NUM_ORD - 1]

    class_probs = np.clip(class_probs, 0.0, None)
    total = float(class_probs.sum())
    if total <= 0:
        class_probs[:] = 1.0 / NUM_CLASSES
    else:
        class_probs /= total
    return class_probs


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


class SwinMetaOrdinalClassifier(nn.Module):
    """Matches the multimodal ordinal Swin checkpoint saved in training."""

    def __init__(self, meta_dim: int = META_DIM, num_ord: int = NUM_ORD):
        super().__init__()
        self.backbone = swin_t(weights=None)
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


def _build_swin_ordinal_classifier() -> nn.Module:
    """Build the plain torchvision Swin-Tiny ordinal model exactly like training."""
    model = swin_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_ORD),
    )
    return model


def load_model(
    model_name: str,
    weights_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Supported:
      - swin_ord
      - swin_meta_ord
    Backward-compatible aliases:
      - swin_img -> swin_ord
      - swin_meta_multi -> swin_meta_ord
    """
    device = get_device(device)
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model_name = model_name.lower().strip()
    model_name = {
        "swin_img": "swin_ord",
        "swin_meta_multi": "swin_meta_ord",
    }.get(model_name, model_name)

    if model_name == "swin_ord":
        model = _build_swin_ordinal_classifier()
    elif model_name == "swin_meta_ord":
        model = SwinMetaOrdinalClassifier(meta_dim=META_DIM, num_ord=NUM_ORD)
    else:
        raise ValueError("Unknown model_name. Use 'swin_ord' or 'swin_meta_ord'.")

    ckpt = torch.load(weights_path, map_location=device)
    state = _extract_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_swin_ord(
    model: nn.Module,
    image: Union[str, Path, Image.Image],
    device: Optional[Union[str, torch.device]] = None,
    thresholds: Optional[Union[np.ndarray, list, tuple]] = None,
) -> PredictionResult:
    device = get_device(device)
    img_tf = make_image_transform()
    pil_img = load_image(image)
    x = img_tf(pil_img).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    boundary_probs = torch.sigmoid(logits).cpu().numpy()[0]
    thresholds_arr = np.asarray(thresholds if thresholds is not None else DEFAULT_THRESHOLDS, dtype=np.float32)
    if thresholds_arr.shape[0] != NUM_ORD:
        raise ValueError(f"Expected {NUM_ORD} thresholds, got {thresholds_arr.shape[0]}")

    pred = int((boundary_probs > thresholds_arr).sum())
    class_probs = _ordinal_to_class_probs(boundary_probs)
    conf = float(class_probs[pred])

    return PredictionResult(
        model_name="swin_ord",
        kl_grade=pred,
        confidence=conf,
        raw_output=boundary_probs.tolist(),
        probabilities={f"KL{i}": float(class_probs[i]) for i in range(len(class_probs))},
        stage_label=kl_to_stage_label(pred),
    )


@torch.no_grad()
def predict_swin_meta_ord(
    model: nn.Module,
    image: Union[str, Path, Image.Image],
    metadata: Dict[str, Any],
    scaler: Any,
    device: Optional[Union[str, torch.device]] = None,
    thresholds: Optional[Union[np.ndarray, list, tuple]] = None,
) -> PredictionResult:
    device = get_device(device)
    img_tf = make_image_transform()
    pil_img = load_image(image)
    x = img_tf(pil_img).unsqueeze(0).to(device)

    meta_arr = prepare_swin_metadata(metadata, scaler=scaler)
    meta_t = torch.tensor(meta_arr, dtype=torch.float32, device=device)

    model.eval()
    logits = model(x, meta_t)
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise ValueError("Model produced NaN or Inf logits. Check metadata values, scaler, and checkpoint.")

    boundary_probs = torch.sigmoid(logits).cpu().numpy()[0]
    thresholds_arr = np.asarray(thresholds if thresholds is not None else DEFAULT_THRESHOLDS, dtype=np.float32)
    if thresholds_arr.shape[0] != NUM_ORD:
        raise ValueError(f"Expected {NUM_ORD} thresholds, got {thresholds_arr.shape[0]}")

    pred = int((boundary_probs > thresholds_arr).sum())
    class_probs = _ordinal_to_class_probs(boundary_probs)
    conf = float(class_probs[pred])

    return PredictionResult(
        model_name="swin_meta_ord",
        kl_grade=pred,
        confidence=conf,
        raw_output=boundary_probs.tolist(),
        probabilities={f"KL{i}": float(class_probs[i]) for i in range(len(class_probs))},
        stage_label=kl_to_stage_label(pred),
    )


def predict_knee_oa(
    model: nn.Module,
    model_name: str,
    image: Union[str, Path, Image.Image],
    metadata: Optional[Dict[str, Any]] = None,
    scaler: Any = None,
    device: Optional[Union[str, torch.device]] = None,
    thresholds: Optional[Union[np.ndarray, list, tuple]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    model_name = model_name.lower().strip()
    model_name = {
        "swin_img": "swin_ord",
        "swin_meta_multi": "swin_meta_ord",
    }.get(model_name, model_name)

    if model_name == "swin_ord":
        return predict_swin_ord(model, image, device=device, thresholds=thresholds).to_dict()

    if model_name == "swin_meta_ord":
        if scaler is None:
            raise ValueError(
                "A fitted scaler is required for swin_meta_ord. "
                "Load the training scaler (.pkl) and pass it in."
            )
        return predict_swin_meta_ord(
            model=model,
            image=image,
            metadata=metadata,
            scaler=scaler,
            device=device,
            thresholds=thresholds,
        ).to_dict()

    raise ValueError("Unknown model_name. Use 'swin_ord' or 'swin_meta_ord'.")