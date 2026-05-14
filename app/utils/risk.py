# utils/risk.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def get_risk_band(kl_grade: int) -> str:
    """
    Strict KL-to-risk mapping:
    KL 0-1 -> Low Risk
    KL 2-3 -> Medium Risk
    KL 4   -> High Risk
    """
    try:
        kl = int(kl_grade)
    except (TypeError, ValueError):
        return "Unknown"

    if kl <= 1:
        return "Low Risk"
    elif kl <= 3:
        return "Medium Risk"
    return "High Risk"


def get_stage_label(kl_grade: int) -> str:
    """
    Human-friendly disease stage label.
    """
    try:
        kl = int(kl_grade)
    except (TypeError, ValueError):
        return "Unknown stage"

    stage_map = {
        0: "No visible osteoarthritis",
        1: "Doubtful / very early osteoarthritis",
        2: "Mild osteoarthritis",
        3: "Moderate osteoarthritis",
        4: "Severe osteoarthritis",
    }
    return stage_map.get(kl, "Unknown stage")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class RiskResult:
    kl_grade: int
    base_band: str
    final_band: str
    stage_label: str
    confidence_note: str
    summary: str
    reasons: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_risk_result(
    kl_grade: int,
    *,
    age: Any = None,
    bmi: Any = None,
    pain: Any = None,
    jsn_medial: Any = None,
    jsn_lateral: Any = None,
    osteo_femur_medial: Any = None,
    osteo_tibia_lateral: Any = None,
    sclerosis_femur_medial: Any = None,
) -> Dict[str, Any]:
    """
    Return a structured risk summary for the UI.

    Important:
    - final_band is kept equal to base_band
    - metadata is shown only as explanation context
    """
    try:
        kl = int(kl_grade)
    except (TypeError, ValueError):
        kl = -1

    base_band = get_risk_band(kl)
    stage_label = get_stage_label(kl)

    final_band = base_band

    reasons = {
        "age": _safe_float(age),
        "bmi": _safe_float(bmi),
        "pain": _safe_float(pain),
        "jsn_medial": _safe_float(jsn_medial),
        "jsn_lateral": _safe_float(jsn_lateral),
        "osteo_femur_medial": _safe_float(osteo_femur_medial),
        "osteo_tibia_lateral": _safe_float(osteo_tibia_lateral),
        "sclerosis_femur_medial": _safe_float(sclerosis_femur_medial),
    }

    confidence_note = "Risk band is based only on the predicted KL grade."

    summary = (
        f""
    )

    return RiskResult(
        kl_grade=kl,
        base_band=base_band,
        final_band=final_band,
        stage_label=stage_label,
        confidence_note=confidence_note,
        summary=summary,
        reasons=reasons,
    ).to_dict()


def risk_badge_color(risk_band: str) -> str:
    mapping = {
        "Low Risk": "green",
        "Medium Risk": "orange",
        "High Risk": "red",
        "Unknown": "gray",
    }
    return mapping.get(risk_band, "gray")


def risk_badge_emoji(risk_band: str) -> str:
    mapping = {
        "Low Risk": "🟢",
        "Medium Risk": "🟡",
        "High Risk": "🔴",
        "Unknown": "⚪",
    }
    return mapping.get(risk_band, "⚪")


def format_risk_message(risk_data: Dict[str, Any]) -> str:
    kl = risk_data.get("kl_grade", "Unknown")
    base_band = risk_data.get("base_band", "Unknown")
    final_band = risk_data.get("final_band", "Unknown")
    stage = risk_data.get("stage_label", "Unknown stage")

    return (
        f"KL Grade: {kl}\n"
        f"Stage: {stage}\n"
        f"Base Risk: {base_band}\n"
        f"Final Risk: {final_band}"
    )


def prepare_llm_context(risk_data: Dict[str, Any]) -> str:
    reasons = risk_data.get("reasons", {})
    return (
        "Patient risk context:\n"
        f"- KL Grade: {risk_data.get('kl_grade', 'Unknown')}\n"
        f"- Base Risk: {risk_data.get('base_band', 'Unknown')}\n"
        f"- Final Risk: {risk_data.get('final_band', 'Unknown')}\n"
        f"- Stage: {risk_data.get('stage_label', 'Unknown stage')}\n"
        f"- Age: {reasons.get('age', 'Unknown')}\n"
        f"- BMI: {reasons.get('bmi', 'Unknown')}\n"
        f"- Pain: {reasons.get('pain', 'Unknown')}\n"
        f"- JSN Medial: {reasons.get('jsn_medial', 'Unknown')}\n"
        f"- JSN Lateral: {reasons.get('jsn_lateral', 'Unknown')}\n"
        f"- Osteo Femur Medial: {reasons.get('osteo_femur_medial', 'Unknown')}\n"
        f"- Osteo Tibia Lateral: {reasons.get('osteo_tibia_lateral', 'Unknown')}\n"
        f"- Sclerosis Femur Medial: {reasons.get('sclerosis_femur_medial', 'Unknown')}\n"
    )