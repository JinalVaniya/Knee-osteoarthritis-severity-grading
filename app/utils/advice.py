# utils/advice.py

from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

GROQ_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    Groq = None  # type: ignore

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _stage_from_kl(kl: Any) -> str:
    kl_i = _safe_int(kl)
    if kl_i is None:
        return "unknown"
    if kl_i <= 1:
        return "early"
    if kl_i <= 3:
        return "moderate"
    return "severe"


def _build_patient_context(
    kl_grade: Any,
    age: Any = None,
    bmi: Any = None,
    pain: Any = None,
    risk_band: Optional[str] = None,
    stage_label: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "kl_grade": _safe_int(kl_grade),
        "stage": _stage_from_kl(kl_grade),
        "age": _safe_float(age),
        "bmi": _safe_float(bmi),
        "pain": _safe_float(pain),
        "risk_band": risk_band,
        "stage_label": stage_label,
        "extra_metadata": extra_metadata or {},
    }


def build_advice_prompt(ctx: Dict[str, Any]) -> str:
    """
    Stage-aware prompt so the suggestions differ by KL grade.
    """
    kl = ctx.get("kl_grade")
    stage = ctx.get("stage")
    age = ctx.get("age")
    bmi = ctx.get("bmi")
    pain = ctx.get("pain")
    risk_band = ctx.get("risk_band") or "not provided"
    stage_label = ctx.get("stage_label") or "not provided"
    extra_json = json.dumps(ctx.get("extra_metadata", {}), ensure_ascii=False, indent=2)

    return f"""
You are a careful health education assistant for knee osteoarthritis.

Your job is to give personalized, stage-specific lifestyle guidance based on the patient details below.

PATIENT CONTEXT
- KL Grade: {kl}
- OA Stage: {stage}
- Risk Band: {risk_band}
- Stage Label: {stage_label}
- Age: {age if age is not None else "not provided"}
- BMI: {bmi if bmi is not None else "not provided"}
- Pain level: {pain if pain is not None else "not provided"}

EXTRA METADATA
{extra_json}

IMPORTANT RULES
- Do NOT diagnose.
- Do NOT prescribe medication.
- Do NOT mention exact drug names.
- Do NOT present surgery as a default suggestion.
- Use simple language that a non-medical person can understand.
- Make the advice clearly different for early, moderate, and severe stages.
- Avoid overly generic answers.
- Keep recommendations practical and safe.
- If the stage is severe, reduce exercise intensity and emphasize medical follow-up.
- If BMI is high, mention weight management in a supportive way.
- If pain is high, mention symptom monitoring and clinical review.

STAGE-SPECIFIC GUIDANCE
If KL 0-1 (early):
- Focus on prevention and joint protection.
- Encourage gentle activity and strengthening.
- Emphasize maintaining healthy weight and routine movement.

If KL 2-3 (moderate):
- Focus on controlled, low-impact movement.
- Suggest physiotherapy-style strengthening and mobility work.
- Mention avoiding high-impact or pain-triggering activities more clearly.
- Emphasize symptom management and monitoring.

If KL 4 (severe):
- Focus on protecting the joint and reducing strain.
- Keep exercise suggestions very gentle and supervised if possible.
- Strongly recommend seeing a clinician/orthopedic specialist.
- Mention that long walking, stairs, kneeling, and high-load activities may be difficult.
- Avoid sounding too reassuring.

Return the answer in this exact format:

Summary:
- 2 to 4 short bullet points explaining the situation in plain language.

Exercise:
- 3 to 5 bullet points with stage-appropriate movement suggestions.
- Make these different depending on KL stage.

Avoid:
- 3 to 5 bullet points with activities or habits that may worsen symptoms.

Diet:
- 3 to 5 bullet points with general nutrition guidance that may help with weight control or inflammation.

When to seek medical help:
- 1 to 3 bullet points mentioning red flags or when to follow up with a doctor.

Keep the total response concise.
""".strip()


def _extract_sectioned_content(text: str) -> Dict[str, List[str]]:
    sections = {
        "Summary": [],
        "Exercise": [],
        "Avoid": [],
        "Diet": [],
        "When to seek medical help": [],
    }

    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower().rstrip(":")
        if lowered == "summary":
            current = "Summary"
            continue
        if lowered == "exercise":
            current = "Exercise"
            continue
        if lowered == "avoid":
            current = "Avoid"
            continue
        if lowered == "diet":
            current = "Diet"
            continue
        if lowered == "when to seek medical help":
            current = "When to seek medical help"
            continue

        if line.startswith("-") and current:
            item = line.lstrip("-").strip()
            if item:
                sections[current].append(item)
            continue

        if current:
            sections[current].append(line)

    return sections


def _clean_llm_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fallback_advice(ctx: Dict[str, Any]) -> Dict[str, Any]:
    kl = ctx.get("kl_grade", "unknown")
    stage = ctx.get("stage", "unknown")
    bmi = ctx.get("bmi")
    pain = ctx.get("pain")

    summary = [
        f"Your model output suggests KL Grade {kl} ({stage} stage).",
        "This is a general education summary, not a medical diagnosis.",
    ]

    if stage == "early":
        summary.append("The focus is usually on prevention, healthy movement, and protecting the joint.")
    elif stage == "moderate":
        summary.append("The focus is usually on controlled movement, strengthening, and symptom management.")
    elif stage == "severe":
        summary.append("The focus is usually on reducing strain, gentle movement, and clinical follow-up.")

    if bmi is not None and bmi >= 30:
        summary.append("A higher BMI can increase knee load, so weight management may help reduce stress on the joint.")
    if pain is not None and pain >= 6:
        summary.append("Higher pain levels usually mean you should be more cautious and consider clinical review.")

    if stage == "early":
        exercise = [
            "Low-impact walking in short comfortable sessions",
            "Gentle range-of-motion exercises",
            "Light strengthening for the thigh and hip muscles",
            "Regular movement to keep the joint active",
        ]
        avoid = [
            "High-impact running or jumping",
            "Long periods of sitting without movement",
            "Deep squats if they cause discomfort",
            "Heavy lifting without guidance",
        ]
    elif stage == "moderate":
        exercise = [
            "Controlled low-impact walking if symptoms allow",
            "Physiotherapy-style strengthening exercises",
            "Gentle mobility work for the knee and hip",
            "Short, frequent activity breaks instead of one long session",
        ]
        avoid = [
            "High-impact sports",
            "Movements that trigger sharp pain",
            "Long standing without rest",
            "Deep squats or repeated stair climbing if painful",
        ]
    else:
        exercise = [
            "Very gentle movement to avoid stiffness",
            "Short walks only if tolerated",
            "Supervised physiotherapy if available",
            "Simple range-of-motion exercises as advised by a clinician",
        ]
        avoid = [
            "Long walks that worsen pain",
            "High-impact exercise",
            "Kneeling and deep bending",
            "Heavy joint loading or intense stair climbing",
        ]

    diet = [
        "Aim for balanced meals with vegetables, fruits, and enough protein",
        "If BMI is elevated, gradual weight loss may reduce knee stress",
        "Limit highly processed foods and sugary drinks",
        "Stay hydrated",
    ]

    when_help = [
        "See a clinician if pain is worsening, swelling is increasing, or walking becomes difficult",
        "Seek medical advice if you develop sudden severe pain or new instability",
    ]

    return {
        "raw_text": "",
        "sections": {
            "Summary": summary,
            "Exercise": exercise,
            "Avoid": avoid,
            "Diet": diet,
            "When to seek medical help": when_help,
        },
        "source": "fallback",
        "context": ctx,
    }


def _call_groq(prompt: str, model: str = "llama3-70b-8192") -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    if not GROQ_AVAILABLE:
        raise RuntimeError("groq package is not installed. Run: pip install groq")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical education assistant. "
                    "You provide general, non-diagnostic, stage-specific lifestyle guidance."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical education assistant. "
                    "You provide general, non-diagnostic, stage-specific lifestyle guidance."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def generate_advice_llm(
    kl_grade: Any,
    age: Any = None,
    bmi: Any = None,
    pain: Any = None,
    risk_band: Optional[str] = None,
    stage_label: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    provider: str = "groq",
    groq_model: str = "llama3-70b-8192",
    openai_model: str = "gpt-4o-mini",
    return_sections: bool = True,
) -> Dict[str, Any]:
    ctx = _build_patient_context(
        kl_grade=kl_grade,
        age=age,
        bmi=bmi,
        pain=pain,
        risk_band=risk_band,
        stage_label=stage_label,
        extra_metadata=extra_metadata,
    )

    prompt = build_advice_prompt(ctx)

    try:
        if provider.lower() == "groq":
            raw_text = _call_groq(prompt, model=groq_model)
            source = "groq"
        elif provider.lower() == "openai":
            raw_text = _call_openai(prompt, model=openai_model)
            source = "openai"
        else:
            raise ValueError("provider must be 'groq' or 'openai'")

        raw_text = _clean_llm_text(raw_text)

        if return_sections:
            sections = _extract_sectioned_content(raw_text)
            if not any(sections.values()):
                fb = _fallback_advice(ctx)
                fb["source"] = source
                return fb

            return {
                "source": source,
                "raw_text": raw_text,
                "sections": sections,
                "context": ctx,
            }

        return {
            "source": source,
            "raw_text": raw_text,
            "sections": {},
            "context": ctx,
        }

    except Exception:
        fb = _fallback_advice(ctx)
        fb["source"] = "fallback"
        return fb


def advice_to_text(advice_result: Dict[str, Any]) -> str:
    if not advice_result:
        return ""

    if advice_result.get("raw_text"):
        return str(advice_result["raw_text"])

    sections = advice_result.get("sections", {})
    parts: List[str] = []

    for title in ["Summary", "Exercise", "Avoid", "Diet", "When to seek medical help"]:
        items = sections.get(title, [])
        if not items:
            continue
        parts.append(f"{title}:")
        for item in items:
            parts.append(f"- {item}")
        parts.append("")

    return "\n".join(parts).strip()


def advice_for_streamlit(advice_result: Dict[str, Any]) -> Dict[str, List[str]]:
    if not advice_result:
        return {
            "Summary": [],
            "Exercise": [],
            "Avoid": [],
            "Diet": [],
            "When to seek medical help": [],
        }

    return advice_result.get("sections", {
        "Summary": [],
        "Exercise": [],
        "Avoid": [],
        "Diet": [],
        "When to seek medical help": [],
    })


def build_disclaimer() -> str:
    return (
        "This information is for general education only and is not a medical diagnosis. "
        "Please consult a qualified clinician for personal medical advice."
    )