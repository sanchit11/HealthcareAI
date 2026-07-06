"""
aiprompts/vital_trends.py
==========================
System prompt and user prompt builder for the Vital Trends AI agent.

The agent receives two vitals readings (previous + latest), computed deltas,
and patient context, then returns a structured clinical interpretation of the
trend — including per-vital significance, an overall status, and prioritised
action items.
"""

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a board-certified internal medicine AI integrated into an EHR system.

Your role is to analyse a patient's vital sign trends between two visits, interpret the clinical
significance of each change in the context of the patient's age, sex, active problems, and
current medications, and produce a structured clinical assessment with actionable recommendations.

CRITICAL RULES:
- Interpret deltas in the context of the patient's known conditions (e.g. a BP rise is more
  concerning in a known hypertensive on antihypertensives than in a healthy adult).
- Consider medication effects (e.g. a beta-blocker masking tachycardia, ACE inhibitor interaction
  with rising BP).
- Apply standard clinical thresholds:
    Blood Pressure: normal <120/80; elevated 120-129/<80; stage 1 HTN 130-139/80-89;
                    stage 2 HTN ≥140/90; hypertensive urgency ≥180/110
    Pulse: normal 60-100 bpm; tachycardia >100; bradycardia <60
    Temperature: normal 97.0-99.5°F; low-grade fever 99.6-100.3°F; fever >100.4°F
    Respiration: normal 12-20 br/min; tachypnoea >20; bradypnoea <12
    SpO2: normal ≥95%; concerning 91-94%; critical ≤90%
    BMI: underweight <18.5; normal 18.5-24.9; overweight 25-29.9; obese ≥30
- If a vital is null / missing for either reading, omit that row from vitals_trends and note
  it is unavailable — do not fabricate values.
- The overall status MUST reflect the single most urgent finding:
    Normal       — all vitals within normal limits, no concerning trend
    Stable       — minor deviations, no clinical action needed at this visit
    Monitoring Required — one or more vitals trending toward abnormal thresholds
    Concerning Trend    — one or more vitals outside normal range or worsening significantly
    Critical     — any vital in an immediately dangerous range
- urgency MUST align with overall status:
    Normal / Stable           → Routine
    Monitoring Required       → Within 1 Week
    Concerning Trend          → Prompt (Within 24-48 Hours)
    Critical                  → Urgent (Same Day) or Emergency
- Return ONLY valid JSON. No markdown, no prose outside the JSON structure.

Response JSON structure (follow EXACTLY):
{
  "status": "success",
  "patient_summary": {
    "pid": <integer>,
    "age": <integer>,
    "sex": "<string>",
    "active_problems": ["<string>"],
    "active_medications": ["<string>"],
    "allergies": ["<string>"]
  },
  "clinical_analysis": {
    "status": "<Normal | Stable | Monitoring Required | Concerning Trend | Critical>",
    "title": "One line summary of the overall trend (e.g. 'Rising Blood Pressure with Stable Pulse')",
    "overall_interpretation": "<narrative summary of the combined vital sign trend — 2-3 sentences>",
    "vitals_trends": [
      {
        "vital_sign": "<human-readable name>",
        "previous": "<formatted value with unit>",
        "latest": "<formatted value with unit>",
        "delta": "<e.g. '+20 / +12 mmHg' or '-3 bpm' or '+2.2%'>",
        "percent_change": <number or object e.g. {"systolic": 15.6, "diastolic": 14.6} for BP>,
        "conclusion": "<one-word interpretation to tell if it is elevated or dangerous compared to normal thresholds, e.g. 'Normal', 'Elevated', 'Stage 1 HTN', 'Tachycardia', 'Fever', 'Critical'. This basically alerts the clinician to the most concerning aspect of the trend for that vital sign.>",
        "clinical_significance": "<one-sentence clinical interpretation>"
      }
    ]
  },
  "recommendations": {
    "urgency": "<Routine | Within 1 Week | Prompt (Within 24-48 Hours) | Urgent (Same Day) | Emergency>",
    "action_items": ["<string>"]
  }
}
"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _fmt(value: Any, unit: str = "") -> str:
    """Format a scalar value + unit as a display string, or 'N/A' if missing."""
    if value is None:
        return "N/A"
    unit_str = f" {unit.strip()}" if unit else ""
    return f"{value}{unit_str}"


def _fmt_bp(systolic: Any, diastolic: Any) -> str:
    if systolic is None and diastolic is None:
        return "N/A"
    s = str(systolic) if systolic is not None else "?"
    d = str(diastolic) if diastolic is not None else "?"
    return f"{s}/{d} mmHg"


def _get_reading(readings: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    """Return the reading dict with reading == label, or None."""
    for r in readings:
        if r.get("reading", "").lower() == label.lower():
            return r
    return None


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    """Returns the static vital trends system prompt."""
    return SYSTEM_PROMPT


def generate_vital_trends_prompt(payload: Dict[str, Any]) -> str:
    """
    Build the user-turn prompt for the vital trends AI agent.

    Args:
        payload: The full validated VitalTrendsRequest as a dict.

    Returns:
        A JSON-serialised prompt string ready to send to the agent.
    """
    patient_info   = payload.get("patient_info", {})
    readings       = payload.get("vitals_readings", [])
    deltas         = payload.get("deltas", {})

    previous = _get_reading(readings, "previous")
    latest   = _get_reading(readings, "latest")

    # Build a human-readable vitals comparison table for the prompt
    vitals_comparison: List[Dict[str, Any]] = []

    def _add_row(
        name: str,
        prev_val: Any,
        latest_val: Any,
        unit: str,
        delta_info: Optional[Dict[str, Any]],
    ) -> None:
        if prev_val is None and latest_val is None:
            return  # Skip entirely missing vitals
        row: Dict[str, Any] = {
            "vital_sign": name,
            "previous_value": _fmt(prev_val, unit),
            "latest_value":   _fmt(latest_val, unit),
        }
        if delta_info:
            row["delta_value"]      = delta_info.get("value")
            row["direction"]        = delta_info.get("direction")
            row["percent_change"]   = delta_info.get("percent_change")
        vitals_comparison.append(row)

    if previous and latest:
        prev_bp   = previous.get("blood_pressure", {})
        latest_bp = latest.get("blood_pressure", {})
        if not (prev_bp.get("systolic") is None and prev_bp.get("diastolic") is None
                and latest_bp.get("systolic") is None and latest_bp.get("diastolic") is None):
            vitals_comparison.append({
                "vital_sign":     "Blood Pressure",
                "previous_value": _fmt_bp(prev_bp.get("systolic"), prev_bp.get("diastolic")),
                "latest_value":   _fmt_bp(latest_bp.get("systolic"), latest_bp.get("diastolic")),
                "delta_systolic":           deltas.get("systolic_bp", {}).get("value"),
                "delta_diastolic":          deltas.get("diastolic_bp", {}).get("value"),
                "direction_systolic":       deltas.get("systolic_bp", {}).get("direction"),
                "direction_diastolic":      deltas.get("diastolic_bp", {}).get("direction"),
                "percent_change_systolic":  deltas.get("systolic_bp", {}).get("percent_change"),
                "percent_change_diastolic": deltas.get("diastolic_bp", {}).get("percent_change"),
            })

        _add_row(
            "Pulse",
            previous.get("pulse", {}).get("value"),
            latest.get("pulse", {}).get("value"),
            "bpm",
            deltas.get("pulse"),
        )
        _add_row(
            "Temperature",
            previous.get("temperature", {}).get("value"),
            latest.get("temperature", {}).get("value"),
            "°F",
            deltas.get("temperature"),
        )
        _add_row(
            "Respiration Rate",
            previous.get("respiration", {}).get("value"),
            latest.get("respiration", {}).get("value"),
            "br/min",
            deltas.get("respiration"),
        )
        _add_row(
            "Oxygen Saturation",
            previous.get("oxygen_saturation", {}).get("value"),
            latest.get("oxygen_saturation", {}).get("value"),
            "%",
            deltas.get("oxygen_saturation"),
        )
        _add_row(
            "Weight",
            previous.get("weight", {}).get("value"),
            latest.get("weight", {}).get("value"),
            "lbs",
            deltas.get("weight"),
        )
        _add_row(
            "BMI",
            previous.get("bmi", {}).get("value"),
            latest.get("bmi", {}).get("value"),
            "",
            deltas.get("bmi"),
        )

    prompt_payload = {
        "task": (
            "Analyse the vital sign trends between the two visits. "
            "For each vital sign that changed, assess the clinical significance in the context "
            "of the patient's active problems, medications, and demographic profile. "
            "Produce an overall status, a narrative interpretation, and prioritised action items."
        ),
        "patient_info": patient_info,
        "previous_reading_date": previous.get("date") if previous else None,
        "latest_reading_date":   latest.get("date") if latest else None,
        "vitals_comparison":     vitals_comparison,
        "raw_deltas":            deltas,
    }

    return (
        "Analyse the following patient vital sign data and return a structured clinical assessment.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )
