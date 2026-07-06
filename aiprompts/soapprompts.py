import json
from typing import List, Dict, Any

# SYSTEM PROMPT: Defines persona, clinical standards, and safety guardrails
SYSTEM_PROMPT = """You are an enterprise-grade clinical documentation AI integrated into an EHR.
Generate a medically accurate SOAP note using the inputs below.

Documentation Rules:
- Use concise professional physician documentation
- Preserve clinically meaningful details
- Include:
  - HPI
  - ROS-relevant symptoms
  - pertinent negatives
  - objective findings
  - prioritized differential/assessment
  - evidence-based plan
- Assessment should reflect medical decision-making
- Avoid redundancy
- Never fabricate findings
- If information is uncertain, omit it

Plan Section Requirements (MANDATORY):
- List ALL medications with FULL prescribing details using this exact format:
    Rx: [Generic name] ([Brand name]) [dose] [route] [frequency] x [duration] — Indication: [reason]
  Example: Rx: amoxicillin (Amoxil) 500mg PO TID x 10 days — Indication: bacterial sinusitis
- Include BOTH generic AND brand names for every medication
- Separate medications, labs, imaging, referrals, and follow-up into labeled subsections:
    ## Medications
    ## Labs / Diagnostics
    ## Imaging
    ## Referrals
    ## Patient Education
    ## Follow-Up
- If no medications are warranted, explicitly state: "No pharmacologic therapy indicated at this time"

Prior Physician-Approved Examples (when provided):
- The user prompt may include a PHYSICIAN-APPROVED PRIOR SOAP EXAMPLES block.
- These are SOAP drafts the physician previously rated and approved for this same patient.
- Each example carries per-section ratings (1–10) and a weight score (0.1–1.0).
- Use them to calibrate your writing style, terminology, level of detail, and structure.
- Weighting rule:
    weight 0.8–1.0 (8–10/10) → Very strongly mirror this note's style and structure
    weight 0.6–0.8 (6–8/10)  → Moderately follow this style
    weight 0.2–0.6 (2–6/10)  → Use as a loose reference only
- CRITICAL: Do NOT copy clinical content (diagnoses, medications, findings) from prior notes
  into the current note unless the current visit transcript independently supports it.
- Style elements to mirror: sentence structure, abbreviation patterns, level of detail,
  subsection formatting, and documentation verbosity.

Response JSON Structure:
{
    "subjective": "String containing fully detailed paragraph narrative for Subjective findings",
    "objective": "String containing fully detailed structured text for Objective data",
    "assessment": "String containing Medical Decision Making narrative, differentials, and diagnostic codes",
    "soap": "String containing overall care plan narrative with all required subsections",
    "current_vitals": "String containing current vital signs if provided in input, otherwise 'Not provided'",
    "vitals": "String containing vital signs if provided in SOAP message during conversation between Patient and Practitioner, otherwise 'Not provided'",
    "confidence_score": <float or decimal value representing overall documentation confidence between 0.0 and 1.0 based on clinical data consistency>
}
"""


def get_system_prompt() -> str:
    """Returns the static system instructions defining the agent's behavior."""
    return SYSTEM_PROMPT


def _build_prior_liked_notes_block(prior_notes: List[Dict[str, Any]]) -> str:
    """
    Renders prior liked SOAP notes into a clearly labelled prompt block.
    Notes are ordered highest-weight first so the LLM sees the most
    important examples at the top of the context window.
    """
    if not prior_notes:
        return ""

    sorted_notes = sorted(prior_notes, key=lambda n: n.get("weight", 0), reverse=True)

    lines = [
        "\n\n═══════════════════════════════════════════════════════════",
        "PHYSICIAN-APPROVED PRIOR SOAP EXAMPLES (style reference only)",
        "═══════════════════════════════════════════════════════════",
        "The physician approved the following SOAP drafts for this patient.",
        "Mirror the style, structure, and documentation detail of high-weight",
        "examples. Do NOT carry forward any clinical facts unless the current",
        "visit transcript independently supports them.",
        "",
    ]

    for i, note in enumerate(sorted_notes, start=1):
        # Derive weight from per-section ratings (avg / 10) when weight not pre-computed
        section_ratings = [
            note.get("ai_draft_subjective_rating"),
            note.get("ai_draft_objective_rating"),
            note.get("ai_draft_assessment_rating"),
            note.get("ai_draft_plan_rating"),
        ]
        valid_ratings = [r for r in section_ratings if r is not None]
        if valid_ratings:
            avg_rating = sum(valid_ratings) / len(valid_ratings)
            weight = note.get("weight", round(avg_rating / 10, 2))
        else:
            weight = note.get("weight", 0.5)

        date = note.get("date", "unknown date")

        if weight >= 0.8:
            importance = "VERY HIGH — mirror this style closely"
        elif weight >= 0.6:
            importance = "HIGH — follow this style"
        elif weight >= 0.4:
            importance = "MODERATE — use as loose reference"
        else:
            importance = "LOW — minor style reference"

        # Build per-section rating summary
        def _fmt_section(label: str, rating_key: str, feedback_key: str) -> str:
            r = note.get(rating_key)
            f = note.get(feedback_key, "")
            r_str = f"{r}/10" if r is not None else "unrated"
            f_str = f' — "{f}"' if f else ""
            return f"{label}: {r_str}{f_str}"

        rating_summary = " | ".join([
            _fmt_section("S", "ai_draft_subjective_rating",  "ai_draft_subjective_feedback"),
            _fmt_section("O", "ai_draft_objective_rating",   "ai_draft_objective_feedback"),
            _fmt_section("A", "ai_draft_assessment_rating",  "ai_draft_assessment_feedback"),
            _fmt_section("P", "ai_draft_plan_rating",        "ai_draft_plan_feedback"),
        ])

        lines += [
            f"--- Example {i} | Date: {date} | {rating_summary} | Weight: {weight:.2f} | Importance: {importance} ---",
            f"[SUBJECTIVE]\n{note.get('subjective', '').strip()}",
            f"[OBJECTIVE]\n{note.get('objective', '').strip()}",
            f"[ASSESSMENT]\n{note.get('assessment', '').strip()}",
            f"[PLAN]\n{note.get('plan', '').strip()}",
            "",
        ]

    lines.append("═══════════════════════════════════════════════════════════\n")
    return "\n".join(lines)


def generate_user_prompt(input_dict: dict) -> str:
    """
    USER PROMPT: Dynamically bundles the specific patient data into
    a prompt format for a single execution cycle.

    Prior liked notes (if any) are extracted from patient_context and rendered
    as a clearly labelled style-reference block BEFORE the raw session data,
    so the LLM sees them prominently in the context window.
    """
    patient_context = input_dict.get("patient_context", {})
    prior_notes     = patient_context.get("prior_liked_notes", [])
    prior_block     = _build_prior_liked_notes_block(prior_notes)

    # Remove prior_liked_notes from the raw JSON dump — already shown in prior_block
    clean_dict = json.loads(json.dumps(input_dict))
    if "prior_liked_notes" in clean_dict.get("patient_context", {}):
        del clean_dict["patient_context"]["prior_liked_notes"]

    return (
        "Please generate a medically accurate structured SOAP note matching the specified JSON format "
        "for the following patient session data.\n\n"
        "IMPORTANT: In the plan's Medications subsection, every drug must include: "
        "generic name, brand name, dose, route, frequency, duration, and indication.\n"
        + prior_block
        + "\n--- CURRENT VISIT DATA ---\n\n"
        + json.dumps(clean_dict, indent=2)
    )