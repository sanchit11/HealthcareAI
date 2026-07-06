"""
presidio_service.py  —  PHI Masking layer using Microsoft Presidio

Masks Patient PHI / sensitive data BEFORE it is sent to any LLM (OpenAI or Ollama).

Install:
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg

Usage:
    from presidio_service import mask_phi, mask_phi_in_dict

Supported PHI entities (HIPAA Safe Harbor + clinical extras):
    PERSON, PHONE_NUMBER, EMAIL_ADDRESS, LOCATION, DATE_TIME,
    US_SSN, US_PASSPORT, US_DRIVER_LICENSE, CREDIT_CARD,
    IBAN_CODE, IP_ADDRESS, URL, MEDICAL_LICENSE, NRP,
    MRN (custom regex recognizer)
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy-load Presidio so server starts even if not yet installed ─────────────
_analyzer  = None
_anonymizer = None


def _get_engines():
    """Load Presidio engines once and cache them."""
    global _analyzer, _anonymizer

    if _analyzer and _anonymizer:
        return _analyzer, _anonymizer

    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        from presidio_analyzer import PatternRecognizer, Pattern
    except ImportError:
        raise RuntimeError(
            "Presidio is not installed.\n"
            "Run:\n"
            "  pip install presidio-analyzer presidio-anonymizer spacy\n"
            "  python -m spacy download en_core_web_lg\n"
            "Then restart the FastAPI server."
        )

    # ── NLP engine: use spaCy en_core_web_lg for best accuracy ───────────────
    try:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
    except Exception:
        # Fall back to the smaller model if lg is not downloaded
        logger.warning(
            "en_core_web_lg not found — falling back to en_core_web_sm. "
            "For better PHI detection run: python -m spacy download en_core_web_lg"
        )
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

    # ── Custom recognizer: Medical Record Number (MRN) ───────────────────────
    # Matches: "MRN: 123456", "MRN#1234567", "Medical Record 00123456"
    mrn_pattern = Pattern(
        name="mrn_pattern",
        regex=r"\b(?:MRN|Medical\s+Record(?:\s+Number)?|Chart)[:\s#]*\d{4,12}\b",
        score=0.85,
    )
    mrn_recognizer = PatternRecognizer(
        supported_entity="MRN",
        name="MrnRecognizer",
        patterns=[mrn_pattern],
    )

    # ── Custom recognizer: NPI (National Provider Identifier) ────────────────
    npi_pattern = Pattern(
        name="npi_pattern",
        regex=r"\b(?:NPI)[:\s#]*\d{10}\b",
        score=0.90,
    )
    npi_recognizer = PatternRecognizer(
        supported_entity="NPI",
        name="NpiRecognizer",
        patterns=[npi_pattern],
    )

    # ── Custom recognizer: DEA Number ────────────────────────────────────────
    dea_pattern = Pattern(
        name="dea_pattern",
        regex=r"\b[A-Z]{2}\d{7}\b",
        score=0.75,
    )
    dea_recognizer = PatternRecognizer(
        supported_entity="DEA_NUMBER",
        name="DeaRecognizer",
        patterns=[dea_pattern],
    )

    # ── Build registry with built-ins + custom recognizers ───────────────────
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    registry.add_recognizer(mrn_recognizer)
    registry.add_recognizer(npi_recognizer)
    registry.add_recognizer(dea_recognizer)

    _analyzer  = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    _anonymizer = AnonymizerEngine()

    logger.info("✅ Presidio PHI masking engines loaded.")
    return _analyzer, _anonymizer


# ── PHI entity types to detect ────────────────────────────────────────────────
# Subset of Presidio's built-in entities relevant to healthcare / HIPAA
PHI_ENTITIES = [
    "PERSON",                # Patient and provider names
    "PHONE_NUMBER",          # All phone number formats
    "EMAIL_ADDRESS",         # Email addresses
    "LOCATION",              # Addresses, cities, states, ZIP codes
    "DATE_TIME",             # DOB, appointment dates, timestamps
    "US_SSN",                # Social Security Numbers
    "US_PASSPORT",           # Passport numbers
    "US_DRIVER_LICENSE",     # Driver's license numbers
    "CREDIT_CARD",           # Credit card numbers
    "IBAN_CODE",             # Bank account numbers
    "IP_ADDRESS",            # IP addresses
    "URL",                   # URLs (may embed patient IDs)
    "MEDICAL_LICENSE",       # Medical license numbers
    "NRP",                   # National Registration/ID
    "MRN",                   # Medical Record Number (custom)
    "NPI",                   # National Provider Identifier (custom)
    "DEA_NUMBER",            # DEA Number (custom)
]


def mask_phi(text: str, score_threshold: float = 0.60) -> str:
    """
    Detect and anonymize PHI in a plain-text string.

    Args:
        text:             Raw text that may contain PHI.
        score_threshold:  Minimum Presidio confidence score to mask (0.0–1.0).
                          Default 0.60 balances recall vs. false positives.

    Returns:
        Text with PHI replaced by <ENTITY_TYPE> tags,
        e.g. "John Smith" → "<PERSON>", "01/15/1980" → "<DATE_TIME>".
        Returns the original text unchanged on any error.
    """
    if not text or not text.strip():
        return text

    try:
        analyzer, anonymizer = _get_engines()

        results = analyzer.analyze(
            text=text,
            entities=PHI_ENTITIES,
            language="en",
            score_threshold=score_threshold,
        )

        if not results:
            return text  # No PHI found — return as-is

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        masked_count = len(results)
        logger.info(f"🔒 Presidio masked {masked_count} PHI item(s).")
        return anonymized.text

    except Exception as e:
        # Never block the pipeline on a masking failure — log and continue
        logger.error(f"❌ Presidio masking error: {e}")
        return text


def mask_phi_in_dict(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """
    Mask PHI in specific string fields of a dictionary (in-place copy).

    Args:
        data:   Dictionary (e.g. payload dict from model_dump()).
        fields: Dot-notation field paths to mask, e.g.
                ["ambient_details.raw_transcript", "ambient_details.diarized_transcript"]

    Returns:
        A shallow copy of the dict with specified string fields masked.

    Example:
        masked = mask_phi_in_dict(payload_dict, [
            "ambient_details.raw_transcript",
            "ambient_details.diarized_transcript",
        ])
    """
    import copy
    data = copy.deepcopy(data)

    for field_path in fields:
        keys = field_path.split(".")
        obj = data
        try:
            # Traverse to the parent
            for key in keys[:-1]:
                obj = obj[key]
            leaf_key = keys[-1]
            value = obj.get(leaf_key)
            if isinstance(value, str):
                obj[leaf_key] = mask_phi(value)
            elif isinstance(value, list):
                # Handle lists of strings (e.g. diarized_transcript lines)
                obj[leaf_key] = [
                    mask_phi(item) if isinstance(item, str) else item
                    for item in value
                ]
        except (KeyError, TypeError):
            logger.warning(f"mask_phi_in_dict: field path '{field_path}' not found — skipping.")

    return data


def mask_phi_in_text_fields(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively walk a dict and mask PHI in every string value.

    Use this for free-form nested dicts where you cannot enumerate
    all field paths up front (e.g. patient_context sub-sections).

    Args:
        data: Arbitrary nested dict.

    Returns:
        Deep copy with all string values masked.
    """
    import copy

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return mask_phi(obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj  # int, float, bool, None — leave unchanged

    return _walk(copy.deepcopy(data))