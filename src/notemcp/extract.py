"""Main entry point: extract, then apply grounding and safety checks."""
from __future__ import annotations

import os

from .rules import extract_rules
from .safety import allergy_status_flag, dose_format_flags, enforce_grounding
from .schema import NoteExtraction


def structure_note(note: str, method: str = "auto") -> NoteExtraction:
    """method: 'rules', 'llm', or 'auto' (LLM if ANTHROPIC_API_KEY is set, else rules)."""
    if method == "auto":
        method = "llm" if os.getenv("ANTHROPIC_API_KEY") else "rules"
    if method == "llm":
        from .llm import extract_llm
        result = extract_llm(note)
    else:
        result = extract_rules(note)

    result = enforce_grounding(result, note)
    result.flags += dose_format_flags(note) + allergy_status_flag(result)
    return result
