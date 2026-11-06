"""LLM extractor using Claude with a forced tool call, so the output always
matches the Pydantic schema."""
from __future__ import annotations

import os

from .schema import NoteExtraction

SYSTEM = """You extract structured data from UK clinical notes.
Rules:
- Only extract what is written. Never infer diagnoses or add medications.
- For every item, `evidence` must be copied EXACTLY from the note (a short span).
- Expand abbreviations in names (T2DM -> type 2 diabetes, HTN -> hypertension, AF -> atrial fibrillation).
- Problem status is "history" if described as past/known/PMH, otherwise "current".
- Allergies: use substance "none known" for NKDA. Leave the list empty if not mentioned.
- Vitals names: blood_pressure, heart_rate, temperature, spo2, respiratory_rate."""

_FIELDS = ["medications", "vitals", "problems", "allergies", "follow_up"]


def _tool_schema() -> dict:
    full = NoteExtraction.model_json_schema()
    return {
        "type": "object",
        "properties": {k: full["properties"][k] for k in _FIELDS},
        "required": _FIELDS,
        "$defs": full["$defs"],
    }


def extract_llm(note: str, model: str | None = None) -> NoteExtraction:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or os.getenv("LLM_MODEL", "claude-haiku-5-5"),
        max_tokens=2000,
        system=SYSTEM,
        tools=[{
            "name": "record_extraction",
            "description": "Record the structured contents of the clinical note.",
            "input_schema": _tool_schema(),
        }],
        tool_choice={"type": "tool", "name": "record_extraction"},
        messages=[{"role": "user", "content": f"<note>\n{note}\n</note>"}],
    )
    data = next(b.input for b in resp.content if b.type == "tool_use")
    return NoteExtraction(**data, method="llm")
