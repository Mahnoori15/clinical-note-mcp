"""Safety checks.

1. Dose formats that are known to cause medication errors (based on the ISMP
   list of error-prone abbreviations): trailing zeros ("1.0 mg" misread as 10 mg),
   naked decimals (".5 mg" misread as 5 mg), "U"/"IU" for units, "QD"/"OD" ambiguity.
2. Grounding: any item whose evidence does not appear in the note is removed,
   so the model cannot invent findings.
"""
from __future__ import annotations

import re

from .schema import Flag, NoteExtraction

UNSAFE_PATTERNS = [
    (r"\b\d+\.0\s?(mg|mcg|g|ml|units?)\b", "Trailing zero (e.g. '1.0 mg' can be misread as 10 mg). Write '1 mg'."),
    (r"(?<![\d.])\.\d+\s?(mg|mcg|g|ml)\b", "Naked decimal (e.g. '.5 mg' can be misread as 5 mg). Write '0.5 mg'."),
    (r"\b\d+\s?(U|IU)\b", "'U'/'IU' can be misread as 0 or 4. Write 'units'."),
    (r"\b(QD|qd|Q\.D\.)\b", "'QD' can be confused with QID. Write 'once daily'."),
]


def dose_format_flags(note: str) -> list[Flag]:
    flags = []
    for pattern, advice in UNSAFE_PATTERNS:
        for m in re.finditer(pattern, note):
            flags.append(Flag(type="unsafe_dose_format", detail=f"'{m.group(0)}': {advice}"))
    return flags


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def enforce_grounding(result: NoteExtraction, note: str) -> NoteExtraction:
    """Drop any item whose evidence is not literally present in the note."""
    text = _norm(note)
    for field in ["medications", "vitals", "problems", "allergies", "follow_up"]:
        kept = []
        for item in getattr(result, field):
            if item.evidence and _norm(item.evidence) in text:
                kept.append(item)
            else:
                result.flags.append(Flag(
                    type="ungrounded_item_removed",
                    detail=f"{field}: '{item.model_dump(exclude={'evidence'})}' had no supporting text",
                ))
        setattr(result, field, kept)
    return result


def allergy_status_flag(result: NoteExtraction) -> list[Flag]:
    if not result.allergies:
        return [Flag(type="no_allergy_status", detail="No allergy status recorded in note. Please confirm.")]
    return []
