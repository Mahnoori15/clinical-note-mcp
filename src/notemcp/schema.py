"""Structured output schema. Every extracted item carries `evidence`: the exact
text span from the note that supports it, so a clinician can verify it."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Item(BaseModel):
    evidence: str = Field(..., description="Exact text copied from the note that supports this item")


class Medication(Item):
    name: str
    dose: float | None = None
    unit: str | None = None
    frequency: str | None = None


class Vital(Item):
    name: Literal["blood_pressure", "heart_rate", "temperature", "spo2", "respiratory_rate"]
    value: str
    unit: str


class Problem(Item):
    name: str
    status: Literal["current", "history"] = "current"


class Allergy(Item):
    substance: str  # "none known" for NKDA


class FollowUp(Item):
    action: str


class Flag(BaseModel):
    type: Literal["unsafe_dose_format", "ungrounded_item_removed", "no_allergy_status"]
    detail: str


class NoteExtraction(BaseModel):
    medications: list[Medication] = []
    vitals: list[Vital] = []
    problems: list[Problem] = []
    allergies: list[Allergy] = []
    follow_up: list[FollowUp] = []
    flags: list[Flag] = []
    method: Literal["rules", "llm"] = "rules"
    disclaimer: str = "Decision support only. A clinician must review all output."
