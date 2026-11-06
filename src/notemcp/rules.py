"""Rule-based extractor. Works offline, is fully transparent, and acts as a
baseline to compare the LLM against. Lexicons cover common UK primary-care terms."""
from __future__ import annotations

import re

from .schema import Allergy, FollowUp, Medication, NoteExtraction, Problem, Vital

DRUGS = [
    "metformin", "gliclazide", "amlodipine", "ramipril", "lisinopril", "losartan", "bisoprolol",
    "atorvastatin", "simvastatin", "apixaban", "rivaroxaban", "warfarin", "aspirin", "clopidogrel",
    "salbutamol", "beclometasone", "levothyroxine", "omeprazole", "lansoprazole", "paracetamol",
    "ibuprofen", "naproxen", "codeine", "sertraline", "citalopram", "fluoxetine", "furosemide",
    "amoxicillin", "doxycycline", "nitrofurantoin", "prednisolone", "insulin glargine",
]
FREQ = r"(OD|BD|TDS|QDS|QD|PRN|nocte|mane|once daily|twice daily|three times daily|weekly|stat)"
UNIT = r"(mg|mcg|micrograms|g|ml|units)"

PROBLEMS = {
    "type 2 diabetes": [r"T2DM", r"type 2 diabetes", r"T2 diabetes"],
    "hypertension": [r"HTN", r"hypertension"],
    "atrial fibrillation": [r"AF", r"atrial fibrillation"],
    "asthma": [r"asthma"],
    "COPD": [r"COPD"],
    "chronic kidney disease": [r"CKD(?: stage \d)?", r"chronic kidney disease"],
    "ischaemic heart disease": [r"IHD", r"ischaemic heart disease"],
    "myocardial infarction": [r"MI", r"myocardial infarction", r"heart attack"],
    "stroke": [r"CVA", r"stroke"],
    "heart failure": [r"heart failure", r"CCF", r"HFrEF"],
    "hypothyroidism": [r"hypothyroidism", r"underactive thyroid"],
    "depression": [r"depression", r"low mood"],
    "anxiety": [r"anxiety"],
    "GORD": [r"GORD", r"reflux"],
    "osteoarthritis": [r"osteoarthritis", r"OA"],
    "hyperlipidaemia": [r"hyperlipidaemia", r"high cholesterol"],
    "urinary tract infection": [r"UTI", r"urinary tract infection"],
    "community acquired pneumonia": [r"CAP", r"pneumonia"],
}
HISTORY_CUES = r"\b(PMH|hx|h/o|history of|previous|prev\.?|known|post)\b"

VITALS = [
    ("blood_pressure", r"\b(?:BP|blood pressure)[:\s]*(\d{2,3}/\d{2,3})(?:\s?mmHg)?", "mmHg"),
    ("heart_rate", r"\b(?:(?i:HR|pulse)|P)[:\s]*(\d{2,3})(?:\s?bpm)?\b", "bpm"),
    ("temperature", r"\b(?:temp|T)[:\s]*(\d{2}(?:\.\d)?)\s?(?:°?C)?\b", "C"),
    ("spo2", r"\b(?:SpO2|sats|O2 sats)[:\s]*(\d{2,3})\s?%", "%"),
    ("respiratory_rate", r"\b(?:RR|resp rate)[:\s]*(\d{1,2})\b", "breaths/min"),
]

NO_ALLERGY = r"\b(NKDA|NKA|no known (?:drug )?allergies|no allergies known|allerg(?:y|ies)[:\s]+(?:none|nil)\b)"
ALLERGY_CLAUSE = r"(?:allergic to|allerg(?:y|ies)[:\s]+)[^.;\n]*"

FOLLOW_UP = r"((?:review|f/u|follow[- ]up|refer(?:ral)?|repeat|recheck|safety[- ]?net(?:ted)?)[^.;\n]*)"


def _clause_start(note: str, pos: int) -> int:
    return max(note.rfind(c, 0, pos) for c in ".;\n") + 1


def extract_rules(note: str) -> NoteExtraction:
    out = NoteExtraction(method="rules")
    allergy_spans = [m.span() for m in re.finditer(ALLERGY_CLAUSE, note, re.I)]

    for drug in DRUGS:
        pat = rf"\b({drug})\b\s*(\d*\.?\d+)?\s*{UNIT}?\s*{FREQ}?"
        for m in re.finditer(pat, note, re.I):
            if any(a <= m.start() < b for a, b in allergy_spans):
                continue  # a drug the patient is allergic to is NOT a medication
            dose = float(m.group(2)) if m.group(2) else None
            out.medications.append(Medication(
                name=drug, dose=dose, unit=m.group(3), frequency=m.group(4),
                evidence=m.group(0).strip(),
            ))

    for name, patterns in PROBLEMS.items():
        for p in patterns:
            flags = 0 if p.isupper() or re.search(r"[A-Z]{2}", p) else re.I  # abbreviations are case-sensitive
            m = re.search(rf"\b{p}\b", note, flags)
            if m:
                clause = note[_clause_start(note, m.start()):m.start()]
                status = "history" if re.search(HISTORY_CUES, clause, re.I) else "current"
                out.problems.append(Problem(name=name, status=status, evidence=m.group(0)))
                break

    for name, pat, unit in VITALS:
        m = re.search(pat, note, 0 if name == "heart_rate" else re.I)
        if m:
            out.vitals.append(Vital(name=name, value=m.group(1), unit=unit, evidence=m.group(0).strip()))

    m = re.search(NO_ALLERGY, note, re.I)
    if m:
        out.allergies.append(Allergy(substance="none known", evidence=m.group(0)))
    else:
        m = re.search(r"(?:allergic to|allerg(?:y|ies)[:\s]+)\s*([A-Za-z ,/-]+?)(?=[.;\n(]|$)", note, re.I)
        if m:
            for sub in re.split(r",|/| and ", m.group(1)):
                sub = sub.strip()
                if sub:
                    out.allergies.append(Allergy(substance=sub.lower(), evidence=sub))

    for m in re.finditer(FOLLOW_UP, note, re.I):
        out.follow_up.append(FollowUp(action=m.group(1).strip(), evidence=m.group(1).strip()))

    return out
