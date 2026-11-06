"""Field-level precision / recall / F1 against the hand-labelled notes.

Usage:
    python eval/run_eval.py --method rules
    python eval/run_eval.py --method llm      # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from notemcp import structure_note

DATA = Path(__file__).parent.parent / "data" / "notes.jsonl"


def keys(result, field):
    if field == "medications":
        return {(m.name.lower(), m.dose) for m in result.medications}
    if field == "vitals":
        return {(v.name, v.value) for v in result.vitals}
    if field == "problems":
        return {(p.name.lower(), p.status) for p in result.problems}
    if field == "allergies":
        return {a.substance.lower() for a in result.allergies}


def gold_keys(gold, field):
    if field == "allergies":
        return {a.lower() for a in gold[field]}
    if field == "problems":
        return {(x[0].lower(), x[1]) for x in gold[field]}
    return {tuple(x) if field != "medications" else (x[0].lower(), x[1]) for x in gold[field]}


def evaluate(method: str = "rules", path=DATA) -> dict:
    fields = ["medications", "vitals", "problems", "allergies"]
    tp = dict.fromkeys(fields, 0); fp = dict.fromkeys(fields, 0); fn = dict.fromkeys(fields, 0)
    errors = []
    for line in open(path):
        case = json.loads(line)
        res = structure_note(case["note"], method)
        for f in fields:
            pred, gold = keys(res, f), gold_keys(case["gold"], f)
            tp[f] += len(pred & gold); fp[f] += len(pred - gold); fn[f] += len(gold - pred)
            if pred != gold:
                errors.append({"id": case["id"], "field": f,
                               "missed": sorted(map(str, gold - pred)), "extra": sorted(map(str, pred - gold))})
    out = {}
    for f in fields:
        p = tp[f] / (tp[f] + fp[f]) if tp[f] + fp[f] else 1.0
        r = tp[f] / (tp[f] + fn[f]) if tp[f] + fn[f] else 1.0
        out[f] = {"precision": round(p, 3), "recall": round(r, 3),
                  "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    out["micro_f1"] = round(sum(2 * tp[f] for f in fields) /
                            sum(2 * tp[f] + fp[f] + fn[f] for f in fields), 3)
    return {"method": method, "scores": out, "errors": errors}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="rules", choices=["rules", "llm"])
    ap.add_argument("--show-errors", action="store_true")
    a = ap.parse_args()
    res = evaluate(a.method)
    print(json.dumps(res["scores"], indent=2))
    if a.show_errors:
        print(json.dumps(res["errors"], indent=2))
