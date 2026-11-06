import asyncio
import os

os.environ.pop("ANTHROPIC_API_KEY", None)  # tests run offline with the rule engine

from eval.run_eval import evaluate
from notemcp import structure_note
from notemcp.safety import dose_format_flags, enforce_grounding
from notemcp.schema import Medication, NoteExtraction

NOTE = ("58M. PMH: T2DM, HTN. Meds: metformin 500mg BD, amlodipine 5mg OD. "
        "Allergic to penicillin. BP 152/94, HR 88. Review in 2 weeks.")


def test_extracts_core_fields():
    r = structure_note(NOTE, "rules")
    assert {(m.name, m.dose, m.frequency) for m in r.medications} == {
        ("metformin", 500, "BD"), ("amlodipine", 5, "OD")}
    assert {(v.name, v.value) for v in r.vitals} == {("blood_pressure", "152/94"), ("heart_rate", "88")}
    assert {(p.name, p.status) for p in r.problems} == {
        ("type 2 diabetes", "history"), ("hypertension", "history")}
    assert [a.substance for a in r.allergies] == ["penicillin"]
    assert r.follow_up and "Review in 2 weeks" in r.follow_up[0].action


def test_every_item_is_grounded_in_note():
    r = structure_note(NOTE, "rules")
    for field in ["medications", "vitals", "problems", "allergies", "follow_up"]:
        for item in getattr(r, field):
            assert item.evidence.lower() in NOTE.lower()


def test_allergen_is_not_extracted_as_medication():
    r = structure_note("Allergy: ibuprofen. On aspirin 75mg OD.", "rules")
    assert [m.name for m in r.medications] == ["aspirin"]
    assert [a.substance for a in r.allergies] == ["ibuprofen"]


def test_grounding_removes_hallucinated_item():
    fake = NoteExtraction(medications=[Medication(name="warfarin", dose=5, evidence="warfarin 5mg")])
    out = enforce_grounding(fake, "Patient on aspirin 75mg OD.")
    assert out.medications == []
    assert out.flags[0].type == "ungrounded_item_removed"


def test_unsafe_dose_formats_flagged():
    flags = dose_format_flags("warfarin 1.0mg OD, digoxin .5mg, insulin 10U QD")
    assert len(flags) == 4


def test_missing_allergy_status_flagged():
    r = structure_note("On metformin 500mg BD.", "rules")
    assert any(f.type == "no_allergy_status" for f in r.flags)


def test_eval_set_score():
    assert evaluate("rules")["scores"]["micro_f1"] >= 0.95


def test_mcp_server_lists_and_calls_tools():
    from notemcp.server import server

    async def run():
        tools = {t.name for t in await server.list_tools()}
        assert {"structure_note", "check_dose_safety"} <= tools
        return await server.call_tool("check_dose_safety", {"text": "give .5mg"})

    assert asyncio.run(run())
