# Clinical Note MCP

![CI](https://github.com/Mahnoori15/clinical-note-mcp/actions/workflows/ci.yml/badge.svg)

Turns messy free-text clinical notes into **structured, evidence-linked data**: medications, vital signs, problems, allergies and follow-up actions. It runs as an **MCP server**, so Claude Desktop or any other MCP client can use it as a tool.

> Synthetic data only. Decision support, not a medical device. A clinician must review all output.

**Input**
```
72M known AF on apixaban 5mg BD. Pulse 112. warfarin 1.0mg
```
**Output** (shortened)
```json
{
  "medications": [{"name": "apixaban", "dose": 5.0, "unit": "mg", "frequency": "BD", "evidence": "apixaban 5mg BD"}, ...],
  "vitals":      [{"name": "heart_rate", "value": "112", "unit": "bpm", "evidence": "Pulse 112"}],
  "problems":    [{"name": "atrial fibrillation", "status": "history", "evidence": "AF"}],
  "flags": [
    {"type": "unsafe_dose_format", "detail": "'1.0mg': Trailing zero (e.g. '1.0 mg' can be misread as 10 mg). Write '1 mg'."},
    {"type": "no_allergy_status",  "detail": "No allergy status recorded in note. Please confirm."}
  ]
}
```

## Why this matters
Clinicians write notes in shorthand ("T2DM", "BD", "NKDA"). Structured data is needed for coding, audits, decision support and handovers. Ambient AI scribes now generate many notes, so tools that **turn them into checkable structure without inventing anything** are increasingly important.

## Safety design
| Risk | What the system does |
|---|---|
| LLM invents a finding | **Grounding check**: every item must quote its supporting text, and items whose evidence isn't in the note are removed and flagged |
| An allergen gets listed as a medication | Drugs mentioned in an allergy clause are excluded from medications |
| Error-prone dose notation | Flags trailing zeros (`1.0mg`), naked decimals (`.5mg`), `U`/`IU` and `QD`, based on the ISMP list of error-prone abbreviations |
| Allergy status missing | Flagged for the clinician to confirm |
| Over-trust | Every response carries a decision-support disclaimer |

## Two extractors
- **Rules** (`rules.py`): offline, transparent regex and lexicon engine for common UK primary-care terms. Also serves as the baseline.
- **LLM** (`llm.py`): Claude with a forced tool call, so output always matches the Pydantic schema. Used automatically when `ANTHROPIC_API_KEY` is set.

Both go through the same grounding and safety layer (`safety.py`).

## Evaluation
15 hand-written synthetic notes in different styles (`data/notes.jsonl`), each with a hand-labelled answer key. Scored per field (precision, recall, F1).

```bash
python eval/run_eval.py --method rules --show-errors
python eval/run_eval.py --method llm       # needs ANTHROPIC_API_KEY
```

| Version | Micro-F1 | Notes |
|---|---|---|
| Rules v0 (notes never seen before) | **0.952** | Honest baseline |
| Rules v1 (after error analysis) | 1.000 | Optimistic, because it was tuned on the same notes |
| LLM | *run it* | |

**Error analysis behind v0 → v1:**
1. Allergens extracted as medications ("Allergy: ibuprofen"), which is a clinically dangerous mistake. Fixed by excluding drugs that appear in allergy clauses.
2. "Pulse 112" was missed because of case-sensitive matching.
3. "Post MI" was marked as current instead of history. Added "post" as a history cue.
4. "Allergies: none" was read as an allergen called "none".
5. One labelling mistake in the answer key (n08) was corrected.

**Next step:** write 15 *new* notes as a fresh test set, then compare rules with the LLM on notes neither has seen.

## Quick start
```bash
pip install -e ".[dev]"
pytest -q
python -c "from notemcp import structure_note; print(structure_note('On metformin 500mg BD. NKDA.').model_dump_json(indent=2))"
```

### Use it from Claude Desktop (MCP)
Add this to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "clinical-note": {
      "command": "clinical-note-mcp",
      "env": {"ANTHROPIC_API_KEY": "optional"}
    }
  }
}
```
Tools exposed: `structure_note(note, method)` and `check_dose_safety(text)`.

## Project structure
```
src/notemcp/
  schema.py    Pydantic output schema (every item has `evidence`)
  rules.py     rule-based extractor
  llm.py       Claude extractor with a forced tool call
  safety.py    grounding check and dose-safety flags
  extract.py   pipeline: extract -> ground -> flag
  server.py    MCP server
eval/run_eval.py   field-level precision / recall / F1
data/notes.jsonl   synthetic notes with answer keys
```
