"""MCP server exposing the extractor as tools for any MCP client (e.g. Claude Desktop).

Run:  python -m notemcp.server
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .extract import structure_note as _structure
from .safety import dose_format_flags

server = MCPServer(
    name="clinical-note-structurer",
    instructions="Turns free-text clinical notes into structured, evidence-linked data. "
                 "Output is decision support only and must be reviewed by a clinician.",
)


@server.tool()
def structure_note(note: str, method: str = "auto") -> dict:
    """Extract medications, vitals, problems, allergies and follow-up actions from a clinical note.
    Every item includes the exact text it came from. method: 'auto', 'rules' or 'llm'."""
    return _structure(note, method).model_dump()


@server.tool()
def check_dose_safety(text: str) -> list[dict]:
    """Flag error-prone dose notation (trailing zeros, naked decimals, 'U', 'QD')."""
    return [f.model_dump() for f in dose_format_flags(text)]


def main():
    server.run()


if __name__ == "__main__":
    main()
