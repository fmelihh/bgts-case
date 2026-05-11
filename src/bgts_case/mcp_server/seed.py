from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bgts_case.mcp_server.models import (
    Ticket,
    TicketCategory,
    TicketPriority,
    TicketStatus,
)
from bgts_case.mcp_server.session import get_session

JSON_PATH = Path(__file__).resolve().parents[3] / "static" / "itsm_tickets.json"


def _to_ticket(raw: dict) -> Ticket:
    return Ticket(
        ticket_id=raw["ticket_id"],
        title=raw["title"],
        category=TicketCategory[raw["category"]],
        sub_category=raw.get("sub_category"),
        priority=TicketPriority[raw["priority"]],
        status=TicketStatus[raw["status"]],
        opened_at=datetime.fromisoformat(raw["opened_at"]),
        last_updated_at=datetime.fromisoformat(raw["last_updated_at"]),
        reporter=raw["reporter"],
        assigned_team=raw.get("assigned_team"),
        affected_user_count=raw.get("affected_user_count"),
        affected_system=raw.get("affected_system"),
        description=raw.get("description"),
        error_messages=raw.get("error_messages", []),
        affected_services=raw.get("affected_services", []),
        past_similar_incidents=raw.get("past_similar_incidents", []),
        resolution_time_hours=raw.get("resolution_time_hours"),
        resolution_summary=raw.get("resolution_summary"),
    )


def seed_tickets() -> None:
    """Entry point: load `static/itsm_tickets.json` into the tickets table.

    Idempotent — if the table already has rows, the seed is skipped.
    """
    raw_tickets: list[dict] = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    with get_session() as session:
        if session.query(Ticket).first() is not None:
            print("tickets table already populated; skipping seed")
            return
        session.add_all(_to_ticket(r) for r in raw_tickets)
        print(f"inserted {len(raw_tickets)} tickets")
