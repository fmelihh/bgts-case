from __future__ import annotations

from bgts_case.mcp_server.models import Ticket, TicketDTO
from bgts_case.mcp_server.session import get_session


def get_ticket_by_id(ticket_id: str) -> TicketDTO | None:
    """Return the ticket whose business `ticket_id` matches, or `None`.

    Owns its own session lifecycle so callers (e.g. MCP tools) only need to
    consume the resulting DTO.
    """
    with get_session() as session:
        ticket = (
            session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        )
        return ticket.to_dto() if ticket is not None else None
