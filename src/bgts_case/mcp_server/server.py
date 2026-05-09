"""MCP server exposing a single read-only tool over the ITSM tickets table."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from bgts_case.mcp_server.crud import get_ticket_by_id
from bgts_case.mcp_server.models import TicketDTO

mcp: FastMCP = FastMCP("bgts-itsm-tickets")


@mcp.tool
def get_ticket_with_id(
    ticket_id: Annotated[
        str,
        Field(
            description=(
                "Business ticket identifier such as 'INC-2024-001'. "
                "Case-sensitive; must match exactly."
            ),
        ),
    ],
) -> TicketDTO:
    """Fetch a single ITSM ticket by its business ID.

    Returns the full ticket payload (title, category, status, priority,
    reporter, error messages, affected services, resolution summary, etc.)
    as a Pydantic DTO. Raises a `ToolError` if no ticket with that ID
    exists so the calling agent surfaces a clear error rather than a
    silent null.
    """
    ticket = get_ticket_by_id(ticket_id)
    if ticket is None:
        raise ToolError(f"No ticket found with ticket_id={ticket_id!r}")
    return ticket


def run_ticket_mcp_server() -> None:
    """Entry point: start the ITSM ticket MCP server over stdio."""
    mcp.run(transport="stdio")
