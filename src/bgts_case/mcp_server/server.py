"""MCP server exposing read-only tools over the ITSM tickets table."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from urllib.parse import urlparse

from fastmcp import FastMCP
from pydantic import Field

from bgts_case.mcp_server import crud
from bgts_case.mcp_server.models import (
    AggregationBucket,
    AggregationGroupBy,
    EnumValuesDTO,
    RelatedTicketsDTO,
    TicketCategory,
    TicketDTO,
    TicketOrderBy,
    TicketPriority,
    TicketStatus,
    TicketSummaryDTO,
)
from bgts_case.secret import secrets

mcp: FastMCP = FastMCP("bgts-itsm-tickets")


@mcp.tool
def get_ticket(
    ticket_id: Annotated[
        str,
        Field(description="Ticket ID such as 'INC-2024-002'. Case-sensitive."),
    ],
) -> TicketDTO | None:
    """Return the FULL detail of a single ticket.

    Includes description, all error_messages, and resolution_summary.

    WHEN TO USE:
      - You know the ticket ID and need the complete content.
      - You are drilling down on a ticket surfaced by `search_tickets` or
        `get_related_tickets`.
      - You need to read a past ticket's resolution_summary to validate an
        RCA hypothesis.

    WHEN NOT TO USE:
      - You need multiple tickets' detail → use `get_tickets` (one round-trip).
      - You only need to filter / list tickets → use `search_tickets`.

    Returns the TicketDTO, or None if no ticket has that ID.
    """
    return crud.get_ticket_by_id(ticket_id)


@mcp.tool
def get_tickets(
    ticket_ids: Annotated[
        list[str],
        Field(description="List of ticket IDs. Max 20 (extras are dropped)."),
    ],
) -> list[TicketDTO]:
    """Return FULL detail for multiple tickets in a single call.

    WHEN TO USE:
      - You want to inspect all tickets referenced in a `past_similar_incidents` list.
      - After `get_related_tickets`, you need `resolution_summary` for several
        related tickets at once.
      - During RCA you are comparing multiple historical incidents.

    WHEN NOT TO USE:
      - For a single ticket → use `get_ticket`.
      - For more than 20 tickets → narrow with `search_tickets` and work with
        TicketSummaryDTO.

    Missing IDs are silently skipped; order is not preserved.
    """
    return crud.get_tickets_by_ids(ticket_ids)


@mcp.tool
def search_tickets(
    category: Annotated[
        TicketCategory | None,
        Field(description="Restrict to one category."),
    ] = None,
    priority: Annotated[
        TicketPriority | None,
        Field(description="Restrict to one priority."),
    ] = None,
    status: Annotated[
        TicketStatus | None,
        Field(description="Restrict to one status."),
    ] = None,
    sub_category_contains: Annotated[
        str | None,
        Field(description="ILIKE substring match on sub_category."),
    ] = None,
    opened_after: Annotated[
        datetime | None,
        Field(description="Tickets opened on or after this timestamp."),
    ] = None,
    opened_before: Annotated[
        datetime | None,
        Field(description="Tickets opened on or before this timestamp."),
    ] = None,
    affected_system_contains: Annotated[
        str | None,
        Field(
            description="ILIKE substring match on affected_system (e.g. 'FGT-600E')."
        ),
    ] = None,
    assigned_team: Annotated[
        str | None,
        Field(description="Exact match on team name (case-sensitive)."),
    ] = None,
    text_query: Annotated[
        str | None,
        Field(
            description=(
                "Postgres full-text search over title + description + "
                "resolution_summary. Use specific technical terms; generic "
                "words ('connection problem') will not match well."
            )
        ),
    ] = None,
    error_message_contains: Annotated[
        str | None,
        Field(
            description=(
                "ILIKE substring inside any element of the error_messages "
                "array — ideal for log fragments and error codes "
                "('MACFLAP_NOTIF', '%BGP-3-NOTIFICATION', 'DHCPNAK')."
            )
        ),
    ] = None,
    has_resolution: Annotated[
        bool | None,
        Field(
            description=(
                "True → only resolved tickets; False → only unresolved; "
                "None → no filter."
            )
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Max rows (1-50; values above 50 are clamped)."),
    ] = 20,
    order_by: Annotated[
        TicketOrderBy,
        Field(description="Sort order."),
    ] = TicketOrderBy.OPENED_AT_DESC,
) -> list[TicketSummaryDTO]:
    """Filter tickets and return TRUNCATED summaries (TicketSummaryDTO).

    WHEN TO USE:
      - Pattern search ("CRITICAL firewall tickets in the last 30 days").
      - Looking for past similar incidents by error_message substring.
      - Pre-triage state checks ("how many P1 tickets are still open?").
      - Drill-down after `aggregate_tickets` into a specific bucket.

    WHEN NOT TO USE:
      - You need full content of a single ticket → `get_ticket`.
      - You need related tickets across multiple signals → `get_related_tickets`.
      - You only need counts / grouping → `aggregate_tickets` (cheaper).

    QUERY TIPS:
      - `text_query` runs PostgreSQL `to_tsvector` / `plainto_tsquery` over
        title + description + resolution_summary. Use SPECIFIC terms — error
        codes, device names, protocol terms.
            GOOD: "MTU keepalive hold timer"
            BAD:  "connection problem"
      - `error_message_contains` does ILIKE substring across error_messages
        JSONB array. Ideal for log substrings ("MACFLAP_NOTIF", "DHCPNAK").
      - `affected_system_contains` is ILIKE ("FGT-600E", "DC01").
      - All filters combine with AND.

    Returns TicketSummaryDTOs (description truncated to 200 chars,
    error_messages to first element). Call `get_ticket` for full content.
    """
    return crud.search_tickets(
        category=category,
        priority=priority,
        status=status,
        sub_category_contains=sub_category_contains,
        opened_after=opened_after,
        opened_before=opened_before,
        affected_system_contains=affected_system_contains,
        assigned_team=assigned_team,
        text_query=text_query,
        error_message_contains=error_message_contains,
        has_resolution=has_resolution,
        limit=limit,
        order_by=order_by,
    )


@mcp.tool
def aggregate_tickets(
    group_by: Annotated[
        AggregationGroupBy,
        Field(description="Column to group by."),
    ],
    category: Annotated[
        TicketCategory | None,
        Field(description="Restrict to one category (None → all)."),
    ] = None,
    priority: Annotated[
        TicketPriority | None,
        Field(description="Restrict to one priority (None → all)."),
    ] = None,
    time_range_days: Annotated[
        int,
        Field(description="Days back to include (1-365, default 90)."),
    ] = 90,
    min_count: Annotated[
        int,
        Field(description="Drop groups with fewer than this many tickets."),
    ] = 1,
    limit: Annotated[
        int,
        Field(description="Max buckets (1-100, default 20)."),
    ] = 20,
) -> list[AggregationBucket]:
    """Group and aggregate tickets — for trend, pattern, and capacity analysis.

    WHEN TO USE:
      - "Which affected_system has caused the most issues in the last 90 days?"
      - "Which category has the longest average resolution time?"
      - In RCA: confirm whether a pattern is recurring vs. first-time.
      - Capacity planning and prioritizing preventive actions.

    WHEN NOT TO USE:
      - You need specific tickets → `search_tickets`.
      - You need related tickets for one source ticket → `get_related_tickets`.

    Returns AggregationBuckets sorted by total_count descending. Each bucket
    contains total / critical / high counts, still-open count, average
    resolution hours, and the most recent ticket ID + its opened_at.
    """
    return crud.aggregate_tickets(
        group_by=group_by,
        category=category,
        priority=priority,
        time_range_days=time_range_days,
        min_count=min_count,
        limit=limit,
    )


@mcp.tool
def get_related_tickets(
    ticket_id: Annotated[
        str,
        Field(description="Source ticket ID."),
    ],
    lookback_days: Annotated[
        int,
        Field(
            description=(
                "Window for same_affected_system and same_error_pattern "
                "channels (1-365, default 90). same_category_recent is always 30 days."
            )
        ),
    ] = 90,
    limit_per_channel: Annotated[
        int,
        Field(description="Max results per channel (1-20, default 5)."),
    ] = 5,
) -> RelatedTicketsDTO | None:
    """Gather tickets related to a source ticket via four independent channels.

    WHEN TO USE:
      - The "history search" step of RCA — pulls past_similar_incidents,
        same-system, same-error-pattern, and same-category tickets in one call.
      - Determining whether an incident is isolated or part of a pattern.
      - Gathering evidence before forming a hypothesis (multiple channels =
        stronger signal).

    CHANNELS:
      1. explicit_links — direct references from the source ticket's
         `past_similar_incidents` field.
      2. same_affected_system — same `affected_system` within the lookback window.
      3. same_error_pattern — error_messages contains a substring (first 30
         chars) of the source's first error_message.
      4. same_category_recent — same category opened in the last 30 days
         (fixed window regardless of `lookback_days`).

    Channels may overlap. Overlap is INTENTIONALLY not deduplicated: a ticket
    appearing in both explicit_links AND same_error_pattern is a strong signal.

    Returns RelatedTicketsDTO, or None if the source ticket is not found.
    Use `get_tickets` for batch full detail of channel results.
    """
    return crud.get_related_tickets(
        ticket_id=ticket_id,
        lookback_days=lookback_days,
        limit_per_channel=limit_per_channel,
    )


@mcp.tool
def list_enum_values() -> EnumValuesDTO:
    """Return valid enum values for use in filter parameters.

    WHEN TO USE:
      - Before calling `search_tickets` or `aggregate_tickets` when you are
        unsure which category / priority / status values are valid.
      - When the user asks "what categories exist?" or similar.

    WHEN NOT TO USE:
      - You already know the correct enum value — do not call redundantly.
    """
    return crud.list_enum_values()


def run_ticket_mcp_server() -> None:
    """Entry point: start the ITSM ticket MCP server over HTTP."""
    parsed = urlparse(secrets.mcp_server_url)
    mcp.run(
        transport="http",
        host=parsed.hostname or "localhost",
        port=parsed.port or 8765,
        path=parsed.path or "/mcp/",
    )
