from __future__ import annotations

import enum
from datetime import datetime, timedelta

from sqlalchemy import Text, case, cast, func

from bgts_case.mcp_server.models import (
    AggregationBucket,
    AggregationGroupBy,
    EnumValuesDTO,
    RelatedTicketsDTO,
    Ticket,
    TicketCategory,
    TicketDTO,
    TicketOrderBy,
    TicketPriority,
    TicketStatus,
    TicketSummaryDTO,
)
from bgts_case.mcp_server.session import get_session

# Per-tool clamps. Kept here so the tool layer stays thin.
SEARCH_LIMIT_MAX = 50
AGGREGATE_LIMIT_MAX = 100
GET_TICKETS_MAX = 20
RELATED_LIMIT_PER_CHANNEL_MAX = 20
LOOKBACK_DAYS_MAX = 365
SAME_CATEGORY_FIXED_WINDOW_DAYS = 30
SAME_ERROR_PATTERN_PREFIX = 30


_GROUP_BY_COLUMN = {
    AggregationGroupBy.CATEGORY: Ticket.category,
    AggregationGroupBy.SUB_CATEGORY: Ticket.sub_category,
    AggregationGroupBy.AFFECTED_SYSTEM: Ticket.affected_system,
    AggregationGroupBy.ASSIGNED_TEAM: Ticket.assigned_team,
    AggregationGroupBy.PRIORITY: Ticket.priority,
    AggregationGroupBy.STATUS: Ticket.status,
}


def _priority_rank_desc():
    """Sort key putting critical first, then high, medium, low."""
    return case(
        (Ticket.priority == TicketPriority.CRITICAL, 4),
        (Ticket.priority == TicketPriority.HIGH, 3),
        (Ticket.priority == TicketPriority.MEDIUM, 2),
        (Ticket.priority == TicketPriority.LOW, 1),
        else_=0,
    ).desc()


def get_ticket_by_id(ticket_id: str) -> TicketDTO | None:
    """Return the ticket whose business `ticket_id` matches, or `None`."""
    with get_session() as session:
        ticket = session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        return ticket.to_dto() if ticket is not None else None


def get_tickets_by_ids(ticket_ids: list[str]) -> list[TicketDTO]:
    """Return full detail for multiple tickets. Missing IDs are skipped."""
    if not ticket_ids:
        return []
    ids = ticket_ids[:GET_TICKETS_MAX]
    with get_session() as session:
        rows = session.query(Ticket).filter(Ticket.ticket_id.in_(ids)).all()
        return [r.to_dto() for r in rows]


def search_tickets(
    *,
    category: TicketCategory | None = None,
    priority: TicketPriority | None = None,
    status: TicketStatus | None = None,
    sub_category_contains: str | None = None,
    opened_after: datetime | None = None,
    opened_before: datetime | None = None,
    affected_system_contains: str | None = None,
    assigned_team: str | None = None,
    text_query: str | None = None,
    error_message_contains: str | None = None,
    has_resolution: bool | None = None,
    limit: int = 20,
    order_by: TicketOrderBy = TicketOrderBy.OPENED_AT_DESC,
) -> list[TicketSummaryDTO]:
    """Filter tickets and return TicketSummaryDTOs. See server tool for arg docs."""
    limit = max(1, min(limit, SEARCH_LIMIT_MAX))
    with get_session() as session:
        q = session.query(Ticket)

        if category is not None:
            q = q.filter(Ticket.category == category)
        if priority is not None:
            q = q.filter(Ticket.priority == priority)
        if status is not None:
            q = q.filter(Ticket.status == status)
        if sub_category_contains:
            q = q.filter(Ticket.sub_category.ilike(f"%{sub_category_contains}%"))
        if opened_after is not None:
            q = q.filter(Ticket.opened_at >= opened_after)
        if opened_before is not None:
            q = q.filter(Ticket.opened_at <= opened_before)
        if affected_system_contains:
            q = q.filter(Ticket.affected_system.ilike(f"%{affected_system_contains}%"))
        if assigned_team:
            q = q.filter(Ticket.assigned_team == assigned_team)
        if text_query:
            doc = func.concat_ws(
                " ",
                func.coalesce(Ticket.title, ""),
                func.coalesce(Ticket.description, ""),
                func.coalesce(Ticket.resolution_summary, ""),
            )
            q = q.filter(
                func.to_tsvector("simple", doc).op("@@")(
                    func.plainto_tsquery("simple", text_query)
                )
            )
        if error_message_contains:
            # JSONB cast to text includes every element's value, so ILIKE
            # substring matching works for log fragments / error codes.
            q = q.filter(
                cast(Ticket.error_messages, Text).ilike(f"%{error_message_contains}%")
            )
        if has_resolution is True:
            q = q.filter(Ticket.resolution_summary.is_not(None))
        elif has_resolution is False:
            q = q.filter(Ticket.resolution_summary.is_(None))

        match order_by:
            case TicketOrderBy.OPENED_AT_DESC:
                q = q.order_by(Ticket.opened_at.desc())
            case TicketOrderBy.OPENED_AT_ASC:
                q = q.order_by(Ticket.opened_at.asc())
            case TicketOrderBy.PRIORITY_DESC:
                q = q.order_by(_priority_rank_desc(), Ticket.opened_at.desc())
            case TicketOrderBy.RESOLUTION_TIME_ASC:
                q = q.order_by(Ticket.resolution_time_hours.asc().nulls_last())
            case TicketOrderBy.RESOLUTION_TIME_DESC:
                q = q.order_by(Ticket.resolution_time_hours.desc().nulls_last())

        rows = q.limit(limit).all()
        return [r.to_summary_dto() for r in rows]


def aggregate_tickets(
    *,
    group_by: AggregationGroupBy,
    category: TicketCategory | None = None,
    priority: TicketPriority | None = None,
    time_range_days: int = 90,
    min_count: int = 1,
    limit: int = 20,
) -> list[AggregationBucket]:
    """Group tickets by a column and return counts / averages per bucket."""
    time_range_days = max(1, min(time_range_days, LOOKBACK_DAYS_MAX))
    limit = max(1, min(limit, AGGREGATE_LIMIT_MAX))
    cutoff = datetime.now() - timedelta(days=time_range_days)

    col = _GROUP_BY_COLUMN[group_by]

    with get_session() as session:
        filters = [Ticket.opened_at >= cutoff]
        if category is not None:
            filters.append(Ticket.category == category)
        if priority is not None:
            filters.append(Ticket.priority == priority)

        agg_rows = (
            session.query(
                col.label("group_value"),
                func.count().label("total_count"),
                func.sum(
                    case((Ticket.priority == TicketPriority.CRITICAL, 1), else_=0)
                ).label("critical_count"),
                func.sum(
                    case((Ticket.priority == TicketPriority.HIGH, 1), else_=0)
                ).label("high_count"),
                func.sum(
                    case(
                        (
                            Ticket.status.in_(
                                [TicketStatus.OPEN, TicketStatus.INVESTIGATING]
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label("open_or_investigating_count"),
                func.avg(Ticket.resolution_time_hours).label("avg_resolution_hours"),
                func.max(Ticket.opened_at).label("most_recent_opened_at"),
            )
            .filter(*filters)
            .group_by(col)
            .having(func.count() >= min_count)
            .order_by(func.count().desc())
            .limit(limit)
            .all()
        )

        group_values = [r.group_value for r in agg_rows]
        recent_map: dict[object, str] = {}
        if group_values:
            recent_rows = (
                session.query(col, Ticket.ticket_id)
                .filter(*filters)
                .filter(col.in_(group_values))
                .distinct(col)
                .order_by(col, Ticket.opened_at.desc())
                .all()
            )
            recent_map = {row[0]: row[1] for row in recent_rows}

    def _stringify(v: object) -> str:
        if v is None:
            return "(none)"
        if isinstance(v, enum.Enum):
            return str(v.value)
        return str(v)

    return [
        AggregationBucket(
            group_value=_stringify(r.group_value),
            total_count=int(r.total_count),
            critical_count=int(r.critical_count or 0),
            high_count=int(r.high_count or 0),
            open_or_investigating_count=int(r.open_or_investigating_count or 0),
            avg_resolution_hours=(
                float(r.avg_resolution_hours)
                if r.avg_resolution_hours is not None
                else None
            ),
            most_recent_opened_at=r.most_recent_opened_at,
            most_recent_ticket_id=recent_map.get(r.group_value),
        )
        for r in agg_rows
    ]


def get_related_tickets(
    *,
    ticket_id: str,
    lookback_days: int = 90,
    limit_per_channel: int = 5,
) -> RelatedTicketsDTO | None:
    """Gather tickets related to `ticket_id` across four signal channels."""
    lookback_days = max(1, min(lookback_days, LOOKBACK_DAYS_MAX))
    limit_per_channel = max(1, min(limit_per_channel, RELATED_LIMIT_PER_CHANNEL_MAX))
    now = datetime.now()
    cutoff = now - timedelta(days=lookback_days)
    fixed_30 = now - timedelta(days=SAME_CATEGORY_FIXED_WINDOW_DAYS)

    with get_session() as session:
        src = session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        if src is None:
            return None

        explicit_ids = list(src.past_similar_incidents or [])
        if explicit_ids:
            explicit_rows = (
                session.query(Ticket)
                .filter(Ticket.ticket_id.in_(explicit_ids))
                .limit(limit_per_channel)
                .all()
            )
        else:
            explicit_rows = []

        if src.affected_system:
            same_sys_rows = (
                session.query(Ticket)
                .filter(Ticket.affected_system == src.affected_system)
                .filter(Ticket.ticket_id != ticket_id)
                .filter(Ticket.opened_at >= cutoff)
                .order_by(Ticket.opened_at.desc())
                .limit(limit_per_channel)
                .all()
            )
        else:
            same_sys_rows = []

        errs = list(src.error_messages or [])
        if errs and errs[0]:
            needle = errs[0][:SAME_ERROR_PATTERN_PREFIX]
            same_err_rows = (
                session.query(Ticket)
                .filter(cast(Ticket.error_messages, Text).ilike(f"%{needle}%"))
                .filter(Ticket.ticket_id != ticket_id)
                .filter(Ticket.opened_at >= cutoff)
                .order_by(Ticket.opened_at.desc())
                .limit(limit_per_channel)
                .all()
            )
        else:
            same_err_rows = []

        same_cat_rows = (
            session.query(Ticket)
            .filter(Ticket.category == src.category)
            .filter(Ticket.ticket_id != ticket_id)
            .filter(Ticket.opened_at >= fixed_30)
            .order_by(Ticket.opened_at.desc())
            .limit(limit_per_channel)
            .all()
        )

        return RelatedTicketsDTO(
            source_ticket_id=ticket_id,
            explicit_links=[r.to_summary_dto() for r in explicit_rows],
            same_affected_system=[r.to_summary_dto() for r in same_sys_rows],
            same_error_pattern=[r.to_summary_dto() for r in same_err_rows],
            same_category_recent=[r.to_summary_dto() for r in same_cat_rows],
        )


def list_enum_values() -> EnumValuesDTO:
    """Return the literal valid values for the filter enums."""
    return EnumValuesDTO(
        categories=[c.value for c in TicketCategory],
        priorities=[p.value for p in TicketPriority],
        statuses=[s.value for s in TicketStatus],
    )
