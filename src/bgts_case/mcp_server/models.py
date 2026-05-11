from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCategory(str, enum.Enum):
    NETWORK_WAN = "network_wan"
    FIREWALL = "firewall"
    VPN = "vpn"
    SWITCH = "switch"
    DNS = "dns"
    SD_WAN = "sd_wan"
    WIRELESS = "wireless"
    SERVER = "server"
    SECURITY = "security"
    EMAIL = "email"


class TicketOrderBy(str, enum.Enum):
    OPENED_AT_DESC = "opened_at_desc"
    OPENED_AT_ASC = "opened_at_asc"
    PRIORITY_DESC = "priority_desc"
    RESOLUTION_TIME_ASC = "resolution_time_asc"
    RESOLUTION_TIME_DESC = "resolution_time_desc"


class AggregationGroupBy(str, enum.Enum):
    CATEGORY = "category"
    SUB_CATEGORY = "sub_category"
    AFFECTED_SYSTEM = "affected_system"
    ASSIGNED_TEAM = "assigned_team"
    PRIORITY = "priority"
    STATUS = "status"


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512))
    category: Mapped[TicketCategory] = mapped_column(
        Enum(TicketCategory, name="ticket_category"), index=True
    )
    sub_category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    priority: Mapped[TicketPriority] = mapped_column(
        Enum(TicketPriority, name="ticket_priority"), index=True
    )
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus, name="ticket_status"), index=True
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False))
    reporter: Mapped[str] = mapped_column(String(128))
    assigned_team: Mapped[str | None] = mapped_column(String(128), nullable=True)
    affected_user_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    affected_system: Mapped[str | None] = mapped_column(String(256), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_messages: Mapped[list[str]] = mapped_column(JSONB, default=list)
    affected_services: Mapped[list[str]] = mapped_column(JSONB, default=list)
    past_similar_incidents: Mapped[list[str]] = mapped_column(JSONB, default=list)
    resolution_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dto(self) -> "TicketDTO":
        """Wrap this ORM row into its serializable Pydantic DTO."""
        return TicketDTO.model_validate(self)

    def to_summary_dto(self) -> "TicketSummaryDTO":
        """Wrap this ORM row into its truncated summary DTO."""
        errors = list(self.error_messages or [])
        desc = self.description
        preview = desc[:200] if desc else None
        return TicketSummaryDTO(
            ticket_id=self.ticket_id,
            title=self.title,
            category=self.category,
            sub_category=self.sub_category,
            priority=self.priority,
            status=self.status,
            opened_at=self.opened_at,
            resolution_time_hours=self.resolution_time_hours,
            affected_system=self.affected_system,
            affected_user_count=self.affected_user_count,
            description_preview=preview,
            error_messages_preview=errors[:1],
            error_messages_total=len(errors),
            past_similar_incidents=list(self.past_similar_incidents or []),
        )


class TicketDTO(BaseModel):
    """Serialization-friendly view of a `Ticket` row."""

    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    title: str
    category: TicketCategory
    sub_category: str | None
    priority: TicketPriority
    status: TicketStatus
    opened_at: datetime
    last_updated_at: datetime
    reporter: str
    assigned_team: str | None
    affected_user_count: int | None
    affected_system: str | None
    description: str | None
    error_messages: list[str]
    affected_services: list[str]
    past_similar_incidents: list[str]
    resolution_time_hours: float | None
    resolution_summary: str | None


class TicketSummaryDTO(BaseModel):
    """Truncated view of a ticket for list endpoints.

    The agent inspects summaries, then pulls full detail for the ones it
    cares about via `get_ticket` / `get_tickets`.
    """

    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    title: str
    category: TicketCategory
    sub_category: str | None
    priority: TicketPriority
    status: TicketStatus
    opened_at: datetime
    resolution_time_hours: float | None
    affected_system: str | None
    affected_user_count: int | None
    description_preview: str | None = Field(
        description="First 200 chars of description. Call get_ticket for full text."
    )
    error_messages_preview: list[str] = Field(
        description="First element of error_messages array only."
    )
    error_messages_total: int = Field(
        description="Total count of error_messages so the agent knows how many were hidden."
    )
    past_similar_incidents: list[str]


class AggregationBucket(BaseModel):
    """One row of the aggregate_tickets output."""

    group_value: str = Field(description="Group key value (e.g. 'vpn' or 'FGT-600E').")
    total_count: int
    critical_count: int
    high_count: int
    open_or_investigating_count: int = Field(
        description="Tickets still active (status in [open, investigating])."
    )
    avg_resolution_hours: float | None
    most_recent_opened_at: datetime | None
    most_recent_ticket_id: str | None


class RelatedTicketsDTO(BaseModel):
    """Related tickets gathered from four independent signal channels."""

    source_ticket_id: str
    explicit_links: list[TicketSummaryDTO] = Field(
        description="Tickets directly referenced in the source ticket's past_similar_incidents field."
    )
    same_affected_system: list[TicketSummaryDTO] = Field(
        description="Other tickets with the same affected_system within the lookback window."
    )
    same_error_pattern: list[TicketSummaryDTO] = Field(
        description="Tickets whose error_messages array contains a substring of the source's first error message."
    )
    same_category_recent: list[TicketSummaryDTO] = Field(
        description="Tickets in the same category opened within the last 30 days."
    )


class EnumValuesDTO(BaseModel):
    """Output of list_enum_values — valid values for filter parameters."""

    categories: list[str]
    priorities: list[str]
    statuses: list[str]
