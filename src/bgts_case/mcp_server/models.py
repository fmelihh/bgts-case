from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, ConfigDict
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
