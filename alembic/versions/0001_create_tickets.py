"""create tickets table

Revision ID: 0001
Revises:
Create Date: 2026-05-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


ticket_status = sa.Enum(
    "OPEN",
    "INVESTIGATING",
    "RESOLVED",
    "CLOSED",
    name="ticket_status",
)
ticket_priority = sa.Enum(
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
    name="ticket_priority",
)
ticket_category = sa.Enum(
    "NETWORK_WAN",
    "FIREWALL",
    "VPN",
    "SWITCH",
    "DNS",
    "SD_WAN",
    "WIRELESS",
    "SERVER",
    "SECURITY",
    "EMAIL",
    name="ticket_category",
)


def upgrade() -> None:
    bind = op.get_bind()
    ticket_status.create(bind, checkfirst=True)
    ticket_priority.create(bind, checkfirst=True)
    ticket_category.create(bind, checkfirst=True)

    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticket_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("category", ticket_category, nullable=False),
        sa.Column("sub_category", sa.String(length=128), nullable=True),
        sa.Column("priority", ticket_priority, nullable=False),
        sa.Column("status", ticket_status, nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("last_updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("reporter", sa.String(length=128), nullable=False),
        sa.Column("assigned_team", sa.String(length=128), nullable=True),
        sa.Column("affected_user_count", sa.Integer(), nullable=True),
        sa.Column("affected_system", sa.String(length=256), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "error_messages",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "affected_services",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "past_similar_incidents",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("resolution_time_hours", sa.Float(), nullable=True),
        sa.Column("resolution_summary", sa.Text(), nullable=True),
        sa.UniqueConstraint("ticket_id", name="uq_tickets_ticket_id"),
    )
    op.create_index("ix_tickets_ticket_id", "tickets", ["ticket_id"])
    op.create_index("ix_tickets_category", "tickets", ["category"])
    op.create_index("ix_tickets_priority", "tickets", ["priority"])
    op.create_index("ix_tickets_status", "tickets", ["status"])
    op.create_index("ix_tickets_opened_at", "tickets", ["opened_at"])


def downgrade() -> None:
    op.drop_index("ix_tickets_opened_at", table_name="tickets")
    op.drop_index("ix_tickets_status", table_name="tickets")
    op.drop_index("ix_tickets_priority", table_name="tickets")
    op.drop_index("ix_tickets_category", table_name="tickets")
    op.drop_index("ix_tickets_ticket_id", table_name="tickets")
    op.drop_table("tickets")
    bind = op.get_bind()
    ticket_category.drop(bind, checkfirst=True)
    ticket_priority.drop(bind, checkfirst=True)
    ticket_status.drop(bind, checkfirst=True)
