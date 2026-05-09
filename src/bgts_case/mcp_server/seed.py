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

_STATUS_MAP = {
    "Açık": TicketStatus.OPEN,
    "İnceleniyor": TicketStatus.INVESTIGATING,
    "Çözümlendi": TicketStatus.RESOLVED,
    "Kapatıldı": TicketStatus.CLOSED,
}

_PRIORITY_MAP = {
    "Düşük": TicketPriority.LOW,
    "Orta": TicketPriority.MEDIUM,
    "Yüksek": TicketPriority.HIGH,
    "Kritik": TicketPriority.CRITICAL,
}

_CATEGORY_MAP = {
    "Network/WAN": TicketCategory.NETWORK_WAN,
    "Firewall": TicketCategory.FIREWALL,
    "VPN": TicketCategory.VPN,
    "Switch": TicketCategory.SWITCH,
    "DNS": TicketCategory.DNS,
    "SD-WAN": TicketCategory.SD_WAN,
    "Wireless": TicketCategory.WIRELESS,
    "Sunucu": TicketCategory.SERVER,
    "Güvenlik": TicketCategory.SECURITY,
    "Email": TicketCategory.EMAIL,
}


def _to_ticket(raw: dict) -> Ticket:
    return Ticket(
        ticket_id=raw["id"],
        title=raw["baslik"],
        category=_CATEGORY_MAP[raw["kategori"]],
        sub_category=raw.get("alt_kategori"),
        priority=_PRIORITY_MAP[raw["oncelik"]],
        status=_STATUS_MAP[raw["durum"]],
        opened_at=datetime.fromisoformat(raw["acilis_tarihi"]),
        last_updated_at=datetime.fromisoformat(raw["son_guncelleme"]),
        reporter=raw["bildiren"],
        assigned_team=raw.get("atanan_ekip"),
        affected_user_count=raw.get("etkilenen_kullanici_sayisi"),
        affected_system=raw.get("etkilenen_sistem"),
        description=raw.get("aciklama"),
        error_messages=raw.get("hata_mesajlari", []),
        affected_services=raw.get("etkilenen_servisler", []),
        past_similar_incidents=raw.get("gecmis_benzer_olaylar", []),
        resolution_time_hours=raw.get("cozum_suresi_saat"),
        resolution_summary=raw.get("cozum_ozeti"),
    )


def seed_tickets() -> None:
    """Entry point: load `static/itsm_tickets.json` into the tickets table.

    Idempotent — if the table already has rows, the seed is skipped.
    """
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    raw_tickets: list[dict] = payload["tickets"]

    with get_session() as session:
        if session.query(Ticket).first() is not None:
            print("tickets table already populated; skipping seed")
            return
        session.add_all(_to_ticket(r) for r in raw_tickets)
        print(f"inserted {len(raw_tickets)} tickets")
