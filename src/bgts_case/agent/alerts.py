"""Alert hooks for the agents layer.

Single general-purpose entry point — :func:`mock_slack_alert` — that pretends
to push a message to Slack but only writes to the local logger. Swap it for a
real Slack/Teams/PagerDuty client by passing a different callable with the
same signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

AlertFn = Callable[..., None]


def mock_slack_alert(
    message: str,
    *,
    channel: str = "alerts",
    level: str = "warning",
    **context: Any,
) -> None:
    """Pretend to post ``message`` to Slack ``channel``; just logs locally.

    ``context`` lets callers attach arbitrary structured fields (pdf name,
    doc title, counts, ...) without inventing a new function per call site.
    """
    extras = " ".join(f"{k}={v!r}" for k, v in context.items())
    suffix = f" | {extras}" if extras else ""
    logger.log(
        level.upper(),
        f"[SLACK mock channel={channel}] {message}{suffix}",
    )


__all__ = ["AlertFn", "mock_slack_alert"]
