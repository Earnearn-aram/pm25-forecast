"""The one webhook sender (KAN-18, moved here in KAN-19): collector, notify and heartbeat.

The webhook URL in ALERT_WEBHOOK_URL is a secret (Telegram and Discord put the token in the path):
it is only ever passed to httpx. It is never logged, printed or put in an exception message.
"""

import logging
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_S = 10.0
MAX_CHARS = 1900  # Discord rejects content over 2000 characters; Telegram's limit is 4096


class WebhookError(RuntimeError):
    """Delivery failed. The message carries the error type or HTTP status only, never the URL."""


def payload(text: str) -> dict[str, str]:
    """JSON body accepted by the supported webhooks.

    ``text`` is what Telegram's sendMessage (chat_id in the URL query) and Slack-style webhooks
    read; ``content`` is what Discord reads. Each service ignores the other key.
    """
    text = text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 3] + "..."
    return {"text": text, "content": text}


def post_message(webhook_url: str, text: str) -> None:
    """POST `text` to the webhook. Raises WebhookError (without the URL) on any failure."""
    try:
        response = httpx.post(webhook_url, json=payload(text), timeout=TIMEOUT_S)
    except Exception as exc:  # httpx puts the URL in its messages (InvalidURL too): drop them
        raise WebhookError(f"alert webhook failed: {type(exc).__name__}") from None
    if response.status_code >= 400:
        raise WebhookError(f"alert webhook returned HTTP {response.status_code}")


def warn_missing_webhook(what: str) -> None:
    logger.warning("ALERT_WEBHOOK_URL is not set: %s is logged only, not sent", what)


def make_alert(
    webhook_url: str | None, *, prefix: str = "[pm25 collector]"
) -> Callable[[str], None]:
    """Alert function posting ``"<prefix> <message>"`` to the webhook; it always logs at ERROR.

    Without a webhook the alert is logged only, plus a warning saying so; it never raises.
    Delivery failures raise WebhookError; callers wrap alerts in `safe_alert`.
    """

    def alert(message: str) -> None:
        logger.error("ALERT: %s", message)
        if not webhook_url:
            warn_missing_webhook("this alert")
            return
        post_message(webhook_url, f"{prefix} {message}")

    return alert


def safe_alert(alert: Callable[[str], None] | None, message: str) -> bool:
    """Call `alert`, never raising. Returns False if it raised (the failure is logged)."""
    if alert is None:
        return True
    try:
        alert(message)
    except Exception:
        logger.exception("alert failed: %s", message)
        return False
    return True
