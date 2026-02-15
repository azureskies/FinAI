"""Alert manager for sending notifications via email and webhooks.

Supports multiple channels: email (SMTP) and webhook (Slack/Discord/LINE).
Configuration is loaded from configs/monitoring.yaml.
"""

from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from enum import IntEnum
from typing import Optional

import yaml
from loguru import logger

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

_CONFIG_PATH = "configs/monitoring.yaml"


class AlertLevel(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


class AlertManager:
    """Send alerts through configured channels."""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config()
        self.channels = self.config.get("alert_channels", {})

    @staticmethod
    def _load_config() -> dict:
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Monitoring config not found at {}", _CONFIG_PATH)
            return {}

    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        channel: Optional[str] = None,
    ) -> bool:
        """Send an alert through the specified channel.

        Args:
            level: Alert severity level.
            title: Alert title / subject.
            message: Alert body text.
            channel: Channel name (email, webhook). If None, sends to all.

        Returns:
            True if at least one channel succeeded.
        """
        level_name = level.name
        logger.info("Alert [{}] {}: {}", level_name, title, message)

        if channel:
            return self._send_to_channel(channel, level_name, title, message)

        # Send to all configured channels
        success = False
        for ch_name in self.channels:
            if self._send_to_channel(ch_name, level_name, title, message):
                success = True
        return success

    def _send_to_channel(
        self, channel: str, level: str, title: str, message: str
    ) -> bool:
        """Dispatch to the appropriate channel sender."""
        ch_config = self.channels.get(channel, {})
        if not ch_config or not ch_config.get("enabled", False):
            logger.debug("Channel {} not enabled, skipping", channel)
            return False

        ch_type = ch_config.get("type", channel)

        if ch_type == "email":
            return self._send_email(ch_config, level, title, message)
        elif ch_type == "webhook":
            return self._send_webhook(ch_config, level, title, message)
        else:
            logger.warning("Unknown channel type: {}", ch_type)
            return False

    def _send_email(
        self, config: dict, level: str, title: str, message: str
    ) -> bool:
        """Send alert via SMTP email."""
        try:
            smtp_host = config.get("smtp_host", "smtp.gmail.com")
            smtp_port = config.get("smtp_port", 587)
            username = config.get("username", "")
            password = config.get("password", "")
            from_addr = config.get("from_addr", username)
            to_addrs = config.get("to_addrs", [])

            if not to_addrs:
                logger.warning("No email recipients configured")
                return False

            subject = f"[FinAI {level}] {title}"
            msg = MIMEText(message, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = ", ".join(to_addrs)

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if username and password:
                    server.login(username, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())

            logger.info("Email alert sent to {}", to_addrs)
            return True
        except Exception as exc:
            logger.error("Failed to send email alert: {}", exc)
            return False

    def _send_webhook(
        self, config: dict, level: str, title: str, message: str
    ) -> bool:
        """Send alert via webhook (Slack/Discord/LINE compatible)."""
        if requests is None:
            logger.error("requests library not installed, cannot send webhook")
            return False

        try:
            url = config.get("url", "")
            if not url:
                logger.warning("No webhook URL configured")
                return False

            payload = {
                "text": f"*[{level}] {title}*\n{message}",
                "username": "FinAI Monitor",
            }
            headers = {"Content-Type": "application/json"}

            resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
            resp.raise_for_status()
            logger.info("Webhook alert sent to {}", url[:50])
            return True
        except Exception as exc:
            logger.error("Failed to send webhook alert: {}", exc)
            return False
