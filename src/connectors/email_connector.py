"""
Email Connector (SMTP)
======================
Sends emails via any SMTP server (Gmail, Outlook, custom, etc.).

This is the "write / deliver" connector in the workflow: it takes the
summary text produced by the LLM step and delivers it as an email.

Operations
----------
send_email  – compose and send an email with the given subject and body

Environment variables
---------------------
SMTP_HOST      – SMTP server hostname    (default: smtp.gmail.com)
SMTP_PORT      – SMTP server port        (default: 587 for STARTTLS)
SMTP_USER      – login username / email
SMTP_PASSWORD  – login password or app-specific password
EMAIL_FROM     – sender address          (defaults to SMTP_USER)
EMAIL_TO       – default recipient(s)    (comma-separated)

Pluggability note
-----------------
This connector follows the same BaseConnector interface as GitHub and LLM.
To swap Email for Slack (or Twilio, Discord, etc.), simply:
  1. Create a new connector class inheriting BaseConnector
  2. Register it in main.py:  registry.register("slack", SlackConnector())
  3. Update the planner prompt in llm_connector.py to mention "slack"
No changes needed in the Orchestrator or TraceBuilder.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional, Tuple

from src.connectors.base import (
    BaseConnector,
    AuthError,
    TransientError,
    EmptyResultError,
)


class EmailConnector(BaseConnector):
    """Connector for sending emails via SMTP."""

    name = "email"

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
    ):
        """
        All parameters fall back to environment variables if not provided.

        Args:
            smtp_host:     SMTP server hostname.
            smtp_port:     SMTP server port (587 for TLS, 465 for SSL).
            smtp_user:     SMTP login username.
            smtp_password: SMTP login password / app password.
            email_from:    Sender email address.
            email_to:      Default recipient(s), comma-separated.
        """
        self.smtp_host     = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port     = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user     = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.email_from    = email_from or os.getenv("EMAIL_FROM", "") or self.smtp_user
        self.email_to      = email_to or os.getenv("EMAIL_TO", "")

    # ------------------------------------------------------------------ #
    # Public interface (BaseConnector)                                     #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[int, Any],
    ) -> Tuple[Any, str, List[str]]:
        """Route to the correct email operation handler."""
        if operation == "send_email":
            return self._send_email(params, context)
        raise ValueError(f"EmailConnector: unknown operation '{operation}'")

    # ------------------------------------------------------------------ #
    # Operations                                                           #
    # ------------------------------------------------------------------ #

    def _send_email(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[Dict, str, List[str]]:
        """
        Compose and send an email.

        Params (from execution plan):
            to      – recipient email address (falls back to EMAIL_TO env var)
            subject – email subject line (default: "Workflow Summary")
            body    – email body text; if "__from_previous_step__" or empty,
                      pulls the text from the most recent upstream step

        The method automatically appends a citation footer listing all
        source IDs (issue numbers, etc.) found across the execution context.
        """
        raw_to  = params.get("to", "__default__")
        subject = params.get("subject", "Workflow Summary")
        body    = params.get("body", "__from_previous_step__")

        # Resolve the recipient:
        # - "__default__" or empty → use EMAIL_TO env var
        # - Any real address → use it directly
        if not raw_to or raw_to == "__default__":
            to = self.email_to
        else:
            to = raw_to

        # Resolve body from upstream LLM summary if it's a placeholder
        if body == "__from_previous_step__" or not body:
            depends_on = params.get("depends_on")
            body = self._get_upstream_text(context, depends_on)

        if not body:
            raise EmptyResultError("EmailConnector: no body text to send")

        if not to:
            raise EmptyResultError(
                "EmailConnector: no recipient. Set EMAIL_TO env var "
                "or pass 'to' in params."
            )

        # Collect source IDs scoped to the upstream summarize step
        depends_on = params.get("depends_on")
        source_ids = self._collect_all_source_ids(context, depends_on)

        # Append citation footer if not already present in the body
        if source_ids and "Based on:" not in body:
            body = body.rstrip() + f"\n\nBased on: {', '.join(source_ids)}"

        # Build the MIME message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self.email_from
        msg["To"]      = to

        # Attach plain-text body
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Send via SMTP
        response_data = self._smtp_send(msg, to)

        result_summary = f"Email sent to {to} (subject: {subject})"
        return response_data, result_summary, source_ids

    # ------------------------------------------------------------------ #
    # SMTP transport                                                       #
    # ------------------------------------------------------------------ #

    def _smtp_send(self, msg: MIMEMultipart, to: str) -> Dict[str, Any]:
        """
        Connect to the SMTP server and send the email.

        Uses STARTTLS on the configured port. Maps common SMTP errors
        to our ConnectorError hierarchy.
        """
        if not self.smtp_user or not self.smtp_password:
            raise AuthError(
                "SMTP_USER and SMTP_PASSWORD must be set. "
                "For Gmail, use an App Password "
                "(https://myaccount.google.com/apppasswords)."
            )

        try:
            # Connect and start TLS
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
            server.ehlo()
            server.starttls()
            server.ehlo()

            # Authenticate
            server.login(self.smtp_user, self.smtp_password)

            # Send the email
            server.sendmail(self.email_from, to.split(","), msg.as_string())
            server.quit()

            return {
                "ok": True,
                "to": to,
                "subject": msg["Subject"],
                "from": self.email_from,
                "smtp_host": self.smtp_host,
            }

        except smtplib.SMTPAuthenticationError as e:
            raise AuthError(
                f"SMTP authentication failed: {e}. "
                "Check SMTP_USER and SMTP_PASSWORD. "
                "For Gmail, you may need an App Password."
            )
        except smtplib.SMTPRecipientsRefused as e:
            raise TransientError(f"SMTP recipients refused: {e}")
        except smtplib.SMTPException as e:
            raise TransientError(f"SMTP error: {e}")
        except TimeoutError:
            raise TransientError(
                f"SMTP connection timed out ({self.smtp_host}:{self.smtp_port})"
            )
        except ConnectionRefusedError:
            raise TransientError(
                f"SMTP connection refused ({self.smtp_host}:{self.smtp_port})"
            )
        except OSError as e:
            raise TransientError(f"SMTP network error: {e}")

    # ------------------------------------------------------------------ #
    # Context helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_upstream_text(self, context: Dict[int, Any], depends_on: Optional[int] = None) -> str:
        """
        Find the upstream LLM summary text from context.

        Prefers the step pointed to by depends_on (the LLM summarize step).
        Falls back to scanning in reverse order for the most recent string
        longer than 20 chars, to skip short status messages.
        """
        # Prefer the explicit upstream step set by _inject_upstream
        if depends_on is not None and depends_on in context:
            val = context[depends_on]
            if isinstance(val, str) and len(val) > 20:
                return val

        # Fallback: scan in reverse (original behaviour)
        for step_id in sorted(context.keys(), reverse=True):
            val = context[step_id]
            if isinstance(val, str) and len(val) > 20:
                return val
        return ""

    def _collect_all_source_ids(self, context: Dict[int, Any], depends_on: Optional[int] = None) -> List[str]:
        """
        Collect issue/message identifiers for the citation footer.

        When depends_on is provided (the LLM summarize step), we walk back
        through its dependency chain to collect only the source IDs that
        actually contributed to the summary — not every issue fetched.

        Falls back to scanning the full context if depends_on is absent.
        """
        seen: set = set()
        ids: List[str] = []

        # Prefer scoped context: only the steps that fed the summary
        candidate_step_ids = (
            [depends_on] if depends_on is not None and depends_on in context
            else list(context.keys())
        )

        for step_id in candidate_step_ids:
            val = context[step_id]
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "number" in item:
                        sid = f"#{item['number']}"
                        if sid not in seen:
                            seen.add(sid)
                            ids.append(sid)
        return ids