"""
Mock Data & Mock Connectors
============================
Deterministic mock responses for demo mode (``python -m src.main --demo``).

No real API calls are made.  The mock connectors return realistic data
that mirrors what the real GitHub, LLM, and Email connectors would produce,
so the full orchestration pipeline — planning, execution, tracing — can be
tested end-to-end without any API keys.

Usage:
    from src.examples.mock_data import MOCK_CONNECTORS, DEMO_PLAN_STEPS
"""

import time
from typing import Any, Dict, List, Tuple

from src.connectors.base import BaseConnector

# ------------------------------------------------------------------ #
# Realistic mock GitHub issues                                         #
# ------------------------------------------------------------------ #

MOCK_ISSUES = [
    {
        "number": 5823,
        "title": "Race condition in Redis session cache causes intermittent 401s",
        "html_url": "https://github.com/acme/backend/issues/5823",
        "state": "open",
        "comments": 14,
        "labels": [{"name": "bug"}, {"name": "priority:high"}],
        "created_at": "2025-06-18T09:15:00Z",
        "updated_at": "2025-06-23T17:42:00Z",
        "body": (
            "Under high concurrency, the session cache write and the token "
            "validation read can interleave, causing users to receive 401 "
            "errors even with a valid session. Reproducible at >500 RPS. "
            "Affects `/api/v2/me` endpoint. Needs distributed lock."
        ),
    },
    {
        "number": 5819,
        "title": "Kafka consumer lag spikes every 6h causing delayed notifications",
        "html_url": "https://github.com/acme/backend/issues/5819",
        "state": "open",
        "comments": 9,
        "labels": [{"name": "bug"}, {"name": "kafka"}],
        "created_at": "2025-06-17T14:30:00Z",
        "updated_at": "2025-06-22T11:00:00Z",
        "body": (
            "The notification-worker consumer group falls behind by ~10k messages "
            "every 6 hours. Metrics show GC pause of ~800ms correlated with lag. "
            "Workaround: restart the consumer. Root cause likely off-heap buffer leak."
        ),
    },
    {
        "number": 5811,
        "title": "Search index out of sync after document delete — stale results returned",
        "html_url": "https://github.com/acme/backend/issues/5811",
        "state": "open",
        "comments": 7,
        "labels": [{"name": "bug"}, {"name": "search"}],
        "created_at": "2025-06-16T08:00:00Z",
        "updated_at": "2025-06-21T09:15:00Z",
        "body": (
            "After a document is deleted via the REST API, the Solr index is not "
            "updated within the soft-delete TTL window (30s). Users see deleted "
            "documents in search results for up to 60s. The delete-event Kafka topic "
            "is missing a subscriber on the search-indexer service."
        ),
    },
    {
        "number": 5808,
        "title": "Pagination cursor breaks when items deleted mid-scroll",
        "html_url": "https://github.com/acme/backend/issues/5808",
        "state": "open",
        "comments": 4,
        "labels": [{"name": "bug"}],
        "created_at": "2025-06-16T06:00:00Z",
        "updated_at": "2025-06-20T12:00:00Z",
        "body": (
            "Cursor-based pagination returns duplicate or missing items "
            "when rows are deleted between page fetches."
        ),
    },
]

# Pre-built summary that the mock LLM connector returns
MOCK_SUMMARY = """\
Subject: Weekly Bug Summary — Top 3 by Activity

Hi team,

Here's a summary of the top 3 open bugs from this week, ranked by comment activity:

• #5823 — Race condition in Redis session cache (14 comments)
  Intermittent 401s under high concurrency (>500 RPS). Needs distributed lock.
  https://github.com/acme/backend/issues/5823

• #5819 — Kafka consumer lag spikes every 6h (9 comments)
  Delayed notifications; GC pause of ~800ms correlated. Workaround: restart consumer.
  https://github.com/acme/backend/issues/5819

• #5811 — Solr index out of sync after delete (7 comments)
  Stale search results for up to 60s. Missing delete-event subscriber on search-indexer.
  https://github.com/acme/backend/issues/5811

Based on: #5823, #5819, #5811"""

# Pre-built email response that the mock Email connector returns
MOCK_EMAIL_RESPONSE = {
    "ok": True,
    "to": "team@example.com",
    "subject": "Weekly Bug Summary — Top 3 by Activity",
    "from": "workflow-bot@example.com",
    "smtp_host": "smtp.example.com",
}


# ------------------------------------------------------------------ #
# Mock connectors                                                      #
# ------------------------------------------------------------------ #

class MockGitHubConnector(BaseConnector):
    """Mock GitHub connector — returns MOCK_ISSUES without any API call."""
    name = "github"

    def execute(
        self, operation: str, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[Any, str, List[str]]:
        # Simulate a small network delay
        time.sleep(0.05)

        if operation == "list_issues":
            source_ids = [f"#{i['number']}" for i in MOCK_ISSUES]
            return (
                MOCK_ISSUES,
                f"Fetched {len(MOCK_ISSUES)} open bug issues from "
                f"acme/backend (last 7 days)",
                source_ids,
            )

        if operation == "filter_issues":
            top_n = params.get("top_n", 3)
            # Sort by comment count descending (same logic as real connector)
            sorted_issues = sorted(
                MOCK_ISSUES, key=lambda x: x.get("comments", 0), reverse=True
            )
            top = sorted_issues[:top_n]
            source_ids = [f"#{i['number']}" for i in top]
            return (
                top,
                f"Filtered to top {len(top)} by comments: "
                f"{', '.join(source_ids)}",
                source_ids,
            )

        raise ValueError(f"MockGitHub: unknown operation '{operation}'")


class MockLLMConnector(BaseConnector):
    """Mock LLM connector — returns a pre-built summary string."""
    name = "llm"

    def execute(
        self, operation: str, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[Any, str, List[str]]:
        time.sleep(0.08)  # simulate LLM latency

        if operation == "summarize":
            # Get upstream issues to count them and collect source IDs
            issues = []
            dep = params.get("depends_on")
            if dep in context:
                issues = context[dep]

            source_ids = [f"#{i['number']}" for i in issues]
            return (
                MOCK_SUMMARY,
                f"Generated email summary ({len(MOCK_SUMMARY)} chars) "
                f"from {len(issues)} issues",
                source_ids,
            )

        raise ValueError(f"MockLLM: unknown operation '{operation}'")


class MockEmailConnector(BaseConnector):
    """Mock Email connector — pretends to send an email."""
    name = "email"

    def execute(
        self, operation: str, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[Any, str, List[str]]:
        time.sleep(0.03)  # simulate SMTP latency

        if operation == "send_email":
            to = params.get("to", "team@example.com")

            # Collect source IDs from the entire context chain
            source_ids = []
            seen = set()
            for v in context.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "number" in item:
                            sid = f"#{item['number']}"
                            if sid not in seen:
                                seen.add(sid)
                                source_ids.append(sid)

            return (
                MOCK_EMAIL_RESPONSE,
                f"Email sent to {to} "
                f"(subject: Weekly Bug Summary — Top 3 by Activity)",
                source_ids,
            )

        raise ValueError(f"MockEmail: unknown operation '{operation}'")


# Registry of mock connectors — used by main.py in demo mode
MOCK_CONNECTORS = {
    "github": MockGitHubConnector(),
    "llm":    MockLLMConnector(),
    "email":  MockEmailConnector(),
}


# ------------------------------------------------------------------ #
# Hard-coded execution plan for demo mode                              #
# ------------------------------------------------------------------ #
# This mirrors what the LLM planner would produce for the demo goal.

DEMO_PLAN_STEPS = [
    {
        "step_id": 1,
        "connector": "github",
        "operation": "list_issues",
        "params": {
            "repo": "acme/backend",
            "state": "open",
            "labels": ["bug"],
            "days": 7,
        },
        "depends_on": None,
        "description": "Fetch all open bug issues from acme/backend updated in last 7 days",
    },
    {
        "step_id": 2,
        "connector": "github",
        "operation": "filter_issues",
        "params": {
            "top_n": 3,
            "sort_by": "comments",
        },
        "depends_on": 1,
        "description": "Rank issues by comment count, keep top 3",
    },
    {
        "step_id": 3,
        "connector": "llm",
        "operation": "summarize",
        "params": {
            "format": "email",
        },
        "depends_on": 2,
        "description": "Summarize the 3 issues into an email-ready message",
    },
    {
        "step_id": 4,
        "connector": "email",
        "operation": "send_email",
        "params": {
            "to": "team@example.com",
            "subject": "Weekly Bug Summary — Top 3 by Activity",
            "body": "__from_previous_step__",
        },
        "depends_on": 3,
        "description": "Email the summary to the team",
    },
]

DEMO_PLAN_REASONING = (
    "The goal requires reading recent bugs from a repository, identifying the "
    "most-discussed ones, distilling them into a readable summary, and delivering "
    "it via email. I'll chain four steps: (1) GitHub list_issues to fetch raw data, "
    "(2) GitHub filter_issues to select the top 3 by comment activity (a proxy for "
    "severity/impact in active repos), (3) LLM summarize to produce an email-formatted "
    "message the team can skim quickly, (4) Email send_email to deliver it. "
    "Each step's output feeds directly into the next."
)
