"""
GitHub Connector
================
Talks to the GitHub REST API to fetch and filter repository issues.

Operations
----------
list_issues   – GET /repos/{owner}/{repo}/issues with state/label/date filters
filter_issues – pure in-memory: sort fetched issues, return top-N

Real API docs: https://docs.github.com/en/rest/issues/issues

Environment variables
---------------------
GITHUB_TOKEN  – Personal access token (optional for public repos, but
                unauthenticated requests are rate-limited to 60/hour).
"""

import os
import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.connectors.base import (
    BaseConnector,
    AuthError,
    TransientError,
    EmptyResultError,
)


class GitHubConnector(BaseConnector):
    """Connector for reading issue data from GitHub repositories."""

    name = "github"

    # GitHub REST API v3 base URL
    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: GitHub personal access token. Falls back to the
                   GITHUB_TOKEN environment variable.
        """
        self.token = token or os.getenv("GITHUB_TOKEN", "")

        # Reusable session with default headers for all GitHub requests
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    # ------------------------------------------------------------------ #
    # Public interface (BaseConnector)                                     #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[int, Any],
    ) -> Tuple[Any, str, List[str]]:
        """Route to the correct internal operation handler."""
        if operation == "list_issues":
            return self._list_issues(params, context)
        if operation == "filter_issues":
            return self._filter_issues(params, context)
        raise ValueError(f"GitHubConnector: unknown operation '{operation}'")

    # ------------------------------------------------------------------ #
    # Operations                                                           #
    # ------------------------------------------------------------------ #

    def _list_issues(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[List[Dict], str, List[str]]:
        """
        Fetch open issues from a GitHub repo, optionally filtered by label and date.

        Params (from execution plan):
            repo     – "owner/repo"  (required)
            state    – "open" | "closed" | "all"  (default: "open")
            labels   – list of label strings, e.g. ["bug"]
            since    – ISO date string "YYYY-MM-DD"; only issues updated after this
            days     – alternative to `since`: fetch issues from last N days
            per_page – max results per page (default 50)
        """
        repo     = params.get("repo") or params.get("repository")
        state    = params.get("state", "open")
        labels   = params.get("labels", [])
        since    = params.get("since") or self._days_ago(params.get("days", 7))
        per_page = params.get("per_page", 50)

        if not repo:
            raise ValueError("GitHubConnector.list_issues: 'repo' param is required")

        # Build the GitHub API request URL and query params
        url = f"{self.BASE_URL}/repos/{repo}/issues"
        query: Dict[str, Any] = {
            "state": state,
            "per_page": per_page,
            # Ensure ISO 8601 format with time component
            "since": since + "T00:00:00Z" if "T" not in since else since,
        }
        if labels:
            query["labels"] = ",".join(labels)

        resp = self._get(url, params=query)
        issues = resp.json()

        # GitHub's /issues endpoint also returns PRs — filter those out
        issues = [i for i in issues if "pull_request" not in i]

        if not issues:
            raise EmptyResultError(
                f"No issues found in {repo} with state={state}, "
                f"labels={labels}, since={since}"
            )

        # Collect source IDs for citation tracking
        source_ids = [f"#{i['number']}" for i in issues]
        summary = (
            f"Fetched {len(issues)} issue(s) from {repo} "
            f"(state={state}, since={since})"
        )
        return issues, summary, source_ids

    def _filter_issues(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[List[Dict], str, List[str]]:
        """
        Filter and rank issues from a previous step's output.

        This is a pure in-memory operation — no API call. It sorts the
        upstream issue list and returns the top-N.

        Params:
            top_n      – how many to keep (default 3)
            sort_by    – "comments" | "reactions" | "updated" (default "comments")
            depends_on – step_id whose output is the issues list
        """
        top_n   = params.get("top_n", 3)
        sort_by = params.get("sort_by", "comments")

        # Pull the issues list from whichever upstream step produced it
        issues = self._get_upstream_issues(params, context)

        # Sort descending by the chosen metric
        if sort_by == "comments":
            issues.sort(key=lambda x: x.get("comments", 0), reverse=True)
        elif sort_by == "reactions":
            issues.sort(
                key=lambda x: x.get("reactions", {}).get("total_count", 0),
                reverse=True,
            )
        else:
            # Default fallback: sort by most recently updated
            issues.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        top = issues[:top_n]
        source_ids = [f"#{i['number']}" for i in top]
        summary = (
            f"Filtered to top {len(top)} issues by {sort_by}: "
            + ", ".join(source_ids)
        )
        return top, summary, source_ids

    # ------------------------------------------------------------------ #
    # HTTP helper                                                          #
    # ------------------------------------------------------------------ #

    def _get(self, url: str, params: Dict = None) -> requests.Response:
        """
        Make a GET request to the GitHub API with error classification.

        Maps HTTP status codes to our custom exception hierarchy so the
        Orchestrator knows whether to retry, abort, or skip.
        """
        try:
            resp = self.session.get(url, params=params, timeout=10)
        except requests.exceptions.Timeout:
            raise TransientError(f"GitHub request timed out: {url}")
        except requests.exceptions.ConnectionError as e:
            raise TransientError(f"GitHub connection error: {e}")

        # 401/403 → auth problem, don't retry
        if resp.status_code in (401, 403):
            raise AuthError(
                f"GitHub auth failed ({resp.status_code}): "
                "check GITHUB_TOKEN env var. "
                "A public repo without a token works too, but rate-limits apply."
            )
        # 422 → bad request params (our fault, not transient)
        if resp.status_code == 422:
            raise ValueError(f"GitHub validation error: {resp.json()}")
        # 5xx → server error, safe to retry
        if resp.status_code >= 500:
            raise TransientError(f"GitHub server error {resp.status_code}")
        # Other non-2xx → treat as transient
        if not resp.ok:
            raise TransientError(
                f"GitHub error {resp.status_code}: {resp.text[:200]}"
            )

        return resp

    # ------------------------------------------------------------------ #
    # Context helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_upstream_issues(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> List[Dict]:
        """
        Find the issues list from whichever upstream step produced it.

        First checks the explicit depends_on reference, then falls back
        to scanning the context for any list of GitHub-issue-shaped dicts.
        """
        dep = params.get("depends_on")
        if dep is not None and dep in context:
            data = context[dep]
            if isinstance(data, list):
                return list(data)  # shallow copy to avoid mutating upstream

        # Fallback: scan context for any list that looks like GitHub issues
        for v in context.values():
            if (isinstance(v, list) and v
                    and isinstance(v[0], dict) and "number" in v[0]):
                return list(v)

        raise EmptyResultError(
            "filter_issues: no upstream issues list found in context"
        )

    @staticmethod
    def _days_ago(days: int = 7) -> str:
        """Return an ISO date string for N days before today (UTC)."""
        d = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        return d.strftime("%Y-%m-%d")
