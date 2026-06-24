"""
LLM Connector — Provider-Agnostic
===================================
Wraps any LLM provider for natural-language reasoning steps:
summarization, classification, extraction, and planning.

Switching providers is a single line change:

    # Use Gemini (default)
    llm = LLMConnector(provider="gemini")

    # Switch to OpenAI
    llm = LLMConnector(provider="openai")

    # Switch to Anthropic
    llm = LLMConnector(provider="anthropic")

Or set it via environment variable — no code change at all:

    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...

Operations (workflow steps)
---------------------------
summarize  – condense a list of issues into a delivery-ready summary
classify   – assign severity/category labels to a list of items
extract    – pull structured fields out of unstructured text

Environment variables
---------------------
LLM_PROVIDER      – "gemini" | "openai" | "anthropic"  (default: "gemini")

GEMINI_API_KEY    – required when provider=gemini
OPENAI_API_KEY    – required when provider=openai
ANTHROPIC_API_KEY – required when provider=anthropic
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

from src.connectors.base import (
    AuthError,
    BaseConnector,
    EmptyResultError,
    TransientError,
)


# ======================================================================== #
#  Provider adapters — one class per LLM provider                           #
#                                                                            #
#  Each adapter translates the universal (system, user, max_tokens)         #
#  interface into the provider's specific HTTP request format and parses     #
#  the response back into a plain string.                                    #
#                                                                            #
#  To add a new provider: subclass _LLMAdapter, implement the three         #
#  abstract methods, and register it in _PROVIDER_REGISTRY at the bottom.   #
# ======================================================================== #

class _LLMAdapter(ABC):
    """
    Abstract base for all provider adapters.

    The LLMConnector calls only these three methods — it never touches
    provider-specific URLs, payload shapes, or auth headers directly.
    """

    # Retry configuration (inherited by all adapters unless overridden)
    MAX_RETRIES  = 3
    BACKOFF_BASE = 2.0   # seconds: 2 → 4 → 8

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (used in log messages and errors)."""

    @abstractmethod
    def _build_payload(
        self,
        system: Optional[str],
        user: str,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Build the (url, headers, json_body) needed for one API call.

        Returns:
            url      – the full endpoint URL
            headers  – HTTP headers dict (auth, content-type, …)
            payload  – the JSON request body
        """

    @abstractmethod
    def _parse_response(self, data: Dict[str, Any]) -> str:
        """
        Extract the assistant's text from the provider's JSON response.

        Raises EmptyResultError if the response contains no text.
        """

    @abstractmethod
    def _check_auth_error(self, status_code: int, body: str) -> None:
        """
        Raise AuthError for provider-specific auth failure codes/messages.
        Called before the generic status-code checks.
        """

    # ------------------------------------------------------------------ #
    # Shared HTTP transport with retry + model fallback                    #
    # ------------------------------------------------------------------ #

    def call(
        self,
        system: Optional[str],
        user: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a message to the provider and return the response text.

        Handles retries and (where applicable) model fallback automatically.
        """
        url, headers, payload = self._build_payload(system, user, max_tokens)

        last_error: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return self._do_request(url, headers, payload)

            except AuthError:
                raise   # never retry auth errors

            except TransientError as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES - 1:
                    backoff = self.BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "[LLM/%s] %s — retrying in %.0fs (attempt %d/%d)",
                        self.name, exc, backoff, attempt + 1, self.MAX_RETRIES,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "[LLM/%s] All %d retries exhausted.",
                        self.name, self.MAX_RETRIES,
                    )

        raise TransientError(
            f"{self.name}: all {self.MAX_RETRIES} retries failed. "
            f"Last error: {last_error}"
        )

    def _do_request(
        self, url: str, headers: Dict[str, Any], payload: Dict[str, Any]
    ) -> str:
        """Execute a single HTTP POST with no retry logic."""
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            raise TransientError(f"{self.name}: request timed out")
        except requests.exceptions.ConnectionError as exc:
            raise TransientError(f"{self.name}: connection error — {exc}")

        # Let the adapter raise provider-specific auth errors first
        self._check_auth_error(resp.status_code, resp.text)

        if resp.status_code in (401, 403):
            raise AuthError(f"{self.name}: auth failed ({resp.status_code})")
        if resp.status_code == 429:
            raise TransientError(f"{self.name}: rate limited (429)")
        if resp.status_code >= 500:
            raise TransientError(f"{self.name}: server error {resp.status_code}")
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:200]
            raise TransientError(f"{self.name}: error {resp.status_code} — {detail}")

        return self._parse_response(resp.json())


# ======================================================================== #
#  Concrete adapters                                                         #
# ======================================================================== #

class _GeminiAdapter(_LLMAdapter):
    """Google Gemini via the generateContent REST API."""

    API_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"
    MODEL_CHAIN = ["gemini-2.5-flash", "gemini-2.0-flash"]

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "gemini"

    def _build_payload(
        self, system: Optional[str], user: str, max_tokens: int
    ) -> Tuple[str, Dict, Dict]:
        url = f"{self.API_BASE}/{self.MODEL_CHAIN[0]}:generateContent"

        headers = {"Content-Type": "application/json"}
        # Gemini authenticates via query param, not Authorization header
        url = f"{url}?key={self.api_key}"

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        return url, headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            raise EmptyResultError("Gemini: no candidates in response")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p]
        if not text_parts:
            raise EmptyResultError("Gemini: no text in response parts")

        return "\n".join(text_parts)

    def _check_auth_error(self, status_code: int, body: str) -> None:
        if status_code in (401, 403):
            raise AuthError(
                "Gemini auth failed. Check GEMINI_API_KEY. "
                "Get a key at: https://aistudio.google.com/apikey"
            )

    # Gemini has a fallback model chain — override call() to try it
    def call(
        self, system: Optional[str], user: str, max_tokens: int = 1024
    ) -> str:
        last_error: Optional[Exception] = None

        for model_idx, model in enumerate(self.MODEL_CHAIN):
            url, headers, payload = self._build_payload(system, user, max_tokens)

            # Swap the model name in the URL for fallback attempts
            if model_idx > 0:
                url = url.replace(self.MODEL_CHAIN[0], model)
                logger.warning("[LLM/gemini] Falling back to model: %s", model)

            for attempt in range(self.MAX_RETRIES):
                try:
                    return self._do_request(url, headers, payload)
                except AuthError:
                    raise
                except TransientError as exc:
                    last_error = exc
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = self.BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "[LLM/gemini] %s — retry %d/%d in %.0fs (model=%s)",
                            exc, attempt + 1, self.MAX_RETRIES, backoff, model,
                        )
                        time.sleep(backoff)

        raise TransientError(
            f"Gemini: all models and retries exhausted. "
            f"Models tried: {', '.join(self.MODEL_CHAIN)}. "
            f"Last error: {last_error}"
        )


class _OpenAIAdapter(_LLMAdapter):
    """OpenAI via the /v1/chat/completions endpoint."""

    API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)

    @property
    def name(self) -> str:
        return "openai"

    def _build_payload(
        self, system: Optional[str], user: str, max_tokens: int
    ) -> Tuple[str, Dict, Dict]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        return self.API_URL, headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            raise EmptyResultError("OpenAI: no choices in response")

        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise EmptyResultError("OpenAI: empty content in response")

        return text

    def _check_auth_error(self, status_code: int, body: str) -> None:
        if status_code in (401, 403):
            raise AuthError(
                "OpenAI auth failed. Check OPENAI_API_KEY. "
                "Get a key at: https://platform.openai.com/api-keys"
            )


class _AnthropicAdapter(_LLMAdapter):
    """Anthropic Claude via the /v1/messages endpoint."""

    API_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

    @property
    def name(self) -> str:
        return "anthropic"

    def _build_payload(
        self, system: Optional[str], user: str, max_tokens: int
    ) -> Tuple[str, Dict, Dict]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system

        return self.API_URL, headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> str:
        content = data.get("content", [])
        text_blocks = [b["text"] for b in content if b.get("type") == "text"]
        if not text_blocks:
            raise EmptyResultError("Anthropic: no text blocks in response")
        return "\n".join(text_blocks)

    def _check_auth_error(self, status_code: int, body: str) -> None:
        if status_code in (401, 403):
            raise AuthError(
                "Anthropic auth failed. Check ANTHROPIC_API_KEY. "
                "Get a key at: https://console.anthropic.com"
            )


# ======================================================================== #
#  Provider registry — the only place you touch when adding a new provider  #
# ======================================================================== #

def _build_adapter(provider: str) -> _LLMAdapter:
    """
    Instantiate the correct adapter for the given provider name.

    Adding a new provider:
      1. Write a subclass of _LLMAdapter above.
      2. Add one entry to _REGISTRY below — that's it.
    """
    _REGISTRY: Dict[str, Any] = {
        "gemini": lambda: _GeminiAdapter(
            api_key=_require_env("GEMINI_API_KEY", "gemini")
        ),
        "openai": lambda: _OpenAIAdapter(
            api_key=_require_env("OPENAI_API_KEY", "openai"),
        ),
        "anthropic": lambda: _AnthropicAdapter(
            api_key=_require_env("ANTHROPIC_API_KEY", "anthropic"),
        ),
    }

    factory = _REGISTRY.get(provider.lower())
    if factory is None:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: {supported}"
        )

    return factory()


def _require_env(var: str, provider: str) -> str:
    value = os.getenv(var, "")
    if not value:
        raise AuthError(
            f"{var} is not set (required for provider='{provider}'). "
            f"Export it before running the engine."
        )
    return value


# ======================================================================== #
#  LLMConnector — the public class used by the rest of the codebase         #
# ======================================================================== #

class LLMConnector(BaseConnector):
    """
    Provider-agnostic LLM connector.

    The connector itself contains no provider-specific logic — it delegates
    every API call to the adapter selected at construction time.

    Usage
    -----
        # Pick provider explicitly:
        llm = LLMConnector(provider="openai")
        llm = LLMConnector(provider="anthropic")
        llm = LLMConnector(provider="gemini")   # default

        # Or via environment variable (no code change needed):
        # export LLM_PROVIDER=anthropic
        llm = LLMConnector()
    """

    name = "llm"

    def __init__(self, provider: Optional[str] = None):
        """
        Args:
            provider: "gemini" | "openai" | "anthropic".
                      Falls back to the LLM_PROVIDER env var, then "gemini".
        """
        resolved = provider or os.getenv("LLM_PROVIDER", "gemini")
        self._adapter: _LLMAdapter = _build_adapter(resolved)
        logger.info("[LLM] Using provider: %s", self._adapter.name)

    # ------------------------------------------------------------------ #
    # Public interface (BaseConnector)                                     #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[int, Any],
    ) -> Tuple[Any, str, List[str]]:
        """Route to the correct LLM operation handler."""
        if operation == "summarize":
            return self._summarize(params, context)
        if operation == "classify":
            return self._classify(params, context)
        if operation == "extract":
            return self._extract(params, context)
        raise ValueError(f"LLMConnector: unknown operation '{operation}'")

    # ------------------------------------------------------------------ #
    # Operations                                                           #
    # ------------------------------------------------------------------ #

    def _summarize(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[str, str, List[str]]:
        """
        Summarize a list of GitHub issues into a delivery-ready message.

        Params:
            top_n       – how many issues are expected (informational)
            format      – "email" | "slack" | "plain"  (default: "email")
            depends_on  – step_id whose output is the issues list
        """
        issues = self._get_upstream_issues(params, context)
        fmt    = params.get("format", "email")

        if not issues:
            raise EmptyResultError("summarize: no issues to summarize")

        source_ids = [f"#{i['number']}" for i in issues]

        issue_blobs = []
        for i in issues:
            issue_blobs.append(
                f"Issue #{i['number']}: {i['title']}\n"
                f"  URL: {i['html_url']}\n"
                f"  Comments: {i.get('comments', 0)}\n"
                f"  Labels: {', '.join(l['name'] for l in i.get('labels', []))}\n"
                f"  Created: {i.get('created_at', 'unknown')}\n"
                f"  Body preview: {(i.get('body') or '')[:200]}"
            )

        link_rule = (
            "CRITICAL FORMATTING RULE: Every bug ID MUST be a clickable link. "
            "Use the issue URL provided. Format each bug ID as: "
            "#<number> (<url>). "
            "For example: #13022 (https://github.com/cli/cli/issues/13022). "
            "Never show a bare bug number without its link."
        )
        style_rule = (
            "WRITING STYLE: Write in simple, plain English. "
            "Keep each issue summary to 1-2 short sentences maximum. "
            "Avoid jargon, markdown formatting, and overly technical language."
        )

        format_instructions: Dict[str, str] = {
            "email": (
                "Format as a brief, professional email body.\n"
                "Start with 'Subject: ' on the first line.\n"
                "Add a one-line greeting, then list each issue as a bullet.\n"
                "Each bullet: the issue title as a clickable link, then a "
                f"1-sentence plain-English description.\n{link_rule}\n{style_rule}"
            ),
            "slack": (
                "Format as a short Slack message.\n"
                "Start with a one-line intro, then list each issue as a bullet.\n"
                f"Each bullet: the issue title with its link.\n{link_rule}\n{style_rule}"
            ),
            "plain": f"Plain text, concise.\n{link_rule}\n{style_rule}",
        }
        instructions = format_instructions.get(fmt, f"Plain text, concise.\n{link_rule}\n{style_rule}")

        based_on_parts = [f"#{i['number']} ({i['html_url']})" for i in issues]
        based_on_line  = "Based on: " + ", ".join(based_on_parts)

        prompt = (
            f"You are an engineering team assistant. "
            f"Summarize the following {len(issues)} GitHub issue(s) for the team. "
            f"Be brief and to the point — no long paragraphs.\n\n"
            f"{instructions}\n\n"
            + "\n\n".join(issue_blobs)
            + f"\n\nAt the end, include a line: '{based_on_line}'"
        )

        summary_text = self._adapter.call(system=None, user=prompt)
        result_summary = (
            f"Generated {fmt} summary ({len(summary_text)} chars) "
            f"from {len(issues)} issue(s)"
        )
        return summary_text, result_summary, source_ids

    def _classify(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[List[Dict], str, List[str]]:
        """
        Classify items with severity labels (e.g. P0/P1/P2) using an LLM.

        Params:
            labels     – list of possible label values (default: ["P0","P1","P2"])
            depends_on – step_id whose output is the issues list
        """
        issues = self._get_upstream_issues(params, context)
        labels = params.get("labels", ["P0", "P1", "P2"])

        source_ids = [f"#{i['number']}" for i in issues]

        items_json = json.dumps(
            [{"number": i["number"], "title": i["title"],
              "body": (i.get("body") or "")[:300]} for i in issues],
            indent=2,
        )

        prompt = (
            f"Classify each of the following GitHub issues with one of these "
            f"severity labels: {', '.join(labels)}.\n"
            f"Respond ONLY with a JSON array, no markdown, no explanation. "
            f'Each element: {{"number": <int>, "severity": "<label>"}}.\n\n'
            f"{items_json}"
        )

        raw     = self._adapter.call(system=None, user=prompt)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        classifications = json.loads(cleaned)

        sev_map = {c["number"]: c["severity"] for c in classifications}
        for issue in issues:
            issue["severity"] = sev_map.get(issue["number"], "unknown")

        result_summary = f"Classified {len(issues)} issues: " + ", ".join(
            f"#{i['number']}={i['severity']}" for i in issues
        )
        return issues, result_summary, source_ids

    def _extract(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> Tuple[Any, str, List[str]]:
        """
        Generic extraction: pull structured fields from text using an LLM.

        Params:
            prompt – the extraction prompt (required)
        """
        prompt = params.get("prompt", "")
        if not prompt:
            raise ValueError("LLMConnector.extract: 'prompt' param is required")

        result = self._adapter.call(system=None, user=prompt)
        return result, f"Extracted {len(result)} chars", []

    # ------------------------------------------------------------------ #
    # Planning (called by the Planner module, not the Orchestrator)        #
    # ------------------------------------------------------------------ #

    def plan(self, goal: str, repo: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a natural-language goal into a structured execution plan.

        Called by the Planner module. The system prompt describes the
        available connectors so the LLM generates a valid step sequence.

        Args:
            goal: Natural-language goal string.
            repo: Optional repo override ("owner/repo"). If provided,
                  the LLM is told to use it instead of extracting from goal.

        Returns:
            dict with keys: reasoning, steps
            Each step: { step_id, connector, operation, params, depends_on, description }
        """
        repo_rule = (
            f"- ALWAYS use '{repo}' as the repo for list_issues.\n"
            if repo
            else (
                "- Extract repo from the goal if mentioned, else use "
                "'cli/cli' as a demo fallback.\n"
            )
        )

        system = (
            "You are an orchestration planner for an agentic workflow engine. "
            "You have access to three connectors:\n"
            "  1. github  – operations: list_issues(repo, state, labels, days), "
            "filter_issues(top_n, sort_by)\n"
            "  2. llm     – operations: summarize(format), classify(labels), "
            "extract(prompt)\n"
            "  3. email   – operations: send_email(to, subject)\n\n"
            "Given a natural-language goal, produce a JSON execution plan. "
            "Respond ONLY with a valid JSON object — no markdown, no explanation.\n\n"
            "Schema:\n"
            "{\n"
            '  "reasoning": "<one paragraph explaining your plan>",\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_id": 1,\n'
            '      "connector": "github",\n'
            '      "operation": "list_issues",\n'
            '      "params": { "repo": "owner/repo", "state": "open", '
            '"labels": ["bug"], "days": 7 },\n'
            '      "depends_on": null,\n'
            '      "description": "Fetch open bug issues from last 7 days"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- depends_on must be null for step 1, or the step_id of the previous step.\n"
            "- filter_issues always depends_on the list_issues step.\n"
            "- summarize always depends_on the filter_issues step (or list_issues if no filter).\n"
            "- For summarize, set format to 'email' unless the goal specifies otherwise.\n"
            "- send_email always depends_on the summarize step.\n"
            "- For send_email, set 'to' to '__default__' ALWAYS. Do NOT extract an email from the goal.\n"
            "- For send_email, set 'body' to '__from_previous_step__'.\n"
            "- For filter_issues, leave sort_by as 'comments' unless the goal specifies otherwise.\n"
            + repo_rule
        )

        raw     = self._adapter.call(system=system, user=goal, max_tokens=4096)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Planner returned invalid JSON: {exc}\nRaw:\n{raw[:500]}"
            )

    # ------------------------------------------------------------------ #
    # Context helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_upstream_issues(
        self, params: Dict[str, Any], context: Dict[int, Any]
    ) -> List[Dict]:
        """Find the issues list from the upstream step that produced it."""
        dep = params.get("depends_on")
        if dep is not None and dep in context:
            data = context[dep]
            if isinstance(data, list):
                return data

        # Fallback: scan context for any issues-shaped list
        for v in context.values():
            if (isinstance(v, list) and v
                    and isinstance(v[0], dict) and "number" in v[0]):
                return v

        return []