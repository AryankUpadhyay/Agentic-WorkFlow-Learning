# Agentic Workflow Engine

A minimal but production-patterned system that takes a **natural-language goal**, breaks it into a **structured execution plan** via an LLM, runs each step against **real connectors**, and returns a **fully-cited trace** of every action.

> **Example Goal:**  
> *"Find all open bugs in our tracker from the last 7 days, summarize the top 3 by severity, and email the summary to the team."*

---

## Architecture

```
Goal (natural language string)
     │
     ▼
┌─────────────┐
│   Planner   │  ← LLM call (Gemini): parse goal → structured step list
└──────┬──────┘
       │  ExecutionPlan (ordered list of Steps)
       ▼
┌──────────────┐
│ Orchestrator │  ← Routes steps to connectors, passes data, handles errors
└──────┬───────┘
       │  Runs each Step against the right Connector
       ▼
┌─────────────────────────────────────────┐
│  Connectors (pluggable via Registry)    │
│  ● GitHub    – fetch issues (READ)      │
│  ● Gemini    – summarize/classify (LLM) │
│  ● Email     – send via SMTP (WRITE)    │
└──────┬──────────────────────────────────┘
       │  StepResult per step
       ▼
┌───────────────┐
│ Trace Builder │  ← Assembles human-readable JSON trace + citations
└───────────────┘
```

**Data flows as a pipeline:** the output of Step N becomes the input of Step N+1, all mediated by the Orchestrator.

---

## Connectors

| Connector | Role          | Operations                        | Backend                                       |
|-----------|---------------|-----------------------------------|-----------------------------------------------|
| GitHub    | Read data     | `list_issues`, `filter_issues`    | `api.github.com/repos/.../issues`             |
| Gemini    | LLM reasoning | `summarize`, `classify`, `extract`| `generativelanguage.googleapis.com` (REST API) |
| Email     | Write/Deliver | `send_email`                      | Any SMTP server (Gmail, Outlook, etc.)         |

### Pluggable Design

All connectors implement the same `BaseConnector` interface:

```python
class BaseConnector(ABC):
    @abstractmethod
    def execute(self, operation, params, context) -> (raw_output, summary, source_ids):
        ...
```

**To swap or add a connector** (e.g., replace Email with Slack):

1. Create a class inheriting `BaseConnector` (see `email_connector.py` as a template)
2. Register it: `registry.register("slack", SlackConnector())`
3. Update the planner prompt in `llm_connector.py` to mention the new connector

No changes needed in the Orchestrator, TraceBuilder, or models.

---

## Project Structure

```
AgenticWorkflowEngine/
├── .env.example              # Template for environment variables
├── .gitignore                # Ignores .env, __pycache__, trace.json
├── README.md                 # This file
├── requirements.txt          # Python dependencies (requests, python-dotenv)
│
└── src/
    ├── __init__.py
    ├── main.py               # CLI entry point — ties everything together
    │
    ├── connectors/           # External service integrations (pluggable)
    │   ├── __init__.py       # ConnectorRegistry — pluggable connector lookup
    │   ├── base.py           # Abstract base class + exception hierarchy
    │   ├── github_connector.py  # GitHub Issues API
    │   ├── llm_connector.py     # Google Gemini (summarize + plan)
    │   └── email_connector.py   # SMTP email delivery
    │
    ├── core/                 # Engine internals
    │   ├── __init__.py
    │   ├── models.py         # Dataclasses: Step, ExecutionPlan, StepResult, Trace
    │   ├── planner.py        # LLM-based goal → execution plan
    │   ├── orchestrator.py   # Step routing, data passing, error handling + retries
    │   └── trace_builder.py  # Assemble + format the final trace
    │
    └── examples/             # Demo utilities
        ├── __init__.py
        └── mock_data.py      # Deterministic mock connectors for demo mode
```

---

## Code Walkthrough

### 1. Models (`src/core/models.py`)
Defines the core data structures that flow through the system:
- **`Step`** — A single unit of work (connector name, operation, params, dependency).
- **`ExecutionPlan`** — An ordered list of Steps, produced by the Planner.
- **`StepResult`** — The output of executing one Step (status, raw output, source IDs, errors).
- **`Trace`** — The complete execution record (all steps, citations, final output).

### 2. Base Connector (`src/connectors/base.py`)
Abstract class that every connector must inherit. Defines:
- The `execute(operation, params, context)` interface.
- Exception hierarchy: `AuthError`, `TransientError`, `EmptyResultError`.

### 3. GitHub Connector (`src/connectors/github_connector.py`)
- **`list_issues`** — Fetches open issues from a GitHub repo via REST API, filtered by state, labels, and date.
- **`filter_issues`** — Pure in-memory sort and filter (top-N by comments, reactions, or recency).
- Maps HTTP errors to the custom exception hierarchy (401→AuthError, 5xx→TransientError).

### 4. LLM Connector (`src/connectors/llm_connector.py`)
Uses the **Google Gemini REST API** (`generativelanguage.googleapis.com`). Serves a **dual role**:
- **Workflow step:** `summarize`, `classify`, `extract` — used by the Orchestrator as inline steps.
- **Planning backend:** `plan()` — parses a natural-language goal into a structured JSON execution plan. Called by the Planner module.

Key implementation details:
- Auth via API key as query parameter (`?key=...`).
- System instructions use Gemini's `systemInstruction` field for strong steering.
- Response parsing extracts text from `candidates[].content.parts[]`.
- A shared `_send_request()` method eliminates duplication between regular calls and planning calls.

The planner prompt describes all available connectors and their operations, so the LLM can generate valid plans.

### 5. Email Connector (`src/connectors/email_connector.py`)
- **`send_email`** — Composes and sends an email via SMTP (supports Gmail, Outlook, any provider).
- Automatically appends citation footers to the email body.
- Uses Python's built-in `smtplib` — no extra dependencies.

### 6. Connector Registry (`src/connectors/__init__.py`)
A simple dict-backed registry that maps names to connector instances:
```python
registry = ConnectorRegistry()
registry.register("github", GitHubConnector())
registry.register("llm", LLMConnector())
registry.register("email", EmailConnector())
```
The Orchestrator uses `registry.as_dict()` to look up connectors by name.

### 7. Planner (`src/core/planner.py`)
Thin layer that:
1. Calls `LLMConnector.plan(goal)` with the natural-language goal.
2. Translates the JSON response into typed `Step` objects.
3. Returns an `ExecutionPlan`.

### 8. Orchestrator (`src/core/orchestrator.py`)
The heart of the engine. For each step in the plan:
1. Checks if the upstream dependency succeeded (skips if not).
2. Looks up the connector by name.
3. Injects upstream output into the step's params.
4. Calls `connector.execute()` with retry logic.
5. Records the `StepResult` (success, failure, or skip).

**Retry policy:**
| Failure Type | Strategy |
|---|---|
| Transient (5xx, timeout, rate limit) | Retry up to 3× with exponential backoff (1s → 2s → 4s) |
| Auth error (401, 403) | Abort immediately — retrying won't help |
| Empty result (no data) | Mark failed, skip downstream dependents |
| Unknown error | Retry once, then abort |

### 9. Trace Builder (`src/core/trace_builder.py`)
After execution completes:
- Collects all `StepResult`s into an ordered structure.
- De-duplicates citations (source IDs) across all steps.
- Builds a plain-English final output string.
- Writes everything to a JSON file and prints a formatted summary.

### 10. Mock Data (`src/examples/mock_data.py`)
Mock connectors that return realistic but deterministic data. Used in demo mode to test the full pipeline without any API keys.

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo-url>
cd AgenticWorkflowEngine
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your real API keys
```

Required variables:

| Variable | Purpose | Required? |
|---|---|---|
| `GITHUB_TOKEN` | GitHub personal access token | Optional (public repos work without it, but rate-limited) |
| `GEMINI_API_KEY` | Google Gemini API key | **Yes** (for planning + summarization) |
| `SMTP_HOST` | SMTP server hostname | For email delivery |
| `SMTP_PORT` | SMTP server port (587 for TLS) | For email delivery |
| `SMTP_USER` | SMTP login username | For email delivery |
| `SMTP_PASSWORD` | SMTP password / app password | For email delivery |
| `EMAIL_FROM` | Sender email address | For email delivery |
| `EMAIL_TO` | Default recipient | For email delivery |

> **Get a Gemini API key (free):** Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 3. Run — Demo mode (no API keys needed)

```bash
python -m src.main --demo
```

This runs the full pipeline with mock connectors and prints a formatted trace.

### 4. Run — Real mode

```bash
python -m src.main --goal "Find all open bugs in cli/cli from last 7 days, summarize the top 3 by comment count, and email the summary to team@example.com"
```

### 5. Run — Skip email delivery

```bash
python -m src.main --goal "Find open bugs in cli/cli from last 7 days, summarize top 3" --no-email
```

### 6. Output

The engine prints a formatted trace to stdout and writes `trace.json` to disk:

```json
{
  "goal": "Find open bugs from last 7 days, summarize top 3, email to team",
  "plan": [
    { "step": 1, "connector": "github", "operation": "list_issues", "depends_on": null },
    { "step": 2, "connector": "github", "operation": "filter_issues", "depends_on": 1 },
    { "step": 3, "connector": "llm",    "operation": "summarize",     "depends_on": 2 },
    { "step": 4, "connector": "email",  "operation": "send_email",    "depends_on": 3 }
  ],
  "steps": [ "..." ],
  "final_output": "Summary emailed successfully. Based on: #5823, #5819, #5811.",
  "citations": ["#5823", "#5819", "#5811"],
  "errors": []
}
```

---

## Adding a New Connector

The system is designed so that adding a new connector requires **zero changes** to the Orchestrator, TraceBuilder, or models. Here's a step-by-step guide:

### Example: Adding a Slack connector

**Step 1:** Create `src/connectors/slack_connector.py`:

```python
from src.connectors.base import BaseConnector, AuthError, TransientError

class SlackConnector(BaseConnector):
    name = "slack"

    def execute(self, operation, params, context):
        if operation == "post_message":
            return self._post_message(params, context)
        raise ValueError(f"Unknown operation: {operation}")

    def _post_message(self, params, context):
        # ... call Slack API ...
        return response_data, "Posted to #channel", source_ids
```

**Step 2:** Register it in `src/main.py`:

```python
from src.connectors.slack_connector import SlackConnector
registry.register("slack", SlackConnector())
```

**Step 3:** Update the planner prompt in `llm_connector.py` to include `slack` as an available connector.

That's it. The Orchestrator will automatically route `"slack"` steps to your new connector.

---

## Design Decisions

**Why no LangGraph / LangChain?**  
The assignment is about understanding the pattern — planner → orchestrator → connectors → trace. LangGraph adds abstraction overhead without teaching the core loop. This implementation is 100% inspectable: every state transition is explicit Python.

**Why Google Gemini for LLM?**  
Used both as the planner (parse goal → plan) and as a connector step (summarize issues). This shows the dual role an LLM plays in agentic systems. Gemini offers a generous free tier and a straightforward REST API — no SDK needed, just `requests`.

**Why citations?**  
Every `StepResult` carries `source_ids` — the issue numbers, message IDs, or resource identifiers that contributed to that step's output. The TraceBuilder propagates these forward so the final output always says "based on: X, Y, Z", letting readers trace the chain of data.

**Why a Connector Registry?**  
Separates connector wiring from orchestration logic. New connectors are added by calling `register()` — the Orchestrator doesn't need to know about specific connector classes.

**Why Email over Slack?**  
Email (SMTP) uses Python's built-in `smtplib` — no extra dependencies, works with any provider. The pluggable design means swapping to Slack, Discord, or Twilio is trivial.
