"""
Core Data Models
================
Every entity that flows through the system — plans, steps, results, traces —
is defined here so the rest of the codebase has a single source of truth.

Hierarchy:
    Step           → a single unit of work (produced by Planner)
    ExecutionPlan  → ordered list of Steps
    StepError      → one error event during step execution
    StepResult     → the output of executing one Step (produced by Orchestrator)
    Trace          → the complete execution record (produced by TraceBuilder)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time


class StepStatus(str, Enum):
    """Possible states a step can be in during or after execution."""
    PENDING   = "pending"    # Not yet started
    RUNNING   = "running"    # Currently executing
    SUCCESS   = "success"    # Completed successfully
    FAILED    = "failed"     # Failed (after retries if applicable)
    SKIPPED   = "skipped"    # Skipped because an upstream dependency failed
    RETRYING  = "retrying"   # Failed, but retrying (transient error)


@dataclass
class Step:
    """
    A single unit of work in the execution plan.

    Produced by the Planner, consumed by the Orchestrator.
    Each step specifies which connector to call, what operation to perform,
    and which previous step's output it depends on.
    """
    step_id: int                        # 1-indexed position in plan
    connector: str                      # "github" | "llm" | "email"
    operation: str                      # e.g. "list_issues", "summarize", "send_email"
    params: Dict[str, Any]             # Static params from the planner
    depends_on: Optional[int] = None   # step_id this step needs output from
    description: str = ""              # Human-readable intent for this step


@dataclass
class ExecutionPlan:
    """
    Ordered list of Steps produced by the Planner from a natural-language goal.

    The `reasoning` field stores the LLM's chain-of-thought explanation of
    why it chose this sequence of steps.
    """
    goal: str
    steps: List[Step]
    reasoning: str = ""   # Planner's chain-of-thought (included in the trace)


@dataclass
class StepError:
    """
    One error event on a step.

    There may be multiple StepErrors per step if the Orchestrator retried.
    """
    attempt: int              # Which attempt this error occurred on (0-indexed)
    error_type: str           # "auth", "transient", "empty_result", "unknown"
    message: str              # Human-readable error description
    timestamp: float = field(default_factory=time.time)


@dataclass
class StepResult:
    """
    The output of executing one Step.

    Produced by the Orchestrator after running a connector.  These are
    collected by the TraceBuilder to assemble the final trace.
    """
    step_id: int
    connector: str
    operation: str
    params: Dict[str, Any]
    status: StepStatus

    # What came back from the connector (issues list, summary text, etc.)
    raw_output: Any = None

    # Human-readable one-liner for the trace (e.g. "Fetched 12 issues")
    result_summary: str = ""

    # IDs of source items that contributed (issue numbers, message IDs, etc.)
    # These propagate forward through the chain to form the final citations.
    source_ids: List[str] = field(default_factory=list)

    # Errors / retries logged during this step
    errors: List[StepError] = field(default_factory=list)

    # The step_id this result depends on (set by the Orchestrator for tracing)
    depends_on: Optional[int] = None

    # Wall-clock timing
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def duration_ms(self) -> Optional[float]:
        """Calculate execution duration in milliseconds."""
        if self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at) * 1000, 1)

    def to_trace_dict(self) -> Dict[str, Any]:
        """Serialise this result to a dict for the trace JSON output."""
        return {
            "step": self.step_id,
            "connector": self.connector,
            "operation": self.operation,
            "params": self.params,
            "status": self.status.value,
            "result_summary": self.result_summary,
            "source_ids": self.source_ids,
            "duration_ms": self.duration_ms(),
            "errors": [
                {
                    "attempt": e.attempt,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.errors
            ],
        }


@dataclass
class Trace:
    """
    The complete record of a workflow execution.

    Human-readable JSON produced by the TraceBuilder.  Includes every step
    executed, all errors, the final output, and citations pointing back
    to the source data used at each step.
    """
    goal: str
    plan_reasoning: str
    steps: List[StepResult]
    final_output: str
    citations: List[str]          # De-duplicated source IDs across all steps
    errors: List[Dict[str, Any]]  # Summary of all errors

    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the entire trace to a dict (for JSON output).

        The 'plan' section shows each step's connector, operation,
        and dependency chain — using the depends_on field stored
        directly on each StepResult.
        """
        return {
            "goal": self.goal,
            "plan_reasoning": self.plan_reasoning,
            # Plan overview: shows the step graph with dependencies
            "plan": [
                {
                    "step": r.step_id,
                    "connector": r.connector,
                    "operation": r.operation,
                    "depends_on": r.depends_on,
                }
                for r in self.steps
            ],
            # Detailed per-step results
            "steps": [r.to_trace_dict() for r in self.steps],
            "final_output": self.final_output,
            "citations": self.citations,
            "errors": self.errors,
            "duration_ms": (
                round((self.finished_at - self.started_at) * 1000, 1)
                if self.finished_at else None
            ),
        }
