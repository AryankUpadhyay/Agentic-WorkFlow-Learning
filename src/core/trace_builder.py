"""
Trace Builder
=============
Assembles the final human-readable trace from ExecutionPlan + StepResults.

After the Orchestrator finishes running all steps, the TraceBuilder:
  1. Collects all StepResults into an ordered, serialisable structure
  2. De-duplicates citations (source IDs) across all steps
  3. Writes a plain-English final_output string
  4. Surfaces all errors in a top-level errors list
  5. Computes total wall-clock duration

Output: a Trace dataclass that can be serialised to JSON or printed
to the terminal.

The trace is the "receipt" of the entire workflow — a human reader
can look at it and verify exactly what happened, in what order,
with what data, and which source items contributed to the final result.
"""

import json
import logging
import time
from typing import Any, Dict, List

from src.core.models import ExecutionPlan, Trace, StepResult, StepStatus

logger = logging.getLogger(__name__)


class TraceBuilder:
    """Builds, serialises, and prints the execution trace."""

    def build(
        self,
        plan: ExecutionPlan,
        results: List[StepResult],
        started_at: float,
    ) -> Trace:
        """
        Construct a Trace from a completed ExecutionPlan + its StepResults.

        Args:
            plan:       The execution plan that was run.
            results:    The list of StepResults from the Orchestrator.
            started_at: Wall-clock time when execution began (time.time()).

        Returns:
            A fully-populated Trace dataclass.
        """
        citations = self._collect_citations(results)
        errors    = self._collect_errors(results)
        final_out = self._build_final_output(plan.goal, results, citations)

        trace = Trace(
            goal           = plan.goal,
            plan_reasoning = plan.reasoning,
            steps          = results,
            final_output   = final_out,
            citations      = citations,
            errors         = errors,
            started_at     = started_at,
            finished_at    = time.time(),
        )

        logger.info(
            f"[TraceBuilder] Trace complete — "
            f"{len(results)} steps, "
            f"{len(citations)} citations, "
            f"{len(errors)} errors, "
            f"duration={round((trace.finished_at - trace.started_at)*1000, 1)}ms"
        )
        return trace

    def to_json(self, trace: Trace, indent: int = 2) -> str:
        """Serialise trace to a pretty-printed JSON string."""
        return json.dumps(trace.to_dict(), indent=indent, default=str)

    def print_summary(self, trace: Trace) -> None:
        """
        Print a human-readable summary of the trace to stdout.

        Uses Unicode icons (✓, ✗, ⊘, ↻) to indicate step status at a glance.
        """
        d = trace.to_dict()
        total_ms = d.get("duration_ms") or 0

        # --- Header --- #
        print("\n" + "="*60)
        print("  WORKFLOW EXECUTION TRACE")
        print("="*60)
        print(f"  Goal:     {d['goal']}")
        print(f"  Duration: {total_ms:.0f} ms")
        print(f"  Steps:    {len(d['steps'])}")
        print(f"  Errors:   {len(d['errors'])}")
        print()

        # --- Plan reasoning --- #
        print("  PLAN (LLM reasoning)")
        print("  " + "-"*56)
        for line in d["plan_reasoning"].split("\n"):
            print(f"  {line}")
        print()

        # --- Per-step details --- #
        print("  STEPS")
        print("  " + "-"*56)
        for s in d["steps"]:
            # Map status to a visual icon
            status_icon = {
                "success": "✓",
                "failed":  "✗",
                "skipped": "⊘",
                "retrying":"↻",
            }.get(s["status"], "?")

            print(
                f"  [{status_icon}] Step {s['step']}: "
                f"[{s['connector']}] {s['operation']}"
            )
            print(f"      → {s['result_summary']}")
            if s["source_ids"]:
                print(f"      Sources: {', '.join(s['source_ids'])}")
            if s["errors"]:
                for e in s["errors"]:
                    print(
                        f"      ⚠ attempt {e['attempt']}: "
                        f"({e['error_type']}) {e['message']}"
                    )
            print()

        # --- Final output --- #
        print("  FINAL OUTPUT")
        print("  " + "-"*56)
        print(f"  {d['final_output']}")
        print()

        # --- Citations --- #
        if d["citations"]:
            print("  CITATIONS")
            print("  " + "-"*56)
            print(f"  {', '.join(d['citations'])}")
            print()

        # --- Errors summary --- #
        if d["errors"]:
            print("  ERRORS")
            print("  " + "-"*56)
            for e in d["errors"]:
                print(
                    f"  Step {e['step']}: ({e['error_type']}) {e['message']}"
                )
            print()

        print("="*60 + "\n")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _collect_citations(self, results: List[StepResult]) -> List[str]:
        """
        Collect a de-duplicated list of source IDs across all steps.

        Preserves the order in which IDs first appeared, so citations
        are listed chronologically (matching the execution order).
        """
        seen = set()
        ids: List[str] = []
        for r in results:
            for sid in r.source_ids:
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids

    def _collect_errors(self, results: List[StepResult]) -> List[Dict[str, Any]]:
        """Flatten all per-step errors into a single top-level list."""
        errors = []
        for r in results:
            for e in r.errors:
                errors.append({
                    "step":       r.step_id,
                    "connector":  r.connector,
                    "operation":  r.operation,
                    "attempt":    e.attempt,
                    "error_type": e.error_type,
                    "message":    e.message,
                })
        return errors

    def _build_final_output(
        self,
        goal: str,
        results: List[StepResult],
        citations: List[str],
    ) -> str:
        """
        Construct a one-paragraph final output string.

        Tries to pull the actual message text from the last successful step,
        falls back to a status summary if no steps succeeded.
        """
        successful = [r for r in results if r.status == StepStatus.SUCCESS]
        failed     = [r for r in results if r.status == StepStatus.FAILED]
        skipped    = [r for r in results if r.status == StepStatus.SKIPPED]

        # If nothing succeeded, report failure
        if not successful:
            return (
                f"Workflow failed: no steps completed successfully. "
                f"Check the errors list for details. "
                f"Goal was: {goal}"
            )

        last = successful[-1]

        # Build a descriptive final output based on the last successful step
        if last.connector == "email" and last.status == StepStatus.SUCCESS:
            base = f"Summary emailed successfully. {last.result_summary}."
        elif last.connector == "slack" and last.status == StepStatus.SUCCESS:
            base = f"Summary posted to Slack. {last.result_summary}."
        elif last.connector == "llm":
            base = (
                f"LLM summary generated (not yet delivered — "
                f"delivery step may have failed)."
            )
        else:
            base = (
                f"Last completed step: [{last.connector}] {last.operation} "
                f"— {last.result_summary}."
            )

        # Append failure/skip notes if any
        if failed:
            base += f" ({len(failed)} step(s) failed; see errors.)"
        if skipped:
            base += f" ({len(skipped)} step(s) skipped.)"

        # Append citations
        if citations:
            base += f" Based on: {', '.join(citations)}."

        return base
