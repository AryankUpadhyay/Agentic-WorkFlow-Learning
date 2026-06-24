"""
Orchestrator
============
The heart of the engine.  Given an ExecutionPlan, it:

  1. Resolves each Step to the correct Connector (via the registry)
  2. Injects upstream step output into params / context
  3. Handles failures with a clear retry policy
  4. Collects StepResults that feed the TraceBuilder

Retry policy
------------
  - Transient errors (5xx, timeout, rate limit):
        Up to MAX_RETRIES with exponential backoff (1s → 2s → 4s → …)
  - Auth errors (bad credentials):
        Abort immediately — retrying won't help
  - EmptyResult (API returned nothing useful):
        Skip remaining steps that depend on this one, log warning
  - Unknown errors:
        Retry once, then abort

Data flow
---------
  context = {}                     # step_id → raw_output
  for step in plan.steps:
      result = execute(step)
      context[step.step_id] = result.raw_output
  # Each step can read upstream outputs from context via depends_on
"""

import logging
import time
from typing import Any, Dict, List

from src.core.models import ExecutionPlan, Step, StepResult, StepStatus, StepError
from src.connectors.base import (
    BaseConnector,
    AuthError,
    TransientError,
    EmptyResultError,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES   = 3      # Maximum retry attempts for transient errors
BACKOFF_BASE  = 1.0    # Base delay in seconds (doubles each retry: 1s, 2s, 4s)


class Orchestrator:
    """
    Executes an ordered list of Steps, routing each to the right connector,
    passing data between steps, and handling failures gracefully.
    """

    def __init__(self, connectors: Dict[str, BaseConnector]):
        """
        Args:
            connectors: Mapping of name → connector instance, e.g.:
                {
                    "github": GitHubConnector(),
                    "llm":    LLMConnector(),
                    "email":  EmailConnector(),
                }
                Can also be obtained from ConnectorRegistry.as_dict().
        """
        self.connectors = connectors

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(self, plan: ExecutionPlan) -> List[StepResult]:
        """
        Execute all steps in the plan, in order.

        Returns a list of StepResults (one per step).
        This method never raises — errors are captured inside StepResult.errors.

        Args:
            plan: The ExecutionPlan produced by the Planner.

        Returns:
            Ordered list of StepResult objects (success, failed, or skipped).
        """
        results: List[StepResult] = []

        # Context accumulates step outputs: {step_id: raw_output}
        # Downstream steps can read upstream data from here.
        context: Dict[int, Any] = {}

        # Track which steps failed so we can skip their dependents
        failed_step_ids = set()

        for step in plan.steps:
            result = self._execute_step(step, context, failed_step_ids)
            results.append(result)

            if result.status == StepStatus.SUCCESS:
                # Store this step's output so downstream steps can access it
                context[step.step_id] = result.raw_output
            else:
                # Mark as failed so any step depending on this one is skipped
                failed_step_ids.add(step.step_id)

        return results

    # ------------------------------------------------------------------ #
    # Single-step execution with retry logic                               #
    # ------------------------------------------------------------------ #

    def _execute_step(
        self,
        step: Step,
        context: Dict[int, Any],
        failed_step_ids: set,
    ) -> StepResult:
        """
        Execute a single step with full error handling and retry logic.

        Returns a StepResult regardless of success/failure — errors are
        logged inside the result rather than raised.
        """
        # Create the result object (initially PENDING)
        result = StepResult(
            step_id    = step.step_id,
            connector  = step.connector,
            operation  = step.operation,
            params     = step.params,
            status     = StepStatus.PENDING,
            depends_on = step.depends_on,  # Propagate for trace output
        )

        # --- Check: should this step be skipped? --- #
        if step.depends_on is not None and step.depends_on in failed_step_ids:
            result.status = StepStatus.SKIPPED
            result.result_summary = (
                f"Skipped: upstream step {step.depends_on} did not succeed"
            )
            logger.warning(
                f"[Orchestrator] Step {step.step_id} SKIPPED "
                f"(depends_on={step.depends_on} failed)"
            )
            return result

        # --- Check: is the connector registered? --- #
        connector = self.connectors.get(step.connector)
        if connector is None:
            result.status = StepStatus.FAILED
            result.result_summary = (
                f"No connector registered for '{step.connector}'"
            )
            result.errors.append(StepError(
                attempt=0,
                error_type="config",
                message=result.result_summary,
            ))
            logger.error(
                f"[Orchestrator] Step {step.step_id} FAILED: "
                f"{result.result_summary}"
            )
            return result

        # --- Prepare: enrich params with upstream dependency info --- #
        enriched_params = self._inject_upstream(step, context)

        logger.info(
            f"[Orchestrator] Step {step.step_id} START "
            f"[{step.connector}].{step.operation}"
        )
        result.started_at = time.time()
        result.status = StepStatus.RUNNING

        # --- Execute with retry loop --- #
        attempt = 0
        while attempt <= MAX_RETRIES:
            try:
                # Call the connector's execute method
                raw, summary, source_ids = connector.execute(
                    operation=step.operation,
                    params=enriched_params,
                    context=context,
                )
                # Success — populate result and return
                result.raw_output     = raw
                result.result_summary = summary
                result.source_ids     = source_ids
                result.status         = StepStatus.SUCCESS
                result.finished_at    = time.time()
                logger.info(
                    f"[Orchestrator] Step {step.step_id} SUCCESS "
                    f"in {result.duration_ms():.0f}ms — {summary}"
                )
                return result

            except AuthError as e:
                # Auth errors: abort immediately, retrying won't fix this
                result.errors.append(StepError(
                    attempt=attempt, error_type="auth", message=str(e)
                ))
                result.status = StepStatus.FAILED
                result.result_summary = f"Auth error (aborting): {e}"
                result.finished_at = time.time()
                logger.error(
                    f"[Orchestrator] Step {step.step_id} AUTH ERROR: {e}"
                )
                return result

            except EmptyResultError as e:
                # Empty results: don't retry, just mark failed and move on
                result.errors.append(StepError(
                    attempt=attempt, error_type="empty_result", message=str(e)
                ))
                result.status = StepStatus.FAILED
                result.result_summary = f"Empty result: {e}"
                result.finished_at = time.time()
                logger.warning(
                    f"[Orchestrator] Step {step.step_id} EMPTY RESULT: {e}"
                )
                return result

            except TransientError as e:
                # Transient errors: retry with exponential backoff
                result.errors.append(StepError(
                    attempt=attempt, error_type="transient", message=str(e)
                ))
                if attempt < MAX_RETRIES:
                    backoff = BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"[Orchestrator] Step {step.step_id} TRANSIENT ERROR "
                        f"(attempt {attempt+1}/{MAX_RETRIES}), "
                        f"retrying in {backoff:.1f}s: {e}"
                    )
                    result.status = StepStatus.RETRYING
                    time.sleep(backoff)
                    attempt += 1
                else:
                    # Exhausted all retries
                    result.status = StepStatus.FAILED
                    result.result_summary = (
                        f"Failed after {MAX_RETRIES} retries: {e}"
                    )
                    result.finished_at = time.time()
                    logger.error(
                        f"[Orchestrator] Step {step.step_id} FAILED "
                        f"after {MAX_RETRIES} retries: {e}"
                    )
                    return result

            except Exception as e:
                # Unknown/unexpected errors: retry once, then abort
                result.errors.append(StepError(
                    attempt=attempt, error_type="unknown", message=str(e)
                ))
                if attempt < 1:
                    backoff = BACKOFF_BASE
                    logger.warning(
                        f"[Orchestrator] Step {step.step_id} UNKNOWN ERROR "
                        f"(attempt {attempt+1}), retrying once in {backoff}s: {e}"
                    )
                    time.sleep(backoff)
                    attempt += 1
                else:
                    result.status = StepStatus.FAILED
                    result.result_summary = f"Unknown error: {e}"
                    result.finished_at = time.time()
                    logger.error(
                        f"[Orchestrator] Step {step.step_id} FAILED "
                        f"(unknown): {e}"
                    )
                    return result

        # Safety net — should never reach here due to returns in the loop
        result.status = StepStatus.FAILED
        result.finished_at = time.time()
        return result

    # ------------------------------------------------------------------ #
    # Param enrichment                                                     #
    # ------------------------------------------------------------------ #

    def _inject_upstream(
        self, step: Step, context: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Enrich step params with upstream context reference.

        Adds 'depends_on' key so connectors know which step's output to use.
        This is how data flows from one step to the next (e.g., the LLM
        connector reads the issues list that GitHub fetched in step 1).
        """
        params = dict(step.params)  # Shallow copy to avoid mutating the plan
        if step.depends_on is not None:
            params["depends_on"] = step.depends_on
        return params
