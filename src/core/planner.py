"""
Planner
=======
Converts a natural-language goal into a structured ExecutionPlan by calling
an LLM (Claude) with a carefully crafted system prompt.

The planner is intentionally kept thin — it only:
  1. Calls the LLM's ``plan()`` method with the goal string
  2. Translates the JSON response into typed Step objects

All error handling and execution logic live in the Orchestrator.

Flow:
    "Find open bugs..."  →  Planner.plan()  →  ExecutionPlan(steps=[...])
"""

import logging
from typing import Any, Dict

from src.core.models import ExecutionPlan, Step
from src.connectors.llm_connector import LLMConnector

logger = logging.getLogger(__name__)


class Planner:
    """
    Parses natural-language goals into structured execution plans.

    Uses the LLMConnector's ``plan()`` method under the hood.
    """

    def __init__(self, llm: LLMConnector):
        """
        Args:
            llm: An initialised LLMConnector instance (with valid API key).
        """
        self.llm = llm

    def plan(self, goal: str, repo: str = None) -> ExecutionPlan:
        """
        Parse a natural-language goal into an ordered ExecutionPlan.

        Args:
            goal: The user's natural-language instruction, e.g.
                  "Find open bugs from last 7 days, summarize top 3, email them"
            repo: Optional repo override (e.g. "owner/repo"). If provided,
                  the LLM planner is forced to use this repo for list_issues.

        Returns:
            An ExecutionPlan containing typed Step objects.

        Raises:
            ValueError: If the LLM response is malformed or empty.
            ConnectorError: If the LLM API call fails (auth, timeout, etc.).
        """
        logger.info(f"[Planner] Parsing goal: {goal!r}")
        if repo:
            logger.info(f"[Planner] Repo override: {repo}")

        # Ask the LLM to generate a structured plan as JSON
        raw_plan: Dict[str, Any] = self.llm.plan(goal, repo=repo)

        reasoning = raw_plan.get("reasoning", "")
        raw_steps = raw_plan.get("steps", [])

        if not raw_steps:
            raise ValueError("Planner: LLM returned an empty step list")

        # Convert raw dicts into typed Step dataclass instances
        steps = []
        for s in raw_steps:
            steps.append(
                Step(
                    step_id    = s["step_id"],
                    connector  = s["connector"],
                    operation  = s["operation"],
                    params     = s.get("params", {}),
                    depends_on = s.get("depends_on"),
                    description= s.get("description", ""),
                )
            )

        # Log the generated plan for debugging
        logger.info(f"[Planner] Plan: {len(steps)} steps")
        for st in steps:
            logger.info(
                f"  Step {st.step_id}: [{st.connector}] {st.operation} "
                f"(depends_on={st.depends_on}) — {st.description}"
            )

        return ExecutionPlan(goal=goal, steps=steps, reasoning=reasoning)
