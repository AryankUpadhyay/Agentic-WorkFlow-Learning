#!/usr/bin/env python3
"""
Agentic Workflow Engine — Entry Point
======================================
This is the main script that ties everything together:
  Goal (string) → Planner (LLM) → Execution Plan (JSON) →
  Orchestrator → Steps run in order → Trace Builder →
  Final output + citations

Usage:

  # Real APIs (requires .env file with API keys):
  python -m src.main --goal "Find open bugs from last 7 days, \\\n      summarize top 3, and email the summary"

  # Specify a repository explicitly:
  python -m src.main --goal "Find open bugs, summarize top 3, email summary" --repo "facebook/react"

  # Demo mode (no API keys needed, uses deterministic mock data):
  python -m src.main --demo

  # Real goal but skip email delivery (useful for testing):
  python -m src.main --goal "..." --no-email

  # Save trace to a custom path:
  python -m src.main --demo --output my_trace.json

  # Verbose logging (debug level):
  python -m src.main --demo --verbose
"""

import argparse
import json
import logging
import os
import sys
import time

# Load environment variables from .env file before importing connectors.
# This must happen early so that connectors pick up API keys via os.getenv().
from dotenv import load_dotenv
load_dotenv()  # Reads .env from the project root (or cwd)

from src.core.models import ExecutionPlan, Step
from src.core.planner import Planner
from src.core.orchestrator import Orchestrator
from src.core.trace_builder import TraceBuilder
from src.connectors import ConnectorRegistry
from src.connectors.github_connector import GitHubConnector
from src.connectors.llm_connector import LLMConnector
from src.connectors.email_connector import EmailConnector


def setup_logging(verbose: bool = False) -> None:
    """
    Configure the root logger.

    Args:
        verbose: If True, log at DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------ #
# Demo mode (mock connectors, no real API calls)                       #
# ------------------------------------------------------------------ #

def run_demo(output_path: str) -> None:
    """
    Run the engine in demo mode with deterministic mock data.

    No API keys needed — useful for testing the orchestration logic,
    verifying trace output, or demonstrating the architecture.
    """
    from src.examples.mock_data import (
        MOCK_CONNECTORS,
        DEMO_PLAN_STEPS,
        DEMO_PLAN_REASONING,
    )

    logger = logging.getLogger(__name__)
    logger.info("=== DEMO MODE (mock connectors, no real API calls) ===")

    # Hard-coded goal for demo mode
    goal = (
        "Find all open bugs in acme/backend from last 7 days, "
        "summarize the top 3 by comment count, and email the summary "
        "to team@example.com"
    )

    # Build the execution plan from the pre-defined demo steps
    plan = ExecutionPlan(
        goal=goal,
        steps=[Step(**s) for s in DEMO_PLAN_STEPS],
        reasoning=DEMO_PLAN_REASONING,
    )

    # Execute the plan using mock connectors
    orchestrator = Orchestrator(connectors=MOCK_CONNECTORS)
    builder      = TraceBuilder()

    started_at = time.time()
    results    = orchestrator.run(plan)
    trace      = builder.build(plan, results, started_at)

    # Print and save the trace
    builder.print_summary(trace)

    trace_json = builder.to_json(trace)
    with open(output_path, "w") as f:
        f.write(trace_json)

    print(f"Full trace written to: {output_path}\n")


# ------------------------------------------------------------------ #
# Real mode (live API calls)                                           #
# ------------------------------------------------------------------ #

def run_real(goal: str, output_path: str, no_email: bool = False, repo: str = None) -> None:
    """
    Run the engine with real API calls to GitHub, Anthropic, and Email.

    Requires environment variables (or .env file):
      - GITHUB_TOKEN       (optional for public repos)
      - ANTHROPIC_API_KEY  (required)
      - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD  (for email)
      - EMAIL_FROM, EMAIL_TO  (for email)

    Args:
        goal:       The natural-language instruction to execute.
        output_path: Where to write the trace JSON file.
        no_email:   If True, skip the email delivery step.
        repo:       Optional repo override (e.g. "owner/repo").
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=== REAL MODE — goal: {goal!r} ===")
    if repo:
        logger.info(f"=== Repo override: {repo} ===")

    # --- Build the connector registry --- #
    # This is where pluggability lives: to swap Email for Slack,
    # just register a different connector under the "email" name
    # (or add "slack" alongside "email").
    registry = ConnectorRegistry()

    github = GitHubConnector()
    llm    = LLMConnector()
    email  = EmailConnector()

    registry.register("github", github)
    registry.register("llm",    llm)
    if not no_email:
        registry.register("email", email)

    logger.info(f"[main] Registered connectors: {registry.list_names()}")

    # --- Plan: ask the LLM to decompose the goal into steps --- #
    planner = Planner(llm)
    plan    = planner.plan(goal, repo=repo)

    if no_email:
        # Drop any email steps from the plan
        plan.steps = [s for s in plan.steps if s.connector != "email"]
        logger.info("[main] --no-email: email steps removed from plan")

    # --- Execute: run each step in order --- #
    orchestrator = Orchestrator(connectors=registry.as_dict())
    builder      = TraceBuilder()

    started_at = time.time()
    results    = orchestrator.run(plan)
    trace      = builder.build(plan, results, started_at)

    # --- Output: print summary and save trace JSON --- #
    builder.print_summary(trace)

    trace_json = builder.to_json(trace)
    with open(output_path, "w") as f:
        f.write(trace_json)

    print(f"Full trace written to: {output_path}\n")


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

def main() -> None:
    """Parse CLI arguments and dispatch to demo or real mode."""
    parser = argparse.ArgumentParser(
        description="Agentic Workflow Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--goal", type=str,
        help="Natural-language goal to execute"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with mock data (no API keys needed)"
    )
    parser.add_argument(
        "--repo", type=str,
        help="GitHub repo to target (e.g. 'owner/repo'). "
             "If omitted, the LLM extracts it from the goal or defaults to 'cli/cli'."
    )
    parser.add_argument(
        "--no-email", action="store_true",
        help="Skip email delivery step"
    )
    parser.add_argument(
        "--output", type=str, default="trace.json",
        help="Path to write trace JSON (default: trace.json)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.demo:
        run_demo(args.output)
    elif args.goal:
        run_real(args.goal, args.output, no_email=args.no_email, repo=args.repo)
    else:
        parser.print_help()
        print("\nError: provide --goal or --demo\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
