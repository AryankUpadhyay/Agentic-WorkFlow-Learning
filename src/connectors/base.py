"""
Abstract Base Connector
=======================
Every connector in the system (GitHub, LLM, Email, or any future one) must
inherit from ``BaseConnector`` and implement the ``execute`` method.

The contract is simple:
    Input:  (operation, params, context)
    Output: (raw_output, result_summary, source_ids)

``context`` is a dict of {step_id: raw_output} from all previously-completed
steps, so any connector can reference upstream data.  For example, the LLM
connector reads the issues list fetched by GitHub in an earlier step.

Custom exception hierarchy
--------------------------
  ConnectorError          ← base for all connector failures
  ├── AuthError           ← bad credentials — never retry
  ├── TransientError      ← timeout / rate-limit / 5xx — safe to retry
  └── EmptyResultError    ← API returned 200 but no useful data
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Exception hierarchy                                                          #
# --------------------------------------------------------------------------- #

class ConnectorError(Exception):
    """Base for all connector-level failures."""
    pass


class AuthError(ConnectorError):
    """Bad credentials — abort immediately, retrying won't help."""
    pass


class TransientError(ConnectorError):
    """Network blip, rate limit, 5xx — safe to retry with backoff."""
    pass


class EmptyResultError(ConnectorError):
    """API succeeded but returned nothing useful (e.g. zero issues found)."""
    pass


# --------------------------------------------------------------------------- #
# Abstract base class                                                          #
# --------------------------------------------------------------------------- #

class BaseConnector(ABC):
    """
    All connectors follow this interface.

    Subclasses must set ``name`` (e.g. "github") and implement ``execute``.

    Returns a 3-tuple:
        raw_output     – the actual data (list of dicts, string, etc.)
        result_summary – a one-liner for the trace log
        source_ids     – identifiers of source items (issue IDs, msg IDs, …)
    """

    # Human-readable name — used as the lookup key in ConnectorRegistry.
    name: str = "base"

    @abstractmethod
    def execute(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[int, Any],   # {step_id: raw_output} of all prior steps
    ) -> Tuple[Any, str, List[str]]:
        """
        Run one operation against the external service.

        Args:
            operation: The specific action to perform (e.g. "list_issues").
            params:    Parameters for this operation (from the execution plan).
            context:   Outputs of all previously-completed steps, keyed by step_id.

        Returns:
            (raw_output, result_summary, source_ids)

        Raises:
            ConnectorError (or subclass) on failure.
            Never returns None — raise EmptyResultError instead.
        """
        ...
