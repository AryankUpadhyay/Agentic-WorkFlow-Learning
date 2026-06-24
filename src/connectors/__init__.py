"""
Connector Registry
==================
Central registry that maps connector names (e.g. "github", "llm", "email")
to their instances.  The Orchestrator uses this registry to look up which
connector should handle each step — so adding a new connector is as simple as:

    1. Create a class that inherits BaseConnector
    2. Call  registry.register("my_connector", MyConnector())

No changes needed in the Orchestrator, Planner, or main module.

Design note:
    We intentionally use a simple dict-backed registry rather than
    metaclass magic or decorators.  Explicit registration is easier
    to debug and easier for newcomers to trace through.
"""

from typing import Dict, Optional

from src.connectors.base import BaseConnector


class ConnectorRegistry:
    """
    A pluggable registry of connector instances.

    Usage:
        registry = ConnectorRegistry()
        registry.register("github", GitHubConnector())
        registry.register("llm", LLMConnector())
        registry.register("email", EmailConnector())

        # Orchestrator retrieves connectors by name:
        connector = registry.get("github")
    """

    def __init__(self) -> None:
        # Internal store: connector_name → connector_instance
        self._connectors: Dict[str, BaseConnector] = {}

    def register(self, name: str, connector: BaseConnector) -> None:
        """
        Register a connector instance under the given name.

        Args:
            name:      Lookup key used in execution plans (e.g. "github").
            connector: An instance of a BaseConnector subclass.

        Raises:
            TypeError: If connector is not a BaseConnector subclass.
        """
        if not isinstance(connector, BaseConnector):
            raise TypeError(
                f"Expected a BaseConnector subclass, got {type(connector).__name__}"
            )
        self._connectors[name] = connector

    def get(self, name: str) -> Optional[BaseConnector]:
        """
        Look up a connector by name.

        Returns:
            The connector instance, or None if not registered.
        """
        return self._connectors.get(name)

    def list_names(self) -> list:
        """Return all registered connector names (useful for the Planner prompt)."""
        return list(self._connectors.keys())

    def as_dict(self) -> Dict[str, BaseConnector]:
        """Return the raw name→connector mapping (for backward compat)."""
        return dict(self._connectors)

    def __contains__(self, name: str) -> bool:
        return name in self._connectors

    def __repr__(self) -> str:
        names = ", ".join(self._connectors.keys())
        return f"ConnectorRegistry([{names}])"
