"""Yantrik Companion Agent — the brain of Yantrik OS."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yantrikdb.agent.companion import CompanionService

__all__ = ["CompanionService"]


def __getattr__(name: str):
    """Keep optional companion dependencies lazy for submodule consumers."""
    if name == "CompanionService":
        from yantrikdb.agent.companion import CompanionService

        return CompanionService
    raise AttributeError(name)
