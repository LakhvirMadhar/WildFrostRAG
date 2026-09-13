"""Single source of truth for chat completion message role identifiers."""

from enum import StrEnum


class ChatRole(StrEnum):
    """The chat message roles this project actually constructs."""

    SYSTEM = "system"
    USER = "user"
