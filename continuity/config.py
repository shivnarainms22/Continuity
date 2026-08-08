"""Environment-backed configuration.

Fails loudly and early. A misconfigured connection that silently falls back to a default
is far worse than one that refuses to start: the former surfaces later as an empty result
set that every downstream stage reports as "no anomaly found".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

_DEFAULT_PORT = 8123
_DEFAULT_USER = "default"
_DEFAULT_DATABASE = "continuity"


def _require(name: str) -> str:
    """Read a variable that must be present and non-blank."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or export {name} in your shell."
        )
    return value


def _optional(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _parse_port(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return _DEFAULT_PORT
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{name} is set but blank. Unset it to use the default {_DEFAULT_PORT}.")
    try:
        port = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name}={port} is outside the valid range 1-65535.")
    return port


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    # Defaulting an unrecognised value to False would silently send credentials over
    # plaintext, so this is an error rather than a fallback.
    raise ValueError(
        f"{name}={raw!r} is not a recognised boolean. "
        f"Use one of: {', '.join(sorted(_TRUTHY | _FALSY))}."
    )


@dataclass(frozen=True)
class ClickHouseConfig:
    """Connection settings for both the MCP gateway and the bulk loader."""

    host: str
    port: int
    user: str
    password: str
    database: str
    secure: bool

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        return cls(
            host=_require("CLICKHOUSE_HOST"),
            port=_parse_port("CLICKHOUSE_PORT"),
            user=_optional("CLICKHOUSE_USER", _DEFAULT_USER),
            password=_require("CLICKHOUSE_PASSWORD"),
            database=_optional("CLICKHOUSE_DATABASE", _DEFAULT_DATABASE),
            secure=_parse_bool("CLICKHOUSE_SECURE", default=False),
        )

    def __repr__(self) -> str:
        # This object is logged during agent runs and appears in demo screenshots.
        return (
            f"{type(self).__name__}(host={self.host!r}, port={self.port}, "
            f"user={self.user!r}, password='***', database={self.database!r}, "
            f"secure={self.secure})"
        )
