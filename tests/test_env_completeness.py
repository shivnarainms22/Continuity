"""Guard against .env drifting behind .env.example.

.env is gitignored, so nothing reviews it. When .env.example gained
GOOGLE_CLOUD_LOCATION=global, the developer's .env kept an earlier us-central1 value and
every Gemini call returned 404 NOT_FOUND for a model that exists only on the global
endpoint. Nothing flagged it, because a missing or stale key looks exactly like a
deliberate one.

Comparing the key sets turns that into a failure that names the missing key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
EXAMPLE_FILE = PROJECT_ROOT / ".env.example"


def _keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


def test_env_example_exists_and_defines_keys():
    assert EXAMPLE_FILE.exists(), ".env.example is the documented contract; it must exist"
    assert _keys(EXAMPLE_FILE), ".env.example defines no keys"


@pytest.mark.skipif(not ENV_FILE.exists(), reason="no local .env (CI or a fresh clone)")
def test_local_env_defines_every_key_from_the_example():
    """A key added to .env.example must also reach the developer's .env.

    Skipped when .env is absent so a fresh clone and CI stay green; it only guards the
    machine that actually has one.
    """
    missing = _keys(EXAMPLE_FILE) - _keys(ENV_FILE)

    assert not missing, (
        f"these keys exist in .env.example but not in your .env: {sorted(missing)}. "
        "Copy them across. A stale .env is invisible because the file is gitignored, and "
        "a wrong value fails at runtime far from its cause."
    )
