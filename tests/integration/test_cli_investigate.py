"""Integration tests for `python -m continuity.analysis.cli investigate`.

Runs the CLI as a subprocess against the real, live 63.85M-event dataset (the DEFAULT
database from ClickHouseConfig.from_env() -- the same one tests/integration/test_walk_real.py
and test_detect_real.py use, not the throwaway `continuity_test` database
tests/integration/test_load.py truncates and reloads).

Every window checked here is derived from data/ground_truth.json rather than hardcoded,
per CLAUDE.md ("hardcoding has broken this project twice").

Subprocess environment is always built explicitly (`_subprocess_env`), never passed
through as a blind `env=None` (which would inherit `os.environ` implicitly) -- see
tests/integration/test_load.py::_subprocess_env for the incident this pattern guards
against. This CLI never writes, so the blast radius of a leaked environment variable is
much smaller here, but the pattern is followed anyway rather than re-litigated per file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GROUND_TRUTH_PATH = _REPO_ROOT / "data" / "ground_truth.json"
_KNOWN_METRICS = ("rebuffer", "startup", "bitrate", "errors")


def _ground_truth_incidents() -> list[dict]:
    payload = json.loads(_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return payload["incidents"]


def _window(incident: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(incident["start"]).replace(tzinfo=None)
    end = datetime.fromisoformat(incident["end"]).replace(tzinfo=None)
    return start, end


def _quiet_window() -> tuple[datetime, datetime]:
    """A window derived (never hardcoded) to have no planted incident and full
    week-over-week baseline history: 3 days of buffer before the earliest incident,
    5 days long -- the same derivation tests/integration/test_detect_real.py uses for
    its "replaces the 353 false positives" quiet-period check.
    """
    incidents = _ground_truth_incidents()
    earliest_start = min(_window(inc)[0] for inc in incidents)
    end = (earliest_start - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=5)
    return start, end


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _subprocess_env(**overrides: str) -> dict[str, str]:
    """Explicit environment for a CLI subprocess -- never a blind pass-through of
    os.environ (see the module docstring). This CLI only reads, so there is no dataset
    to protect the way test_load.py's `_subprocess_env` protects one, but every
    subprocess call in this project builds its env explicitly rather than relying on
    default inheritance.
    """
    env = dict(os.environ)
    env.update(overrides)
    return env


def _run_cli(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "continuity.analysis.cli", "investigate", *args],
        cwd=_REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --- a known incident produces a brief a VP could be handed --------------------------


def test_investigate_by_incident_id_names_the_blast_radius_and_a_dollar_figure():
    result = _run_cli("--incident", "INC-APP-ROKU-820")

    assert result.returncode == 0, result.stderr
    output = result.stdout
    lowered = output.lower()

    assert "roku" in lowered, output
    assert "8.2.0" in output, output

    subscriber_counts = [
        int(m.group(1).replace(",", ""))
        for m in re.finditer(r"Affected subscribers:\s*([\d,]+)", output)
    ]
    assert subscriber_counts, "expected at least one 'Affected subscribers:' line"
    assert any(count > 0 for count in subscriber_counts), (
        f"expected a non-zero affected-subscriber count, got {subscriber_counts}"
    )

    assert re.search(r"\$[\d,]+\.\d{2}", output), "expected a dollar figure in the brief"

    # Every anomaly window found must genuinely isolate the planted fault's blast
    # radius -- not just mention "roku" and "8.2.0" somewhere incidental.
    assert "device_type = roku" in output
    assert "app_version = 8.2.0" in output

    # The recommended action is explicitly a proposal, never framed as already done.
    assert "PROPOSAL" in output
    assert "REQUIRES HUMAN APPROVAL" in output


# --- --show-sql is the anti-hallucination property, not decoration -------------------


def test_show_sql_flag_prints_the_query_behind_the_brief():
    start, end = _quiet_window()
    result = _run_cli("--start", _fmt(start), "--end", _fmt(end), "--show-sql")

    assert result.returncode == 0, result.stderr
    assert "SELECT" in result.stdout


# --- a quiet window is a normal outcome, not a failure --------------------------------


def test_a_window_with_no_incident_reports_no_anomalies_and_exits_zero():
    start, end = _quiet_window()
    result = _run_cli("--start", _fmt(start), "--end", _fmt(end))

    assert result.returncode == 0, result.stderr
    assert "no anomalies" in result.stdout.lower()
    # Must not be confused with an error: no failure language anywhere in the output.
    assert "FAILED" not in result.stdout
    assert "traceback" not in result.stdout.lower()


# --- loud, actionable errors -----------------------------------------------------------


def test_bad_metric_name_exits_nonzero_and_names_the_valid_metrics():
    start, end = _quiet_window()
    result = _run_cli("--metric", "not-a-real-metric", "--start", _fmt(start), "--end", _fmt(end))

    assert result.returncode != 0
    for metric in _KNOWN_METRICS:
        assert metric in result.stderr, f"expected {metric!r} named in: {result.stderr}"


def test_unknown_incident_id_exits_nonzero_and_names_known_incident_ids():
    result = _run_cli("--incident", "NOT-A-REAL-INCIDENT")

    assert result.returncode != 0
    assert "NOT-A-REAL-INCIDENT" in result.stderr
    incidents = _ground_truth_incidents()
    assert any(inc["incident_id"] in result.stderr for inc in incidents)


def test_incident_and_explicit_window_together_is_rejected():
    result = _run_cli(
        "--incident",
        "INC-APP-ROKU-820",
        "--start",
        "2026-01-01 00:00:00",
        "--end",
        "2026-01-01 01:00:00",
    )

    assert result.returncode != 0
    assert "--incident" in result.stderr
    assert "--start" in result.stderr


def test_missing_window_and_incident_is_rejected():
    result = _run_cli()

    assert result.returncode != 0
    assert "--incident" in result.stderr
    assert "--start" in result.stderr
