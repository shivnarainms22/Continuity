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

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from continuity.analysis import cli as cli_module
from continuity.analysis.detect import AnomalyWindow
from continuity.analysis.slices import Slice

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

# Measured directly over INC-APP-ROKU-820's true 8-hour ground-truth window
# (device_type=roku AND app_version=8.2.0, 2026-02-12 18:00 to 2026-02-13 02:00) against
# this frozen, seeded dataset. detect() runs on the WHOLE population, where this slice is
# only ~8% of sessions -- a diluted signal that breaches threshold at just a handful of
# peaks, fragmenting into several raw anomaly windows separated by quiet stretches. Merging
# those windows recovers ONE incident but still bounds it by the population-level span.
# Re-detecting on the isolated blast radius (continuity/analysis/cli.py::refine_incident,
# z 19-83 there versus 4-7 at population level) recovers the true onset/offset almost
# exactly, which is what makes the tolerance below tight rather than the generous one
# merging alone could support.
_TRUE_ROKU_820_SUBSCRIBERS = 3689
_TRUE_ROKU_820_ARR = 36094
_MIN_ACCEPTABLE_FRACTION = 0.85
_MAX_ACCEPTABLE_FRACTION = 1.2

# Ground truth plants a 4.5x rebuffer multiplier for this incident (see
# data/ground_truth.json's "effects": [{"metric": "rebuffer", "multiplier": 4.5}]) --
# i.e. the typical deviation ratio (actual - expected) / expected should land near
# 4.5 - 1 = 3.5. This is a genuinely checkable claim, not a tuned one: the median is
# computed by continuity/analysis/cli.py::_typical_and_peak_deviation from the raw
# bucket series, with no knowledge of this planted value.
_PLANTED_ROKU_820_MULTIPLIER = 4.5
_MIN_TYPICAL_MULTIPLIER = 3.5
_MAX_TYPICAL_MULTIPLIER = 5.5

_DT_PAIR = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"


def _parse_dt(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def test_investigate_by_incident_id_refines_to_the_true_span_and_impact():
    result = _run_cli("--incident", "INC-APP-ROKU-820")

    assert result.returncode == 0, result.stderr
    output = result.stdout
    lowered = output.lower()

    assert "roku" in lowered, output
    assert "8.2.0" in output, output

    # Every anomaly window found must genuinely isolate the planted fault's blast
    # radius -- not just mention "roku" and "8.2.0" somewhere incidental.
    assert "device_type = roku" in output
    assert "app_version = 8.2.0" in output

    # The 4 raw detection windows for this incident (same blast radius, gaps well under
    # the default 2h merge window) must merge into exactly ONE incident, not four separate
    # briefs each describing a fragment.
    section_count = output.count("SUBSCRIBERS AFFECTED AND ARR AT RISK")
    assert section_count == 1, (
        f"expected the raw anomaly windows to merge into ONE incident, got {section_count} "
        f"separate incident sections:\n{output}"
    )

    # Re-detecting on the isolated blast radius must recover a span at least as wide as
    # the population-level one that seeded it -- that is the entire point of refinement.
    pop_pattern = rf"Detected at population level between {_DT_PAIR} and {_DT_PAIR}"
    refined_pattern = rf"the fault actually ran {_DT_PAIR} to {_DT_PAIR} \("
    pop_match = re.search(pop_pattern, output)
    refined_match = re.search(refined_pattern, output)
    assert pop_match, f"expected a population-level span line in:\n{output}"
    assert refined_match, f"expected a refined span line (no fallback) in:\n{output}"
    pop_start, pop_end = _parse_dt(pop_match.group(1)), _parse_dt(pop_match.group(2))
    refined_start = _parse_dt(refined_match.group(1))
    refined_end = _parse_dt(refined_match.group(2))
    assert refined_end - refined_start >= pop_end - pop_start, (
        f"refined span {refined_start}..{refined_end} is narrower than the population-level "
        f"span {pop_start}..{pop_end} it was supposed to widen"
    )

    # The refined onset must be at or before the population-level detection's first burst,
    # and the refined end at or after its last -- refinement only ever widens.
    assert refined_start <= pop_start
    assert refined_end >= pop_end

    subscriber_counts = [
        int(m.group(1).replace(",", ""))
        for m in re.finditer(r"Affected subscribers:\s*([\d,]+)", output)
    ]
    assert len(subscriber_counts) == 1
    count = subscriber_counts[0]
    assert count >= _TRUE_ROKU_820_SUBSCRIBERS * _MIN_ACCEPTABLE_FRACTION, (
        f"refined subscriber count {count} is too far below the true "
        f"~{_TRUE_ROKU_820_SUBSCRIBERS} measured over the incident's full window"
    )
    assert count <= _TRUE_ROKU_820_SUBSCRIBERS * _MAX_ACCEPTABLE_FRACTION, (
        f"refined subscriber count {count} overshoots the true ~{_TRUE_ROKU_820_SUBSCRIBERS} "
        "by more than expected"
    )

    expected_arr_match = re.search(r"expected \$([\d,]+\.\d{2})\)", output)
    assert expected_arr_match, f"expected an 'expected $X' ARR figure in:\n{output}"
    expected_arr = float(expected_arr_match.group(1).replace(",", ""))
    assert expected_arr > 1000, f"expected a substantial ARR figure, got {expected_arr}"
    assert expected_arr >= _TRUE_ROKU_820_ARR * _MIN_ACCEPTABLE_FRACTION, (
        f"expected ARR {expected_arr} is too far below the true ~{_TRUE_ROKU_820_ARR}"
    )
    assert expected_arr <= _TRUE_ROKU_820_ARR * _MAX_ACCEPTABLE_FRACTION, (
        f"expected ARR {expected_arr} overshoots the true ~{_TRUE_ROKU_820_ARR} by more "
        "than expected"
    )

    # Severity fed into impact must be the TYPICAL (median) deviation across the span, not
    # the single worst bucket -- both are shown, clearly labelled, so a reader cannot
    # mistake one for the other.
    typical_match = re.search(r"Typical degradation across the span: ([\d.]+)x baseline", output)
    peak_match = re.search(r"worst single bucket: ([\d.]+)x baseline", output)
    assert typical_match, f"expected a 'Typical degradation' line in:\n{output}"
    assert peak_match, f"expected a 'worst single bucket' line in:\n{output}"
    typical_multiplier = float(typical_match.group(1))
    peak_multiplier = float(peak_match.group(1))
    assert peak_multiplier > typical_multiplier, (
        "the worst single bucket should read worse than the typical (median) deviation, got "
        f"typical={typical_multiplier} peak={peak_multiplier}"
    )

    # The checkable claim: ground truth plants a 4.5x rebuffer multiplier for this incident
    # (see the module comment above), so the independently-measured typical multiplier
    # should land close to it -- this was never told the planted value.
    assert _MIN_TYPICAL_MULTIPLIER <= typical_multiplier <= _MAX_TYPICAL_MULTIPLIER, (
        f"typical degradation multiplier {typical_multiplier}x is far from the planted "
        f"{_PLANTED_ROKU_820_MULTIPLIER}x -- expected the median across anomalous buckets "
        "to recover something close to the planted severity"
    )

    # Correlating against the TRUE (refined) onset must score at least as well as
    # correlating against the first population-level peak did before refinement (0.22).
    score_match = re.search(r"Confidence score: ([\d.]+)", output)
    assert score_match, f"expected a confidence score line in:\n{output}"
    assert float(score_match.group(1)) > 0.3, (
        f"expected refinement to improve the correlation score above the pre-refinement "
        f"0.22, got {score_match.group(1)}"
    )

    # The recommended action is explicitly a proposal, never framed as already done.
    assert "PROPOSAL" in output
    assert "REQUIRES HUMAN APPROVAL" in output


# --- the fallback path: refinement must never silently produce a worse answer --------


def _fake_anomaly_window(start: datetime, end: datetime, peak_z: float = 5.0) -> AnomalyWindow:
    return AnomalyWindow(
        slice=Slice(),
        metric="rebuffer",
        start=start,
        end=end,
        peak_z=peak_z,
        peak_value=0.01,
        expected_at_peak=0.002,
        bucket_count=max(1, int((end - start) / timedelta(minutes=5))),
        sql="SELECT 1",
    )


def _fake_walk_result(final_slice: Slice, window: tuple[datetime, datetime]):
    return cli_module.WalkResult(
        metric="rebuffer",
        window=window,
        baseline_windows=(),
        path=(),
        final_slice=final_slice,
        stop_reason=cli_module.StopReason.DIMENSIONS_EXHAUSTED,
        stop_detail="test fixture",
        elapsed_ms=0.0,
        query_log=(),
    )


class _FakeQueryResult:
    """The minimal shape `_typical_and_peak_deviation` reads: no observations at all,
    so its own re-labelling degrades gracefully to the peak-ratio fallback rather than
    crashing on an empty series -- exercising that fallback-within-a-fallback path too."""

    rows: list[dict] = []


class _FakeGateway:
    """A gateway double for the severity re-labelling query `refine_incident` also
    issues -- `detect()` itself is monkeypatched separately, but `_typical_and_peak_deviation`
    calls `gateway.query()` directly and must have something to call."""

    def __init__(self) -> None:
        self.query_log: list = []

    async def query(self, sql: str) -> _FakeQueryResult:
        return _FakeQueryResult()


def test_refine_incident_falls_back_to_the_population_span_when_it_finds_nothing(monkeypatch):
    """`refine_incident` re-detects on the isolated blast radius -- a pure orchestration
    step this test exercises directly (no DB needed: `detect` is monkeypatched) rather
    than trying to provoke a thin-slice fallback from live data, which cannot be forced
    without hardcoding a fragile dataset property. Never touches the network."""
    slice_ = Slice().refine("device_type", "roku")
    pop_start = datetime(2026, 2, 12, 19, 40, 0)
    pop_end = datetime(2026, 2, 12, 20, 15, 0)
    window = _fake_anomaly_window(pop_start, pop_end)
    walk_result = _fake_walk_result(slice_, (pop_start, pop_end))
    incident = cli_module.MergedIncident(windows=(window,), walks=(walk_result,))

    async def fake_detect(gateway, slice_arg, metric_name, start, end, **kwargs):
        assert slice_arg == slice_
        return cli_module.DetectionResult(
            slice=slice_arg,
            metric=metric_name,
            windows=[],
            total_buckets=1,
            anomalous_buckets=0,
            unknown_buckets=0,
            sql="SELECT 1",
        )

    monkeypatch.setattr(cli_module, "detect", fake_detect)
    refined = asyncio.run(
        cli_module.refine_incident(
            _FakeGateway(),
            incident,
            metric_name="rebuffer",
            refine_padding=timedelta(hours=6),
        )
    )

    assert refined.used_fallback is True
    assert refined.fallback_reason is not None
    assert "population-level span" in refined.fallback_reason
    assert refined.span == incident.span
    assert refined.windows == incident.windows
    # Severity is still measured (over the fallback span), never left undefined.
    assert refined.typical_deviation_ratio > 0
    assert refined.peak_deviation_ratio > 0


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
