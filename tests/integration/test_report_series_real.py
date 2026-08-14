"""Integration test for `continuity/api/report_schema.fetch_incident_series` against the
real dataset, read through the MCP gateway.

This is the hero-chart query on the live investigation path, and it was the last caller
still fetching the whole contiguous month of history that `detect()` had already been
fixed off (see continuity/analysis/detect.py's `build_window_series_sql`). That is the
query shape observed hitting ClickHouse `MEMORY_LIMIT_EXCEEDED` during the agent/walker
comparison, so a live demo could hit it too.

The guard below is the same one that fix rests on, applied to this caller: restricting
the fetch must leave every rendered chart point byte-identical. A faster query that
changes what the chart draws would be a regression, not a fix. Both sides run through
the real `fetch_incident_series`, differing only in the SQL builder it is handed, so
the comparison exercises production point-building rather than a copy of it.

Read-only, and windows are derived from data/ground_truth.json rather than hardcoded,
exactly as tests/integration/test_detect_real.py does and for the same reason.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from continuity.analysis.baseline import DEFAULT_LOOKBACK_WEEKS
from continuity.analysis.detect import build_series_sql, fetch_window_start
from continuity.analysis.slices import Slice
from continuity.api import report_schema
from continuity.gateway.mcp_gateway import QueryError

pytestmark = pytest.mark.integration

_GROUND_TRUTH_PATH = Path(__file__).resolve().parents[2] / "data" / "ground_truth.json"


def _incident_by_kind(kind: str) -> dict:
    payload = json.loads(_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    matches = [inc for inc in payload["incidents"] if inc["kind"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind!r} incident in ground truth"
    return matches[0]


def _span(incident: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(incident["start"]).replace(tzinfo=None)
    end = datetime.fromisoformat(incident["end"]).replace(tzinfo=None)
    return start, end


def _slice_for(incident: dict) -> Slice:
    slice_ = Slice()
    for key, value in incident["predicate"].items():
        slice_ = slice_.refine(key, value)
    return slice_


def _full_contiguous_range_sql(slice_, metric, start, end, **_ignored) -> str:
    """`fetch_incident_series`'s PRE-FIX fetch: the whole contiguous range back to
    `DEFAULT_LOOKBACK_WEEKS` ago. Used only as the "before" side of the guard."""
    return build_series_sql(
        slice_, metric, fetch_window_start(start, DEFAULT_LOOKBACK_WEEKS * 7), end
    )


@pytest.mark.parametrize(
    "kind, metric_name",
    [
        # pop_fault / startup is the exact incident+metric pair observed failing with
        # MEMORY_LIMIT_EXCEEDED under the old full-contiguous-range fetch.
        ("pop_fault", "startup"),
        ("device_app_fault", "rebuffer"),
        ("encode_fault", "bitrate"),
    ],
)
async def test_chart_series_is_unchanged_by_the_restricted_history_fetch(
    gateway, monkeypatch, kind, metric_name
):
    incident = _incident_by_kind(kind)
    slice_, span = _slice_for(incident), _span(incident)

    restricted = await report_schema.fetch_incident_series(
        gateway, slice_=slice_, metric_name=metric_name, span=span
    )
    assert (
        "bucket IN (" in restricted["sql"]
        or "toStartOfFiveMinute(event_time) IN (" in restricted["sql"]
    ), "the chart query must fetch history as an explicit bucket list, not a wide range"
    restricted_rows = gateway.query_log[-1].row_count

    monkeypatch.setattr(report_schema, "build_window_series_sql", _full_contiguous_range_sql)
    try:
        full = await report_schema.fetch_incident_series(
            gateway, slice_=slice_, metric_name=metric_name, span=span
        )
    except QueryError as exc:
        if "MEMORY_LIMIT_EXCEEDED" in str(exc):
            pytest.skip(
                "the pre-fix full-range chart fetch hit MEMORY_LIMIT_EXCEEDED on this "
                "host -- exactly the bug this removes from the demo path; the "
                "restricted fetch above already succeeded, so there is nothing left "
                "to compare against."
            )
        raise
    full_rows = gateway.query_log[-1].row_count

    assert restricted["points"] == full["points"]
    assert restricted_rows < full_rows * 0.5, (
        f"expected the restricted chart fetch to read far fewer rows: "
        f"full={full_rows} restricted={restricted_rows}"
    )
