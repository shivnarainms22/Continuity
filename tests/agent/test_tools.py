"""Tests for continuity.agent.tools: the tool layer that turns the analysis
primitives into ADK FunctionTools.

Everything above the "LIVE DATABASE" section is a unit test against a fake
gateway that stands in for ClickHouse -- no Docker needed. The one test in that
final section is marked `@pytest.mark.integration` and proves, against the real
59.8M+-event dataset, that `split_on_dimension`'s lift signal for
INC-APP-ROKU-820 is actually there in the tool's own output, not just in
`continuity/analysis/split.py`'s internals.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from continuity.agent.tools import DEFAULT_TOP_N, AnalysisTools, build_function_tools
from continuity.gateway.mcp_gateway import QueryError, QueryResult

# ---------------------------------------------------------------------------
# A minimal FIFO fake gateway: every tool method issues its queries in a fixed,
# documented order (see tools.py), so canned responses are consumed in order.
# ---------------------------------------------------------------------------


class _FakeGateway:
    def __init__(self, responses: list[QueryResult | BaseException]) -> None:
        self._responses = list(responses)
        self.queries: list[str] = []

    async def query(self, sql: str) -> QueryResult:
        self.queries.append(sql)
        if not self._responses:
            raise AssertionError(f"_FakeGateway received an unexpected extra query:\n{sql}")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _qr(rows: list[dict]) -> QueryResult:
    columns = list(rows[0].keys()) if rows else []
    return QueryResult(sql="<fake>", columns=columns, rows=rows)


_WINDOW_START = "2026-02-12T18:00:00"
_WINDOW_END = "2026-02-13T02:00:00"


def _measure_ok_responses(value: float = 0.05) -> list[QueryResult]:
    """One actual-value query plus the four week-over-week comparison queries
    measure_slice always issues once the actual value is present."""
    return [_qr([{"value": value}])] + [_qr([{"value": 0.01}]) for _ in range(4)]


# ---------------------------------------------------------------------------
# Slice parsing: valid dict, valid JSON string, empty, invalid dimension.
# ---------------------------------------------------------------------------


async def test_valid_dict_slice_is_accepted():
    tools = AnalysisTools(_FakeGateway(_measure_ok_responses()))

    result = await tools.measure_slice(
        {"device_type": "roku"}, "rebuffer", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result
    assert result["slice"] == {"device_type": "roku"}


async def test_valid_json_string_slice_is_accepted():
    tools = AnalysisTools(_FakeGateway(_measure_ok_responses()))

    result = await tools.measure_slice(
        '{"device_type": "roku"}', "rebuffer", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result
    assert result["slice"] == {"device_type": "roku"}


async def test_empty_slice_means_whole_population():
    tools = AnalysisTools(_FakeGateway(_measure_ok_responses()))

    result = await tools.measure_slice({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert "error" not in result
    assert result["slice"] == {}


async def test_empty_string_slice_means_whole_population():
    tools = AnalysisTools(_FakeGateway(_measure_ok_responses()))

    result = await tools.measure_slice("", "rebuffer", _WINDOW_START, _WINDOW_END)

    assert "error" not in result
    assert result["slice"] == {}


async def test_unknown_dimension_returns_invalid_input_error_naming_valid_dimensions():
    """The acceptance criterion this exists for: a model WILL guess a wrong
    dimension name (e.g. 'devicetype' instead of 'device_type')."""
    tools = AnalysisTools(_FakeGateway([]))  # no query should ever be issued

    result = await tools.measure_slice(
        {"devicetype": "roku"}, "rebuffer", _WINDOW_START, _WINDOW_END
    )

    assert result["error_type"] == "invalid_input"
    assert "devicetype" in result["error"]
    assert "device_type" in result["error"]  # names a real, valid dimension to retry with


async def test_malformed_json_string_slice_returns_invalid_input_error():
    tools = AnalysisTools(_FakeGateway([]))

    result = await tools.measure_slice("{not json", "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "invalid_input"


async def test_non_object_json_string_slice_returns_invalid_input_error():
    tools = AnalysisTools(_FakeGateway([]))

    result = await tools.measure_slice("[1, 2, 3]", "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "invalid_input"


# ---------------------------------------------------------------------------
# Other input validation shared by every tool: unknown metric, bad window.
# ---------------------------------------------------------------------------


async def test_unknown_metric_returns_invalid_input_error_naming_known_metrics():
    tools = AnalysisTools(_FakeGateway([]))

    result = await tools.measure_slice({}, "throughput", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "invalid_input"
    assert "rebuffer" in result["error"]  # names a real, valid metric to retry with


async def test_window_end_before_start_returns_invalid_input_error():
    tools = AnalysisTools(_FakeGateway([]))

    result = await tools.measure_slice({}, "rebuffer", _WINDOW_END, _WINDOW_START)

    assert result["error_type"] == "invalid_input"


async def test_non_iso_datetime_returns_invalid_input_error():
    tools = AnalysisTools(_FakeGateway([]))

    result = await tools.measure_slice({}, "rebuffer", "not-a-date", _WINDOW_END)

    assert result["error_type"] == "invalid_input"


# ---------------------------------------------------------------------------
# Error shapes: infrastructure_failure must never collapse into no_data.
# ---------------------------------------------------------------------------


async def test_gateway_query_error_is_reported_as_infrastructure_failure():
    tools = AnalysisTools(_FakeGateway([QueryError("connection reset by peer")]))

    result = await tools.measure_slice({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "infrastructure_failure"
    assert "connection reset" in result["error"]


async def test_missing_actual_value_is_reported_as_no_data_not_infrastructure_failure():
    """A slice with zero traffic in the window is a real finding, not a broken pipe --
    the two must never be reported the same way."""
    tools = AnalysisTools(_FakeGateway([_qr([{"value": None}])]))

    result = await tools.measure_slice({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "no_data"


async def test_no_data_short_circuits_before_querying_baseline_weeks():
    """Only one query (the actual) should ever be issued when it comes back empty --
    spending four more queries on baseline weeks for a value we already know is
    unmeasurable would be wasted work."""
    fake = _FakeGateway([_qr([{"value": None}])])
    tools = AnalysisTools(fake)

    await tools.measure_slice({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert len(fake.queries) == 1


# ---------------------------------------------------------------------------
# Every successful result carries its SQL.
# ---------------------------------------------------------------------------


async def test_measure_slice_success_carries_sql_and_baseline_sql():
    responses = [_qr([{"value": 0.05}])] + [_qr([{"value": 0.01}]) for _ in range(4)]
    tools = AnalysisTools(_FakeGateway(responses))

    result = await tools.measure_slice({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert "error" not in result
    assert result["sql"] and "SELECT" in result["sql"]
    assert result["baseline_sql"] and "SELECT" in result["baseline_sql"]


async def test_measure_slice_status_and_z_match_baseline_module_directly():
    """The tool must not reinvent the statistics -- its numbers must equal what
    baseline.compute_baseline itself produces for the same inputs."""
    from continuity.analysis.baseline import compute_baseline

    actual = 0.05
    comparisons = [0.008, 0.009, 0.011, 0.012]
    responses = [_qr([{"value": actual}])] + [_qr([{"value": v}]) for v in comparisons]
    tools = AnalysisTools(_FakeGateway(responses))

    result = await tools.measure_slice({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    expected = compute_baseline(actual, comparisons)
    assert result["value"] == actual
    assert result["baseline"] == pytest.approx(expected.expected)
    assert result["z"] == pytest.approx(expected.z)
    assert result["sample_size"] == expected.sample_size
    assert result["status"] == expected.status.value


async def test_detect_anomalies_success_carries_sql():
    tools = AnalysisTools(_FakeGateway([_qr([])]))  # no observations at all

    result = await tools.detect_anomalies({}, "rebuffer", _WINDOW_START, "2026-02-12T18:20:00")

    assert "error" not in result
    assert result["sql"] and "SELECT" in result["sql"]
    # Zero observations -> every bucket UNKNOWN, never silently "quiet".
    assert result["total_buckets"] == 4
    assert result["unknown_buckets"] == 4
    assert result["unknown_fraction"] == pytest.approx(1.0)
    assert result["windows"] == []


async def test_find_changes_success_carries_sql():
    tools = AnalysisTools(_FakeGateway([_qr([])]))  # no change_log rows

    result = await tools.find_changes({}, _WINDOW_START)

    assert "error" not in result
    assert result["sql"] and "SELECT" in result["sql"]
    assert result["candidates"] == []
    assert result["rejected"] == []


async def test_quantify_impact_success_carries_sql():
    tools = AnalysisTools(_FakeGateway([_qr([])]))  # no affected subscribers

    result = await tools.quantify_impact({}, _WINDOW_START, _WINDOW_END, 0.5)

    assert "error" not in result
    assert result["sql"] and "SELECT" in result["sql"]
    assert result["affected_subscribers"] == 0
    assert result["arr_at_risk_expected"] == "0.00"
    assert "notes" in result["methodology"]


async def test_quantify_impact_rejects_negative_severity_ratio():
    tools = AnalysisTools(_FakeGateway([]))

    result = await tools.quantify_impact({}, _WINDOW_START, _WINDOW_END, -0.5)

    assert result["error_type"] == "invalid_input"


# ---------------------------------------------------------------------------
# find_changes: rejected candidates carry human-readable reasons.
# ---------------------------------------------------------------------------


async def test_find_changes_reports_a_too_late_change_as_rejected_with_a_reason():
    onset = datetime(2026, 2, 12, 18, 0, 0)
    too_late = onset + timedelta(hours=2)
    rows = [
        {
            "change_id": 1,
            "changed_at": too_late.strftime("%Y-%m-%d %H:%M:%S"),
            "change_type": "app_release",
            "component": "roku_app",
            "description": "irrelevant, arrived after onset",
            "dimension_key": "app_version",
            "dimension_value": "8.2.0",
        }
    ]
    tools = AnalysisTools(_FakeGateway([_qr(rows)]))

    result = await tools.find_changes({}, onset.isoformat())

    assert result["candidates"] == []
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["reason"]  # non-empty, human-readable
    assert "too late" in result["rejected"][0]["reason"]


# ---------------------------------------------------------------------------
# split_on_dimension: lift is present, and truncation is reported honestly.
# ---------------------------------------------------------------------------


def _split_responses(n_values: int) -> list[QueryResult]:
    """`n_values` distinct device_type values, one wildly worse than baseline
    (so lift is well above 1.0 for it), the rest flat."""
    window_rows = [
        {"dim": "device_type", "value": f"v{i}", "metric_value": 0.001, "weight": 10_000.0}
        for i in range(n_values)
    ]
    window_rows[0] = {
        "dim": "device_type",
        "value": "v0",
        "metric_value": 0.05,
        "weight": 10_000.0,
    }
    baseline_rows = [
        {"dim": "device_type", "value": f"v{i}", "metric_value": 0.001, "weight": 10_000.0}
        for i in range(n_values)
    ]
    # 1 window query + 4 median-baseline weeks, all identical for this fixture.
    return [_qr(window_rows)] + [_qr(baseline_rows) for _ in range(4)]


async def test_split_on_dimension_carries_lift_on_every_value():
    tools = AnalysisTools(_FakeGateway(_split_responses(3)))

    result = await tools.split_on_dimension(
        {}, "rebuffer", "device_type", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result
    assert result["values"], "expected at least one ranked value"
    assert all("lift" in v for v in result["values"])
    top = result["values"][0]
    assert top["value"] == "v0"
    assert top["lift"] is not None and top["lift"] > 1.5
    assert result["sql"] and result["baseline_sql"]


async def test_split_on_dimension_truncates_and_reports_omitted_count():
    n_values = DEFAULT_TOP_N + 5
    tools = AnalysisTools(_FakeGateway(_split_responses(n_values)))

    result = await tools.split_on_dimension(
        {}, "rebuffer", "device_type", _WINDOW_START, _WINDOW_END
    )

    assert len(result["values"]) == DEFAULT_TOP_N
    assert result["values_omitted"] == 5


async def test_split_on_dimension_respects_a_custom_top_n():
    tools = AnalysisTools(_FakeGateway(_split_responses(10)))

    result = await tools.split_on_dimension(
        {}, "rebuffer", "device_type", _WINDOW_START, _WINDOW_END, top_n=2
    )

    assert len(result["values"]) == 2
    assert result["values_omitted"] == 8


async def test_split_on_dimension_no_rows_at_all_is_reported_as_no_data():
    responses = [_qr([])] + [_qr([]) for _ in range(4)]
    tools = AnalysisTools(_FakeGateway(responses))

    result = await tools.split_on_dimension(
        {}, "rebuffer", "device_type", _WINDOW_START, _WINDOW_END
    )

    assert result["error_type"] == "no_data"


async def test_split_on_dimension_unknown_dimension_is_invalid_input_before_any_query():
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)

    result = await tools.split_on_dimension(
        {}, "rebuffer", "devicetype", _WINDOW_START, _WINDOW_END
    )

    assert result["error_type"] == "invalid_input"
    assert "device_type" in result["error"]
    assert fake.queries == []


# ---------------------------------------------------------------------------
# ADK wiring: FunctionTool construction is static (no model call) and the
# gateway never leaks into the schema the model would see.
# ---------------------------------------------------------------------------


def test_build_function_tools_returns_five_tools_named_after_the_primitives():
    fake = _FakeGateway([])
    tools = build_function_tools(fake)

    names = {tool.name for tool in tools}
    assert names == {
        "detect_anomalies",
        "measure_slice",
        "split_on_dimension",
        "find_changes",
        "quantify_impact",
    }


def test_split_on_dimension_tool_schema_never_exposes_the_gateway():
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)
    from google.adk.tools import FunctionTool

    tool = FunctionTool(tools.split_on_dimension)
    declaration = tool._get_declaration()

    assert declaration.description  # the docstring, i.e. the model's prompt for this tool
    assert "lift" in declaration.description
    # ADK derives the schema from the bound method's signature -- `self` is
    # already bound and `gateway` is an instance attribute, so neither can ever
    # appear as something the model is asked to supply.
    schema = declaration.parameters_json_schema or (
        declaration.parameters.model_dump() if declaration.parameters else {}
    )
    property_names = set((schema.get("properties") or {}).keys())
    assert "gateway" not in property_names
    assert "self" not in property_names
    assert {"slice_json", "metric", "dimension", "window_start", "window_end"} <= property_names


# ---------------------------------------------------------------------------
# LIVE DATABASE: proves the lift signal split_on_dimension needs to hand the
# model is genuinely present in the tool's own output, not just internal to
# continuity/analysis/split.py. Read only. Uses the DEFAULT database from
# ClickHouseConfig.from_env() via the `gateway` fixture in tests/conftest.py --
# the full 63.85M-event dataset. Window comes from data/ground_truth.json,
# never hardcoded (per CLAUDE.md).
# ---------------------------------------------------------------------------

_GROUND_TRUTH_PATH = Path(__file__).resolve().parents[2] / "data" / "ground_truth.json"


def _roku_820_window() -> tuple[str, str]:
    payload = json.loads(_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    incident = next(inc for inc in payload["incidents"] if inc["kind"] == "device_app_fault")
    start = datetime.fromisoformat(incident["start"]).replace(tzinfo=None)
    end = datetime.fromisoformat(incident["end"]).replace(tzinfo=None)
    return start.isoformat(), end.isoformat()


@pytest.mark.integration
async def test_split_on_dimension_finds_roku_with_lift_above_threshold_for_the_real_incident(
    gateway,
):
    window_start, window_end = _roku_820_window()
    tools = AnalysisTools(gateway)

    result = await tools.split_on_dimension({}, "rebuffer", "device_type", window_start, window_end)

    assert "error" not in result, result
    assert result["values"], "split returned no device_type values at all"
    top = result["values"][0]
    assert top["value"] == "roku", (
        f"expected roku to rank first by contribution, got {top['value']!r}. "
        f"Full ranking: {[(v['value'], v['lift']) for v in result['values']]}"
    )
    assert top["lift"] is not None and top["lift"] > 1.5, (
        "expected roku's lift > 1.5 -- the signal the model needs to decide to "
        f"descend -- got {top['lift']!r}"
    )
    assert result["sql"] and result["baseline_sql"]
