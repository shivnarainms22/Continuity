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
from continuity.analysis.baseline import DEFAULT_LOOKBACK_WEEKS
from continuity.data.topology import DIMENSION_HIERARCHY
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


async def test_quantify_impact_returns_no_data_when_no_anomaly_window_is_found():
    """quantify_impact measures its own severity -- see the module docstring's
    "DEFECT 2" fix -- so with zero observations (no anomaly window at all) it must
    report no_data rather than fabricate a severity of 0.0."""
    tools = AnalysisTools(_FakeGateway([_qr([])]))  # zero observations -> no windows

    result = await tools.quantify_impact({}, "rebuffer", _WINDOW_START, "2026-02-12T18:20:00")

    assert result["error_type"] == "no_data"
    assert "refine_incident_span" in result["error"] or "detect_anomalies" in result["error"]


async def test_quantify_impact_has_no_severity_ratio_parameter():
    """DEFECT 2: severity_ratio must be unrepresentable, not merely discouraged."""
    import inspect

    params = set(inspect.signature(AnalysisTools.quantify_impact).parameters)
    assert "severity_ratio" not in params
    assert {"slice_json", "metric", "window_start", "window_end"} <= params


async def test_quantify_impact_rejects_unknown_metric_before_any_query():
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)

    result = await tools.quantify_impact({}, "throughput", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "invalid_input"
    assert "rebuffer" in result["error"]
    assert fake.queries == []


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


def _gate_and_share_responses() -> list[QueryResult]:
    """Three `device_type` values within ONE split: `roku` (lift 2.5, share 0.75 --
    the broad, true-blast-radius value), `subset` (lift 2.5, share 0.25 -- a
    proportional THIRD of `roku`'s own weight carrying the identical per-unit
    deviation ratio, so lift comes out equal by construction), and `bystander`
    (zero deviation -- lift 0.0, fails the gate outright)."""
    window_rows = [
        {"dim": "device_type", "value": "roku", "metric_value": 0.05, "weight": 3000.0},
        {"dim": "device_type", "value": "subset", "metric_value": 0.05, "weight": 1000.0},
        {"dim": "device_type", "value": "bystander", "metric_value": 0.01, "weight": 6000.0},
    ]
    baseline_rows = [
        {"dim": "device_type", "value": "roku", "metric_value": 0.01, "weight": 3000.0},
        {"dim": "device_type", "value": "subset", "metric_value": 0.01, "weight": 1000.0},
        {"dim": "device_type", "value": "bystander", "metric_value": 0.01, "weight": 6000.0},
    ]
    return [_qr(window_rows)] + [_qr(baseline_rows) for _ in range(4)]


async def test_split_on_dimension_prefers_broader_value_over_equal_lift_proportional_subset_trap():
    """THE TRAP within a single dimension's own values: `subset` is a proportional
    THIRD of `roku`'s weight carrying the same per-unit deviation ratio, so its
    lift is IDENTICAL to `roku`'s (lift is scale-invariant) but its
    share_of_deviation is a third of `roku`'s. `roku` must rank first."""
    tools = AnalysisTools(_FakeGateway(_gate_and_share_responses()))

    result = await tools.split_on_dimension(
        {}, "rebuffer", "device_type", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result, result
    by_value = {v["value"]: v for v in result["values"]}
    roku, subset = by_value["roku"], by_value["subset"]

    assert roku["meets_lift_gate"] is True
    assert subset["meets_lift_gate"] is True
    assert roku["lift"] == pytest.approx(subset["lift"], rel=1e-6)
    assert roku["share_of_deviation"] > subset["share_of_deviation"]
    assert result["values"][0]["value"] == "roku"


async def test_split_on_dimension_sorts_lift_gate_failures_after_qualifying_values():
    """`bystander` has zero deviation (lift 0.0, fails the gate) and must sort
    after every value that clears it, even though it is not the smallest raw
    contribution -- it is marked `meets_lift_gate: False` rather than hidden."""
    tools = AnalysisTools(_FakeGateway(_gate_and_share_responses()))

    result = await tools.split_on_dimension(
        {}, "rebuffer", "device_type", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result, result
    values_in_order = [v["value"] for v in result["values"]]
    assert values_in_order == ["roku", "subset", "bystander"]
    bystander = next(v for v in result["values"] if v["value"] == "bystander")
    assert bystander["meets_lift_gate"] is False


# ---------------------------------------------------------------------------
# split_all_dimensions: DEFECT 3 -- every candidate dimension in one call, ranked
# by share_of_deviation among lift-qualifying candidates (lift gates, share ranks),
# excluding whatever is already pinned in the slice.
# ---------------------------------------------------------------------------


def _split_all_responses() -> list[QueryResult]:
    """Only `app_version` carries real rows (two values, one wildly worse -> high
    lift); every other candidate dimension has no rows at all, so the tool must
    report those as "no data" rather than erroring the whole call."""
    window_rows = [
        {"dim": "app_version", "value": "8.2.0", "metric_value": 0.05, "weight": 10_000.0},
        {"dim": "app_version", "value": "8.1.0", "metric_value": 0.001, "weight": 10_000.0},
    ]
    baseline_rows = [
        {"dim": "app_version", "value": "8.2.0", "metric_value": 0.001, "weight": 10_000.0},
        {"dim": "app_version", "value": "8.1.0", "metric_value": 0.001, "weight": 10_000.0},
    ]
    return [_qr(window_rows)] + [_qr(baseline_rows) for _ in range(4)]


async def test_split_all_dimensions_ranks_by_lift_and_excludes_the_pinned_dimension():
    fake = _FakeGateway(_split_all_responses())
    tools = AnalysisTools(fake)

    result = await tools.split_all_dimensions(
        {"device_type": "roku"}, "rebuffer", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result, result
    names = [d["dimension"] for d in result["dimensions"]]
    assert "device_type" not in names  # already pinned -- excluded from candidates
    assert names[0] == "app_version"
    top = result["dimensions"][0]
    assert top["top_value"] == "8.2.0"
    assert top["lift"] is not None and top["lift"] > 1.5
    assert result["sql"] and result["baseline_sql"]


async def test_split_all_dimensions_reports_dimensions_with_no_rows_as_no_data():
    tools = AnalysisTools(_FakeGateway(_split_all_responses()))

    result = await tools.split_all_dimensions({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    no_data = next(d for d in result["dimensions"] if d["dimension"] == "cdn")
    assert no_data["top_value"] is None
    assert no_data["lift"] is None
    assert "no data" in no_data["note"]
    # None-lift dimensions must not be ranked above the real signal.
    assert result["dimensions"][0]["dimension"] == "app_version"


async def test_split_all_dimensions_includes_title_id_as_a_candidate():
    """DEFECT 1: title_id used to be structurally excluded from split_all_dimensions,
    making the tool blind to per-title faults (INC-ENCODE-1). It must now appear
    alongside every other candidate dimension, cast to a string so it unions cleanly
    with the other (string-valued) arms."""
    tools = AnalysisTools(_FakeGateway(_split_all_responses()))

    result = await tools.split_all_dimensions({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert "title_id" in {d["dimension"] for d in result["dimensions"]}
    assert "toString(title_id)" in result["sql"]


async def test_split_all_dimensions_excludes_title_id_once_pinned():
    tools = AnalysisTools(_FakeGateway(_split_all_responses()))

    result = await tools.split_all_dimensions(
        {"title_id": "1"}, "rebuffer", _WINDOW_START, _WINDOW_END
    )

    assert "title_id" not in {d["dimension"] for d in result["dimensions"]}


async def test_split_all_dimensions_issues_exactly_one_query_pair_for_every_dimension():
    """The whole point of DEFECT 3: one batched window query and one batched
    baseline query per lookback week, never one pair per dimension."""
    fake = _FakeGateway(_split_all_responses())
    tools = AnalysisTools(fake)

    await tools.split_all_dimensions({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert len(fake.queries) == 1 + DEFAULT_LOOKBACK_WEEKS


async def test_split_all_dimensions_returns_empty_list_when_every_dimension_is_pinned():
    slice_json = {d: "x" for d in DIMENSION_HIERARCHY} | {"title_id": "1"}
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)

    result = await tools.split_all_dimensions(slice_json, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["dimensions"] == []
    assert "note" in result
    assert fake.queries == []


async def test_split_all_dimensions_unknown_metric_is_invalid_input_before_any_query():
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)

    result = await tools.split_all_dimensions({}, "throughput", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "invalid_input"
    assert fake.queries == []


async def test_split_all_dimensions_reports_infrastructure_failure():
    tools = AnalysisTools(_FakeGateway([QueryError("connection reset")]))

    result = await tools.split_all_dimensions({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "infrastructure_failure"


def _proportional_subset_trap_responses() -> list[QueryResult]:
    """Two dimensions, EQUAL lift (4.0), wildly different share_of_deviation:

    `device_type=roku` explains 96% of its own split's deviation (broad, the true
    blast radius). `os_version=roku_os_14.0` is a proportional THIRD of roku's own
    weight_share carrying the identical per-unit deviation ratio (delta=0.04, same
    as roku) -- lift is scale-invariant so it comes out to the exact same 4.0, but
    it only explains 32% of its own split's deviation. Ranking by lift alone cannot
    tell these apart; ranking by share_of_deviation (gated on lift) must prefer the
    broader `device_type` value. Every other candidate dimension carries no rows at
    all, exactly like `_split_all_responses`.
    """
    # os_version's "everything else" is split across FOUR separate sibling values
    # (not one), each individually smaller in raw contribution than roku_os_14.0's
    # own -- exactly like the real dataset, where several other OS versions and
    # device types share the remaining deviation. A single oversized filler value
    # would itself out-contribute roku_os_14.0 and become the (wrong) top value.
    window_rows = [
        {"dim": "device_type", "value": "roku", "metric_value": 0.05, "weight": 2400.0},
        {
            "dim": "device_type",
            "value": "other_dt",
            "metric_value": 0.0015263157894736855,
            "weight": 7600.0,
        },
        {"dim": "os_version", "value": "roku_os_14.0", "metric_value": 0.05, "weight": 800.0},
        *(
            {
                "dim": "os_version",
                "value": f"other_os_{i}",
                "metric_value": 0.008391304347826086,
                "weight": 2300.0,
            }
            for i in range(4)
        ),
    ]
    baseline_rows = [
        {"dim": "device_type", "value": "roku", "metric_value": 0.01, "weight": 2400.0},
        {"dim": "device_type", "value": "other_dt", "metric_value": 0.001, "weight": 7600.0},
        {"dim": "os_version", "value": "roku_os_14.0", "metric_value": 0.01, "weight": 800.0},
        *(
            {"dim": "os_version", "value": f"other_os_{i}", "metric_value": 0.001, "weight": 2300.0}
            for i in range(4)
        ),
    ]
    return [_qr(window_rows)] + [_qr(baseline_rows) for _ in range(4)]


async def test_split_all_dimensions_prefers_broader_value_over_equal_lift_subset_trap():
    """THE TRAP the ranking fix exists to reject: a proportional subset of the true
    blast radius has the SAME lift as the broader true value (lift is scale-
    invariant) but a much SMALLER share_of_deviation. Ranking on lift alone -- the
    pre-fix behaviour -- picks the subset and understates the incident."""
    tools = AnalysisTools(_FakeGateway(_proportional_subset_trap_responses()))

    result = await tools.split_all_dimensions({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert "error" not in result, result
    device_type = next(d for d in result["dimensions"] if d["dimension"] == "device_type")
    os_version = next(d for d in result["dimensions"] if d["dimension"] == "os_version")

    assert device_type["meets_lift_gate"] is True
    assert os_version["meets_lift_gate"] is True
    # Equal by construction -- lift is scale-invariant.
    assert device_type["lift"] == pytest.approx(os_version["lift"], rel=1e-6)
    assert device_type["share_of_deviation"] > os_version["share_of_deviation"]

    # The broader value (device_type=roku) must rank ABOVE the narrower, equally-
    # concentrated subset (os_version=roku_os_14.0) despite their equal lift.
    assert result["dimensions"][0]["dimension"] == "device_type"
    names_in_order = [d["dimension"] for d in result["dimensions"]]
    assert names_in_order.index("device_type") < names_in_order.index("os_version")


async def test_split_all_dimensions_ranks_by_share_not_by_lift_among_qualifying_candidates():
    """Two dimensions both clear the lift gate; the one with the LOWER lift but
    HIGHER share_of_deviation must rank first -- proving share_of_deviation, not
    lift, is the primary sort key once the gate has already been cleared."""
    # cdn's "everything else" is split across FIVE separate sibling values, each
    # individually smaller in raw contribution than cdn_northwind's own -- see the
    # comment on `_proportional_subset_trap_responses` for why a single oversized
    # filler would wrongly become the top value instead.
    window_rows = [
        # device_type: lift 2.0, share 0.8 -- lower lift, higher share.
        {"dim": "device_type", "value": "roku", "metric_value": 0.05, "weight": 4000.0},
        {
            "dim": "device_type",
            "value": "other_dt",
            "metric_value": 0.007666666666666667,
            "weight": 6000.0,
        },
        # cdn: lift 5.0, share 0.2 -- higher lift, lower share.
        {"dim": "cdn", "value": "cdn_northwind", "metric_value": 0.06, "weight": 400.0},
        *(
            {
                "dim": "cdn",
                "value": f"other_cdn_{i}",
                "metric_value": 0.009333333333333333,
                "weight": 1920.0,
            }
            for i in range(5)
        ),
    ]
    baseline_rows = [
        {"dim": "device_type", "value": "roku", "metric_value": 0.01, "weight": 4000.0},
        {"dim": "device_type", "value": "other_dt", "metric_value": 0.001, "weight": 6000.0},
        {"dim": "cdn", "value": "cdn_northwind", "metric_value": 0.01, "weight": 400.0},
        *(
            {"dim": "cdn", "value": f"other_cdn_{i}", "metric_value": 0.001, "weight": 1920.0}
            for i in range(5)
        ),
    ]
    responses = [_qr(window_rows)] + [_qr(baseline_rows) for _ in range(4)]
    tools = AnalysisTools(_FakeGateway(responses))

    result = await tools.split_all_dimensions({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert "error" not in result, result
    device_type = next(d for d in result["dimensions"] if d["dimension"] == "device_type")
    cdn = next(d for d in result["dimensions"] if d["dimension"] == "cdn")

    assert device_type["lift"] == pytest.approx(2.0, rel=1e-3)
    assert cdn["lift"] == pytest.approx(5.0, rel=1e-3)
    assert device_type["lift"] < cdn["lift"]  # device_type has the LOWER lift ...
    assert device_type["share_of_deviation"] > cdn["share_of_deviation"]  # ... but higher share
    assert result["dimensions"][0]["dimension"] == "device_type"  # ... and must rank first


# ---------------------------------------------------------------------------
# refine_incident_span: DEFECT 1 -- re-detect directly on the slice to find its
# true onset/end, never silently returning something worse than the input span.
# ---------------------------------------------------------------------------


async def test_refine_incident_span_returns_input_span_unchanged_when_nothing_found():
    """A thin slice (or a genuinely quiet one): re-detection finds no window, so
    the tool must say so explicitly and hand back the input span verbatim."""
    tools = AnalysisTools(_FakeGateway([_qr([])]))  # zero observations -> no windows

    result = await tools.refine_incident_span(
        {"device_type": "roku"}, "rebuffer", _WINDOW_START, _WINDOW_END
    )

    assert "error" not in result, result
    assert result["refined"] is False
    assert result["start"] == result["input_start"]
    assert result["end"] == result["input_end"]
    assert result["buckets_breached"] == 0
    assert result["typical_severity_ratio"] is None
    assert result["peak_severity_ratio"] is None
    assert result["note"]


async def test_refine_incident_span_unknown_metric_is_invalid_input_before_any_query():
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)

    result = await tools.refine_incident_span({}, "throughput", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "invalid_input"
    assert fake.queries == []


async def test_refine_incident_span_window_end_before_start_is_invalid_input():
    fake = _FakeGateway([])
    tools = AnalysisTools(fake)

    result = await tools.refine_incident_span({}, "rebuffer", _WINDOW_END, _WINDOW_START)

    assert result["error_type"] == "invalid_input"
    assert fake.queries == []


async def test_refine_incident_span_reports_infrastructure_failure():
    tools = AnalysisTools(_FakeGateway([QueryError("connection reset")]))

    result = await tools.refine_incident_span({}, "rebuffer", _WINDOW_START, _WINDOW_END)

    assert result["error_type"] == "infrastructure_failure"


# ---------------------------------------------------------------------------
# ADK wiring: FunctionTool construction is static (no model call) and the
# gateway never leaks into the schema the model would see.
# ---------------------------------------------------------------------------


def test_build_function_tools_returns_seven_tools_named_after_the_primitives():
    fake = _FakeGateway([])
    tools = build_function_tools(fake)

    names = {tool.name for tool in tools}
    assert names == {
        "detect_anomalies",
        "measure_slice",
        "split_on_dimension",
        "split_all_dimensions",
        "refine_incident_span",
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


def _encode_1_window() -> tuple[str, str]:
    payload = json.loads(_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    incident = next(inc for inc in payload["incidents"] if inc["kind"] == "encode_fault")
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


@pytest.mark.integration
async def test_split_all_dimensions_finds_the_real_incidents_top_dimension_in_one_call(gateway):
    """DEFECT 3: one call must surface the same signal 8 separate split_on_dimension
    calls were needed for before."""
    window_start, window_end = _roku_820_window()
    tools = AnalysisTools(gateway)

    result = await tools.split_all_dimensions({}, "rebuffer", window_start, window_end)

    assert "error" not in result, result
    top = result["dimensions"][0]
    assert top["dimension"] in ("device_type", "app_version"), (
        f"expected device_type or app_version to rank first by lift, got "
        f"{[(d['dimension'], d['lift']) for d in result['dimensions']]}"
    )
    assert top["lift"] is not None and top["lift"] > 1.5
    assert result["sql"] and result["baseline_sql"]


@pytest.mark.integration
async def test_split_all_dimensions_finds_title_id_for_the_real_encode_fault(gateway):
    """DEFECT 1: INC-ENCODE-1 is a title-scoped fault (true blast radius
    {title_id: 1}). Before the fix, split_all_dimensions excluded title_id entirely,
    leaving the agent structurally blind to it. It must now rank title_id=1 at the
    top of the batched split for this incident's real window."""
    window_start, window_end = _encode_1_window()
    tools = AnalysisTools(gateway)

    result = await tools.split_all_dimensions({}, "bitrate", window_start, window_end)

    assert "error" not in result, result
    assert result["dimensions"], "expected at least one candidate dimension"
    top = result["dimensions"][0]
    assert top["dimension"] == "title_id", (
        f"expected title_id to rank first by lift/share, got "
        f"{[(d['dimension'], d['lift'], d['share_of_deviation']) for d in result['dimensions']]}"
    )
    assert top["top_value"] == "1"
    assert top["lift"] is not None and top["lift"] > 1.5
    assert "toString(title_id)" in result["sql"]


@pytest.mark.integration
async def test_refine_incident_span_recovers_a_multi_hour_span_from_a_narrow_fragment(gateway):
    """DEFECT 1: handed only a 30-minute fragment of the real ~8-hour incident (what a
    diluted population-level detector would see at its worst peak), refining directly
    on the isolated slice must recover something close to the true span, not just
    re-confirm the fragment it was handed."""
    window_start_str, _window_end_str = _roku_820_window()
    fragment_start = datetime.fromisoformat(window_start_str)
    fragment_end = (fragment_start + timedelta(minutes=30)).isoformat()
    tools = AnalysisTools(gateway)

    result = await tools.refine_incident_span(
        {"device_type": "roku", "app_version": "8.2.0"}, "rebuffer", window_start_str, fragment_end
    )

    assert "error" not in result, result
    assert result["refined"] is True, result
    refined_start = datetime.fromisoformat(result["start"])
    refined_end = datetime.fromisoformat(result["end"])
    refined_hours = (refined_end - refined_start).total_seconds() / 3600
    assert refined_hours > 2, (
        f"expected refinement to recover a multi-hour span from a 30-minute fragment, "
        f"got {refined_hours:.2f}h ({result['start']} to {result['end']})"
    )
    assert result["buckets_breached"] > 0
    assert result["typical_severity_ratio"] is not None
    assert result["peak_severity_ratio"] is not None
    assert result["severity_sql"]


@pytest.mark.integration
async def test_quantify_impact_measures_its_own_severity_for_the_real_incident(gateway):
    """DEFECT 2: quantify_impact takes no severity parameter -- it must measure a
    positive severity ratio itself from the real incident's slice and window."""
    window_start, window_end = _roku_820_window()
    tools = AnalysisTools(gateway)

    result = await tools.quantify_impact(
        {"device_type": "roku", "app_version": "8.2.0"}, "rebuffer", window_start, window_end
    )

    assert "error" not in result, result
    assert result["typical_severity_ratio"] is not None and result["typical_severity_ratio"] > 0
    assert result["affected_subscribers"] > 0
    assert result["severity_sql"] and result["detect_sql"] and result["sql"]
