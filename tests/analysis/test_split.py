"""Unit tests for continuity.analysis.split -- pure maths plus a fake-gateway check
that batching many dimensions into one query yields identical results to querying them
one at a time.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from continuity.analysis.metrics import METRICS
from continuity.analysis.slices import Slice
from continuity.analysis.split import (
    Contribution,
    ValueMeasurement,
    is_informative,
    rank_contributions,
    split_dimension,
    split_dimensions,
)
from continuity.gateway.mcp_gateway import QueryResult

# ---------------------------------------------------------------------------
# THE MANDATORY TEST: a large moderately-degraded slice must outrank a tiny
# wildly-degraded slice by contribution, and a naive raw-deviation ranking must
# be shown to get this exactly backwards.
# ---------------------------------------------------------------------------


def _large_vs_tiny_measurements() -> list[ValueMeasurement]:
    # Watch-time weights sum to 1,000,000ms. "large" carries 60% of it and is 2x
    # baseline; "tiny" carries 0.5% and is 20x baseline; "rest" is unaffected.
    return [
        ValueMeasurement(value="large", metric_value=0.002, baseline_value=0.001, weight=600_000),
        ValueMeasurement(value="tiny", metric_value=0.020, baseline_value=0.001, weight=5_000),
        ValueMeasurement(value="rest", metric_value=0.001, baseline_value=0.001, weight=395_000),
    ]


def test_large_moderately_degraded_slice_ranks_first_by_contribution():
    measurements = _large_vs_tiny_measurements()
    ranked = rank_contributions(measurements, dimension="device_type", higher_is_worse=True)

    by_value = {c.value: c for c in ranked}
    assert ranked[0].value == "large"
    assert by_value["large"].contribution == pytest.approx(0.6 * (0.002 - 0.001))
    assert by_value["tiny"].contribution == pytest.approx(0.005 * (0.020 - 0.001))
    assert by_value["large"].contribution > by_value["tiny"].contribution

    # share_of_deviation: large explains the large majority of the parent's deviation.
    expected_share = 0.6 * 0.001 / (0.6 * 0.001 + 0.005 * 0.019)
    assert by_value["large"].share_of_deviation == pytest.approx(expected_share)
    assert by_value["large"].share_of_deviation > 0.85
    assert by_value["tiny"].share_of_deviation < 0.15


def test_naive_ranking_by_raw_deviation_would_wrongly_promote_the_tiny_slice():
    """Demonstrates the exact bug this module exists to avoid: ranking by the raw
    (unweighted) ratio deviation r_v - baseline promotes the tiny wild slice and
    buries the real, large-volume cause."""
    measurements = _large_vs_tiny_measurements()

    naive_ranking = sorted(
        measurements, key=lambda m: (m.metric_value - m.baseline_value), reverse=True
    )
    assert naive_ranking[0].value == "tiny"  # the bug: naive ranking gets this backwards

    correct_ranking = rank_contributions(
        measurements, dimension="device_type", higher_is_worse=True
    )
    assert correct_ranking[0].value == "large"  # the fix


# ---------------------------------------------------------------------------
# Direction awareness: a bitrate DROP is degradation and must contribute
# positively to the problem, not be filtered out as an improvement.
# ---------------------------------------------------------------------------


def test_bitrate_drop_is_ranked_as_a_positive_contribution_not_an_improvement():
    measurements = [
        ValueMeasurement(
            value="dropped", metric_value=3000.0, baseline_value=5000.0, weight=100.0
        ),
        ValueMeasurement(
            value="improved", metric_value=6000.0, baseline_value=5000.0, weight=100.0
        ),
    ]
    ranked = rank_contributions(measurements, dimension="cdn", higher_is_worse=False)
    by_value = {c.value: c for c in ranked}

    assert by_value["dropped"].contribution > 0
    assert by_value["improved"].contribution < 0
    assert ranked[0].value == "dropped"


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_single_value_dimension_reports_no_information_gained_not_a_meaningless_full_share():
    measurements = [
        ValueMeasurement(value="only", metric_value=0.01, baseline_value=0.001, weight=100.0),
    ]
    ranked = rank_contributions(measurements, dimension="device_type", higher_is_worse=True)

    assert ranked[0].share_of_deviation is None
    assert "no information" in ranked[0].note
    assert is_informative(measurements) is False


def test_zero_net_deviation_across_values_does_not_divide_by_zero():
    measurements = [
        ValueMeasurement(value="up", metric_value=12.0, baseline_value=10.0, weight=100.0),
        ValueMeasurement(value="down", metric_value=8.0, baseline_value=10.0, weight=100.0),
    ]
    ranked = rank_contributions(measurements, dimension="device_type", higher_is_worse=True)
    by_value = {c.value: c for c in ranked}

    assert by_value["up"].contribution == pytest.approx(1.0)
    assert by_value["down"].contribution == pytest.approx(-1.0)
    assert by_value["up"].share_of_deviation is None
    assert by_value["down"].share_of_deviation is None
    assert "zero net deviation" in by_value["up"].note
    # Contribution itself is still ranked correctly even though share is undefined.
    assert ranked[0].value == "up"


def test_value_absent_from_baseline_period_does_not_crash_and_is_not_dropped():
    """A new app version rolled out mid-incident: present in the window, absent from
    the baseline period. Realistic, and must surface rather than vanish."""
    measurements = [
        ValueMeasurement(value="8.2.0", metric_value=0.02, baseline_value=0.001, weight=100.0),
        ValueMeasurement(value="8.3.0-new", metric_value=0.05, baseline_value=None, weight=50.0),
    ]
    ranked = rank_contributions(measurements, dimension="app_version", higher_is_worse=True)
    by_value = {c.value: c for c in ranked}

    assert "8.3.0-new" in by_value  # not silently dropped
    assert by_value["8.3.0-new"].contribution is None
    assert by_value["8.3.0-new"].share_of_deviation is None
    assert "baseline period" in by_value["8.3.0-new"].note
    # The value WITH a baseline is still ranked and unaffected by the missing one.
    assert by_value["8.2.0"].contribution is not None


def test_null_metric_value_does_not_crash_and_is_not_dropped():
    measurements = [
        ValueMeasurement(value="has_data", metric_value=0.02, baseline_value=0.001, weight=100.0),
        ValueMeasurement(value="no_data", metric_value=None, baseline_value=0.001, weight=100.0),
    ]
    ranked = rank_contributions(measurements, dimension="isp", higher_is_worse=True)
    by_value = {c.value: c for c in ranked}

    assert "no_data" in by_value
    assert by_value["no_data"].contribution is None
    assert "no metric value" in by_value["no_data"].note


def test_empty_measurements_returns_empty_tuple_not_an_error():
    assert rank_contributions([], dimension="device_type", higher_is_worse=True) == ()


def test_all_zero_weight_does_not_divide_by_zero():
    measurements = [
        ValueMeasurement(value="a", metric_value=0.01, baseline_value=0.001, weight=0.0),
        ValueMeasurement(value="b", metric_value=0.02, baseline_value=0.001, weight=0.0),
    ]
    ranked = rank_contributions(measurements, dimension="device_type", higher_is_worse=True)
    assert all(c.contribution is None for c in ranked)
    assert all(c.weight_share is None for c in ranked)
    assert all("no weight" in c.note for c in ranked)


def test_contribution_carries_typed_fields_and_provenance_sql():
    measurements = [
        ValueMeasurement(value="roku", metric_value=0.02, baseline_value=0.001, weight=100.0),
    ]
    ranked = rank_contributions(
        measurements,
        dimension="device_type",
        higher_is_worse=True,
        sql="SELECT ...window...",
        baseline_sql="SELECT ...baseline...",
    )
    c = ranked[0]
    assert isinstance(c, Contribution)
    assert c.dimension == "device_type"
    assert c.value == "roku"
    assert c.metric_value == 0.02
    assert c.baseline_value == 0.001
    assert c.weight == 100.0
    assert c.sql == "SELECT ...window..."
    assert c.baseline_sql == "SELECT ...baseline..."


# ---------------------------------------------------------------------------
# Batching: split_dimensions must issue ONE query per window (not one per
# dimension) and must return results identical to calling split_dimension
# separately for each dimension.
# ---------------------------------------------------------------------------


class _FakeGateway:
    """Stands in for ClickHouseMCPGateway: returns canned rows keyed by which SQL
    template (window vs baseline, single-dimension vs batched UNION ALL) was asked."""

    def __init__(
        self,
        window_rows: dict[str, list[dict]],
        baseline_rows: dict[str, list[dict]],
        window: tuple[datetime, datetime],
        baseline_window: tuple[datetime, datetime],
    ) -> None:
        self._window_rows = window_rows
        self._baseline_rows = baseline_rows
        self._window_marker = window[0].strftime("%Y-%m-%d %H:%M:%S")
        self._baseline_marker = baseline_window[0].strftime("%Y-%m-%d %H:%M:%S")
        self.queries: list[str] = []

    async def query(self, sql: str) -> QueryResult:
        self.queries.append(sql)
        is_window = self._window_marker in sql
        is_baseline = self._baseline_marker in sql
        assert is_window != is_baseline, "ambiguous window/baseline marker in test fixture"
        source = self._window_rows if is_window else self._baseline_rows

        if "UNION ALL" in sql:
            rows_out = [{**row, "dim": dim} for dim, rows in source.items() for row in rows]
        else:
            (dim,) = [d for d in source if f"{d} AS value" in sql]
            rows_out = list(source[dim])
        columns = list(rows_out[0].keys()) if rows_out else []
        return QueryResult(sql=sql, columns=columns, rows=rows_out)


def _fixture_data() -> tuple[dict, dict, tuple[datetime, datetime], tuple[datetime, datetime]]:
    window = (datetime(2026, 1, 13, 18, 0, 0), datetime(2026, 1, 14, 2, 0, 0))
    baseline_window = (datetime(2026, 1, 6, 18, 0, 0), datetime(2026, 1, 7, 2, 0, 0))
    window_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.02, "weight": 100_000.0},
            {"value": "ios", "metric_value": 0.001, "weight": 200_000.0},
        ],
        "app_version": [
            {"value": "8.2.0", "metric_value": 0.03, "weight": 150_000.0},
            {"value": "8.1.4", "metric_value": 0.001, "weight": 150_000.0},
        ],
    }
    baseline_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.001, "weight": 90_000.0},
            {"value": "ios", "metric_value": 0.001, "weight": 180_000.0},
        ],
        "app_version": [
            {"value": "8.2.0", "metric_value": 0.001, "weight": 140_000.0},
            {"value": "8.1.4", "metric_value": 0.001, "weight": 140_000.0},
        ],
    }
    return window_rows, baseline_rows, window, baseline_window


async def test_batched_split_issues_exactly_two_queries_regardless_of_dimension_count():
    window_rows, baseline_rows, window, baseline_window = _fixture_data()
    fake = _FakeGateway(window_rows, baseline_rows, window, baseline_window)

    await split_dimensions(
        fake,
        slice_=Slice(),
        metric=METRICS["rebuffer"],
        dimensions=["device_type", "app_version"],
        window=window,
        baseline_window=baseline_window,
    )

    assert len(fake.queries) == 2


async def test_batched_split_matches_result_of_separate_single_dimension_splits():
    window_rows, baseline_rows, window, baseline_window = _fixture_data()

    batched_gateway = _FakeGateway(window_rows, baseline_rows, window, baseline_window)
    batched = await split_dimensions(
        batched_gateway,
        slice_=Slice(),
        metric=METRICS["rebuffer"],
        dimensions=["device_type", "app_version"],
        window=window,
        baseline_window=baseline_window,
    )

    separate_gateway = _FakeGateway(window_rows, baseline_rows, window, baseline_window)
    separate = {}
    for dimension in ("device_type", "app_version"):
        separate[dimension] = await split_dimension(
            separate_gateway,
            slice_=Slice(),
            metric=METRICS["rebuffer"],
            dimension=dimension,
            window=window,
            baseline_window=baseline_window,
        )

    assert len(separate_gateway.queries) == 4  # one window + one baseline query PER dimension
    for dimension in ("device_type", "app_version"):
        # Compare field-by-field rather than the whole SQL string, since the batched
        # arm's SQL differs slightly (it carries a "dim" tag column) from the
        # single-dimension arm's -- the maths must agree even though the SQL text does not.
        def _key(c):
            return (
                c.value,
                c.metric_value,
                c.baseline_value,
                c.weight,
                c.contribution,
                c.share_of_deviation,
            )

        batched_values = [_key(c) for c in batched[dimension].values]
        separate_values = [_key(c) for c in separate[dimension].values]
        assert batched_values == separate_values
        assert batched[dimension].informative == separate[dimension].informative

    assert batched["device_type"].values[0].value == "roku"
    assert batched["app_version"].values[0].value == "8.2.0"


async def test_split_dimension_rejects_unknown_dimension():
    from continuity.analysis.slices import InvalidSliceError

    window_rows, baseline_rows, window, baseline_window = _fixture_data()
    fake = _FakeGateway(window_rows, baseline_rows, window, baseline_window)
    with pytest.raises(InvalidSliceError, match="bogus_dim"):
        await split_dimension(
            fake,
            slice_=Slice(),
            metric=METRICS["rebuffer"],
            dimension="bogus_dim",
            window=window,
            baseline_window=baseline_window,
        )


async def test_split_dimensions_rejects_empty_dimension_list():
    window_rows, baseline_rows, window, baseline_window = _fixture_data()
    fake = _FakeGateway(window_rows, baseline_rows, window, baseline_window)
    with pytest.raises(ValueError, match="non-empty"):
        await split_dimensions(
            fake,
            slice_=Slice(),
            metric=METRICS["rebuffer"],
            dimensions=[],
            window=window,
            baseline_window=baseline_window,
        )


async def test_split_dimension_rejects_inverted_window():
    window_rows, baseline_rows, _window, baseline_window = _fixture_data()
    inverted_window = (datetime(2026, 1, 14), datetime(2026, 1, 13))
    fake = _FakeGateway(window_rows, baseline_rows, inverted_window, baseline_window)
    with pytest.raises(ValueError, match="window"):
        await split_dimension(
            fake,
            slice_=Slice(),
            metric=METRICS["rebuffer"],
            dimension="device_type",
            window=inverted_window,
            baseline_window=baseline_window,
        )
