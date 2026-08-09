"""Unit tests for continuity/analysis/walk.py.

Two layers, matching the module's own split between pure maths and I/O:

* `choose_next_step`, `_population_weight`, `week_over_week_baseline_windows` are pure
  -- exercised directly with hand-built `SplitResult`/`Contribution` objects, no
  gateway involved.
* `walk()` itself is exercised end-to-end against a fake gateway that returns canned
  rows keyed by (predicate, window) -- mirrors tests/analysis/test_split.py's
  `_FakeGateway`, generalised to support a refined slice's predicate and several
  baseline windows.

The median-of-several-baseline-windows robustness requirement (Task 8's carried-
forward note) is tested directly against
`continuity.analysis.split.split_dimensions_median_baseline`, since that is the new
capability added to split.py for this task.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from continuity.analysis.metrics import METRICS
from continuity.analysis.slices import Slice
from continuity.analysis.split import (
    ValueMeasurement,
    is_informative,
    rank_contributions,
    split_dimensions_median_baseline,
)
from continuity.analysis.walk import (
    DEFAULT_MAX_DEPTH,
    RefinementStep,
    StopReason,
    WalkResult,
    _population_weight,
    choose_next_step,
    walk,
    week_over_week_baseline_windows,
)
from continuity.gateway.mcp_gateway import ExecutedQuery, QueryResult

# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def _split(dimension: str, measurements: list[ValueMeasurement], *, higher_is_worse: bool = True):
    from continuity.analysis.split import SplitResult

    values = rank_contributions(measurements, dimension=dimension, higher_is_worse=higher_is_worse)
    return SplitResult(
        dimension=dimension,
        values=values,
        informative=is_informative(measurements),
        sql=f"SELECT ... {dimension} ...",
        baseline_sql=f"SELECT ... {dimension} baseline ...",
    )


def test_week_over_week_baseline_windows_matches_baseline_pys_own_convention():
    window = (datetime(2026, 2, 12, 18, 0, 0), datetime(2026, 2, 13, 2, 0, 0))
    windows = week_over_week_baseline_windows(window, lookback_weeks=4)

    assert windows == (
        (datetime(2026, 2, 5, 18, 0, 0), datetime(2026, 2, 6, 2, 0, 0)),
        (datetime(2026, 1, 29, 18, 0, 0), datetime(2026, 1, 30, 2, 0, 0)),
        (datetime(2026, 1, 22, 18, 0, 0), datetime(2026, 1, 23, 2, 0, 0)),
        (datetime(2026, 1, 15, 18, 0, 0), datetime(2026, 1, 16, 2, 0, 0)),
    )


def test_week_over_week_baseline_windows_rejects_non_positive_lookback():
    window = (datetime(2026, 2, 12), datetime(2026, 2, 13))
    with pytest.raises(ValueError, match="lookback_weeks"):
        week_over_week_baseline_windows(window, lookback_weeks=0)


def test_week_over_week_baseline_windows_rejects_inverted_window():
    with pytest.raises(ValueError, match="window"):
        week_over_week_baseline_windows((datetime(2026, 2, 13), datetime(2026, 2, 12)), 4)


def test_population_weight_takes_the_largest_split_total():
    splits = {
        "device_type": _split(
            "device_type",
            [
                ValueMeasurement(
                    value="roku", metric_value=0.02, baseline_value=0.001, weight=100.0
                ),
                ValueMeasurement(
                    value="ios", metric_value=0.001, baseline_value=0.001, weight=200.0
                ),
            ],
        ),
        "app_version": _split(
            "app_version",
            [
                ValueMeasurement(
                    value="8.2.0", metric_value=0.01, baseline_value=0.001, weight=250.0
                ),
            ],
        ),
    }
    assert _population_weight(splits) == pytest.approx(300.0)


def test_population_weight_is_none_when_every_split_is_empty():
    splits = {"device_type": _split("device_type", [])}
    assert _population_weight(splits) is None


# ---------------------------------------------------------------------------
# choose_next_step: the pure per-level decision, and every stopping rule.
#
# Selection among dimensions is still by raw share (each dimension's top contributor)
# -- LIFT is the gate on whether that winner is worth descending into, not the
# selection criterion. See walk.py's DEFAULT_MIN_LIFT comment for the reasoning behind
# the 1.5 default used throughout this section.
# ---------------------------------------------------------------------------


def test_choose_next_step_picks_the_dimension_with_the_highest_share():
    # device_type: one dominant deviator -> share 1.0, weight_share 0.25 -> lift 4.0.
    device = _split(
        "device_type",
        [
            ValueMeasurement(
                value="roku", metric_value=0.02, baseline_value=0.001, weight=100_000.0
            ),
            ValueMeasurement(
                value="ios", metric_value=0.001, baseline_value=0.001, weight=300_000.0
            ),
        ],
    )
    # app_version: a weaker, partly-offset signal -> share well under 1.0.
    app = _split(
        "app_version",
        [
            ValueMeasurement(
                value="8.2.0", metric_value=0.003, baseline_value=0.001, weight=250_000.0
            ),
            ValueMeasurement(
                value="8.1.4", metric_value=0.006, baseline_value=0.001, weight=250_000.0
            ),
        ],
    )
    splits = {"device_type": device, "app_version": app}

    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.0, root_weight=None
    )

    assert reason is None
    assert detail is None
    assert isinstance(step, RefinementStep)
    assert step.dimension == "device_type"
    assert step.value == "roku"
    assert step.share_of_deviation == pytest.approx(1.0)
    assert step.lift == pytest.approx(4.0)


def test_choose_next_step_returns_single_value_when_no_dimension_is_informative():
    splits = {
        "device_type": _split(
            "device_type",
            [ValueMeasurement(value="roku", metric_value=0.02, baseline_value=0.001, weight=100.0)],
        )
    }
    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.0, root_weight=None
    )
    assert step is None
    assert reason is StopReason.SINGLE_VALUE
    assert detail is not None


def test_choose_next_step_returns_single_value_for_an_empty_splits_dict():
    step, reason, detail = choose_next_step(
        {}, min_lift=1.5, min_weight_fraction=0.0, root_weight=None
    )
    assert step is None
    assert reason is StopReason.SINGLE_VALUE
    assert detail is not None


def test_choose_next_step_returns_low_share_when_net_deviation_is_zero():
    """Both values move, but they cancel -- rank_contributions leaves share_of_deviation
    undefined (None) for exactly this reason (see test_split.py's own zero-net test).
    There is nothing to compute a lift FROM, so this is still LOW_SHARE, not LOW_LIFT."""
    splits = {
        "device_type": _split(
            "device_type",
            [
                ValueMeasurement(value="up", metric_value=12.0, baseline_value=10.0, weight=100.0),
                ValueMeasurement(value="down", metric_value=8.0, baseline_value=10.0, weight=100.0),
            ],
        )
    }
    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.0, root_weight=None
    )
    assert step is None
    assert reason is StopReason.LOW_SHARE
    assert detail is not None


def test_choose_next_step_stops_on_uniform_degradation_not_the_biggest_segment():
    """THE TRAP. Both values degrade by the exact same amount -- a uniformly bad
    dimension, not a localized fault. share_of_deviation then collapses to plain
    weight_share for every value (roku: share 0.7 == weight_share 0.7; ios: share 0.3
    == weight_share 0.3), so a raw-share threshold would happily chase "roku" as if it
    were the biggest, most-explaining segment -- it is merely the biggest segment.
    lift == 1.0 for BOTH values catches this exactly: descending would explain no more
    than population size alone predicts. Do not "fix" this by lowering min_lift below
    1.0 -- that would defeat the entire point of this test.
    """
    uniform = _split(
        "device_type",
        [
            ValueMeasurement(
                value="roku", metric_value=0.002, baseline_value=0.001, weight=700_000.0
            ),
            ValueMeasurement(
                value="ios", metric_value=0.002, baseline_value=0.001, weight=300_000.0
            ),
        ],
    )
    splits = {"device_type": uniform}

    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.0, root_weight=None
    )

    assert step is None
    assert reason is StopReason.LOW_LIFT
    assert "device_type='roku'" in detail
    assert "1.00" in detail  # the measured lift, not just the label


def test_choose_next_step_returns_low_lift_when_best_lift_is_below_threshold():
    splits = {
        "app_version": _split(
            "app_version",
            [
                ValueMeasurement(
                    value="8.2.0", metric_value=0.003, baseline_value=0.001, weight=250_000.0
                ),
                ValueMeasurement(
                    value="8.1.4", metric_value=0.006, baseline_value=0.001, weight=250_000.0
                ),
            ],
        )
    }
    # share_8.2.0 ~= 0.2857, weight_share_8.2.0 = 0.5 -> lift ~= 0.571, well under 1.5.
    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.0, root_weight=None
    )
    assert step is None
    assert reason is StopReason.LOW_LIFT
    assert "lift" in detail


def test_choose_next_step_returns_too_small_when_candidate_weight_is_a_sliver_of_root():
    splits = {
        "device_type": _split(
            "device_type",
            [
                ValueMeasurement(value="roku", metric_value=0.5, baseline_value=0.001, weight=50.0),
                ValueMeasurement(
                    value="ios", metric_value=0.001, baseline_value=0.001, weight=300_000.0
                ),
            ],
        )
    }
    # roku's weight_share is tiny (50 / 300_050), so its lift is enormous -- the lift
    # gate passes easily, and the too-small guard is what actually fires.
    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.01, root_weight=300_050.0
    )
    assert step is None
    assert reason is StopReason.TOO_SMALL
    assert detail is not None


def test_choose_next_step_accepts_when_lift_and_weight_both_clear_their_thresholds():
    splits = {
        "device_type": _split(
            "device_type",
            [
                ValueMeasurement(
                    value="roku", metric_value=0.02, baseline_value=0.001, weight=100_000.0
                ),
                ValueMeasurement(
                    value="ios", metric_value=0.001, baseline_value=0.001, weight=300_000.0
                ),
            ],
        )
    }
    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.01, root_weight=400_000.0
    )
    assert reason is None
    assert detail is None
    assert step is not None
    assert step.value == "roku"
    assert step.lift == pytest.approx(4.0)


def test_choose_next_step_ignores_root_weight_guard_when_root_weight_is_none():
    """root_weight is None at the very first level before any split has run; the
    too-small guard must not divide by / compare against a missing baseline."""
    splits = {
        "device_type": _split(
            "device_type",
            [
                ValueMeasurement(value="roku", metric_value=0.02, baseline_value=0.001, weight=1.0),
                ValueMeasurement(value="ios", metric_value=0.001, baseline_value=0.001, weight=1.0),
            ],
        )
    }
    step, reason, detail = choose_next_step(
        splits, min_lift=1.5, min_weight_fraction=0.5, root_weight=None
    )
    assert reason is None
    assert detail is None
    assert step is not None


def _moderate_share_high_lift_split():
    """Three values sharing the deviation (0.4 / 0.35 / 0.25) so the top-ranked
    value's raw share is a moderate 0.4 -- not close to 1.0 -- while its weight_share
    is small (0.1), giving it a high lift (4.0). Isolates "moderate share, high lift"
    from the min_share guard tests below, independent of the lift gate."""
    return _split(
        "device_type",
        [
            # weight_share 0.1, delta 4.0    -> contribution 0.4 -> share 0.40, lift 4.0
            ValueMeasurement(
                value="roku", metric_value=4.001, baseline_value=0.001, weight=100_000.0
            ),
            # weight_share 0.5, delta 0.7    -> contribution 0.35 -> share 0.35, lift 0.7
            ValueMeasurement(
                value="firetv", metric_value=0.701, baseline_value=0.001, weight=500_000.0
            ),
            # weight_share 0.4, delta 0.625  -> contribution 0.25 -> share 0.25, lift 0.625
            ValueMeasurement(
                value="ios", metric_value=0.626, baseline_value=0.001, weight=400_000.0
            ),
        ],
    )


def test_choose_next_step_optional_min_share_guard_is_secondary_to_lift():
    """min_share, when supplied, is checked only AFTER the lift gate has already
    passed -- it must never be the reason a step with a low lift is accepted, and it
    must not fire ahead of a lift failure either."""
    splits = {"device_type": _moderate_share_high_lift_split()}

    # roku's lift is 4.0 (clears min_lift=1.0 easily) but its raw share is only 0.4 --
    # an explicit min_share above that rejects it even though lift alone would accept.
    step, reason, detail = choose_next_step(
        splits, min_lift=1.0, min_share=0.5, min_weight_fraction=0.0, root_weight=None
    )
    assert step is None
    assert reason is StopReason.LOW_SHARE
    assert "min_share" in detail


def test_choose_next_step_min_share_defaults_to_off():
    """Calling without min_share at all must not silently apply one -- its default
    must not be load-bearing. Same fixture as the test above; the only difference is
    that min_share is omitted here, and the step that guard rejected there is accepted."""
    splits = {"device_type": _moderate_share_high_lift_split()}

    step, reason, detail = choose_next_step(
        splits, min_lift=1.0, min_weight_fraction=0.0, root_weight=None
    )
    assert reason is None
    assert detail is None
    assert step is not None
    assert step.value == "roku"
    assert step.share_of_deviation == pytest.approx(0.4)
    assert step.lift == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# split.py's new capability: median of several baseline windows survives one
# corrupted window (Task 8's carried-forward robustness requirement).
# ---------------------------------------------------------------------------


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class _MultiWindowFakeGateway:
    """Canned rows keyed by the literal window-start timestamp embedded in the SQL --
    same technique as test_split.py's `_FakeGateway`, generalised to more than two
    windows (one test window plus several baseline windows)."""

    def __init__(self, rows_by_window_start: dict[str, dict[str, list[dict]]]) -> None:
        self._rows_by_window_start = rows_by_window_start
        self.query_log: list[ExecutedQuery] = []

    async def query(self, sql: str) -> QueryResult:
        matches = [marker for marker in self._rows_by_window_start if marker in sql]
        assert len(matches) == 1, f"ambiguous/missing window marker in: {sql}"
        source = self._rows_by_window_start[matches[0]]
        # The batched form (split_dimensions / split_dimensions_median_baseline) tags
        # every arm with "'{dim}' AS dim, " even when only one dimension is requested
        # -- so "AS dim" is the reliable marker, not "UNION ALL" (which is absent for a
        # single-dimension batch).
        if "AS dim" in sql:
            rows_out = [{**row, "dim": dim} for dim, rows in source.items() for row in rows]
        else:
            (dim,) = [d for d in source if f"{d} AS value" in sql]
            rows_out = list(source[dim])
        columns = list(rows_out[0].keys()) if rows_out else []
        self.query_log.append(ExecutedQuery(sql=sql, duration_ms=0.0, row_count=len(rows_out)))
        return QueryResult(sql=sql, columns=columns, rows=rows_out)


async def test_median_of_baseline_windows_survives_one_corrupted_window():
    """One of the four trailing comparison windows happens to contain an incident that
    spiked roku's OWN baseline reading -- a single-window baseline that unluckily
    picked that window would rank ios first and show roku as IMPROVING. The median
    across all four windows outvotes the corrupted one and keeps the ranking correct.
    """
    window = (datetime(2026, 3, 2, 18, 0, 0), datetime(2026, 3, 3, 2, 0, 0))
    baseline_windows = tuple(
        (window[0] - timedelta(weeks=i), window[1] - timedelta(weeks=i)) for i in range(1, 5)
    )

    window_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.02, "weight": 100_000.0},
            {"value": "ios", "metric_value": 0.0015, "weight": 200_000.0},
        ],
    }
    clean_baseline_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.001, "weight": 95_000.0},
            {"value": "ios", "metric_value": 0.001, "weight": 190_000.0},
        ],
    }
    # A pre-existing incident sat inside baseline_windows[1] and pushed roku's *baseline*
    # reading up to 0.05 -- nothing to do with the incident under investigation.
    corrupted_baseline_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.05, "weight": 95_000.0},
            {"value": "ios", "metric_value": 0.001, "weight": 190_000.0},
        ],
    }

    rows_by_marker = {
        _fmt(window[0]): window_rows,
        _fmt(baseline_windows[0][0]): clean_baseline_rows,
        _fmt(baseline_windows[1][0]): corrupted_baseline_rows,
        _fmt(baseline_windows[2][0]): clean_baseline_rows,
        _fmt(baseline_windows[3][0]): clean_baseline_rows,
    }
    fake = _MultiWindowFakeGateway(rows_by_marker)

    result = await split_dimensions_median_baseline(
        fake,
        slice_=Slice(),
        metric=METRICS["rebuffer"],
        dimensions=["device_type"],
        window=window,
        baseline_windows=baseline_windows,
    )

    top = result["device_type"].values[0]
    assert top.value == "roku"
    # median([0.001, 0.05, 0.001, 0.001]) == 0.001 -- the corrupted window is outvoted,
    # not blended in (a mean would drag this to ~0.013 and shrink roku's measured
    # deviation by roughly 4x).
    assert top.baseline_value == pytest.approx(0.001)
    assert top.contribution is not None and top.contribution > 0

    # Contrast: a single-baseline-window split that unluckily landed on the corrupted
    # window gets this exactly backwards -- proving the median fix is load-bearing, not
    # cosmetic.
    from continuity.analysis.split import split_dimension

    fragile = await split_dimension(
        fake,
        slice_=Slice(),
        metric=METRICS["rebuffer"],
        dimension="device_type",
        window=window,
        baseline_window=baseline_windows[1],
    )
    fragile_by_value = {c.value: c for c in fragile.values}
    assert fragile_by_value["roku"].contribution < 0, (
        "sanity check: the corrupted single window should make roku look IMPROVED"
    )
    assert fragile.values[0].value == "ios", (
        "sanity check: the corrupted single window should wrongly rank ios first"
    )


# ---------------------------------------------------------------------------
# walk(): end-to-end over a fake gateway. Two dimensions only (device_type,
# app_version) so the scenario is small enough to hand-build, mirroring the real
# INC-APP-ROKU-820 shape (neither dimension alone is the point here -- the walk's
# level-by-level control flow is).
# ---------------------------------------------------------------------------


class _WalkFakeGateway:
    """Canned rows keyed by (predicate substring, window-start literal).

    `predicate` distinguishes the whole-population query ("WHERE 1 AND") from a
    refined slice's query ("WHERE device_type = 'roku' AND"); `window` distinguishes
    the test window from each of the baseline windows -- mirrors
    `_MultiWindowFakeGateway` above, extended with the predicate axis a multi-level
    walk needs.
    """

    def __init__(self, dataset: dict[str, dict[str, dict[str, list[dict]]]]) -> None:
        self._dataset = dataset
        self.query_log: list[ExecutedQuery] = []

    async def query(self, sql: str) -> QueryResult:
        predicate = max((p for p in self._dataset if p in sql), key=len, default=None)
        assert predicate is not None, f"no predicate marker matched: {sql}"
        by_window = self._dataset[predicate]
        window_marker = next((w for w in by_window if w in sql), None)
        assert window_marker is not None, f"no window marker matched: {sql}"
        source = by_window[window_marker]

        if "AS dim" in sql:
            rows_out = [{**row, "dim": dim} for dim, rows in source.items() for row in rows]
        else:
            (dim,) = [d for d in source if f"{d} AS value" in sql]
            rows_out = list(source[dim])
        columns = list(rows_out[0].keys()) if rows_out else []
        self.query_log.append(ExecutedQuery(sql=sql, duration_ms=1.0, row_count=len(rows_out)))
        return QueryResult(sql=sql, columns=columns, rows=rows_out)


def _roku_820_dataset(
    window: tuple[datetime, datetime], baseline_windows: tuple[tuple[datetime, datetime], ...]
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    root_window_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.02, "weight": 100_000.0},
            {"value": "ios", "metric_value": 0.001, "weight": 300_000.0},
            {"value": "firetv", "metric_value": 0.001, "weight": 200_000.0},
        ],
        "app_version": [
            {"value": "8.2.0", "metric_value": 0.003, "weight": 250_000.0},
            {"value": "8.1.4", "metric_value": 0.006, "weight": 250_000.0},
            {"value": "8.0.9", "metric_value": 0.001, "weight": 100_000.0},
        ],
    }
    root_baseline_rows = {
        "device_type": [
            {"value": "roku", "metric_value": 0.001, "weight": 95_000.0},
            {"value": "ios", "metric_value": 0.001, "weight": 295_000.0},
            {"value": "firetv", "metric_value": 0.001, "weight": 195_000.0},
        ],
        "app_version": [
            {"value": "8.2.0", "metric_value": 0.001, "weight": 245_000.0},
            {"value": "8.1.4", "metric_value": 0.001, "weight": 245_000.0},
            {"value": "8.0.9", "metric_value": 0.001, "weight": 95_000.0},
        ],
    }
    # weight_share_8.2.0 = 30_000 / 100_000 = 0.3, so lift = share(1.0) / 0.3 ~= 3.33 --
    # comfortably above DEFAULT_MIN_LIFT (1.5), unlike an 80% weight_share (lift 1.25)
    # which would (correctly) stop the walk one level too early for this test's purpose.
    roku_window_rows = {
        "app_version": [
            {"value": "8.2.0", "metric_value": 0.02, "weight": 30_000.0},
            {"value": "8.1.4", "metric_value": 0.001, "weight": 60_000.0},
            {"value": "8.0.9", "metric_value": 0.001, "weight": 10_000.0},
        ],
    }
    roku_baseline_rows = {
        "app_version": [
            {"value": "8.2.0", "metric_value": 0.001, "weight": 29_000.0},
            {"value": "8.1.4", "metric_value": 0.001, "weight": 58_000.0},
            {"value": "8.0.9", "metric_value": 0.001, "weight": 9_500.0},
        ],
    }

    dataset: dict[str, dict[str, dict[str, list[dict]]]] = {
        "WHERE 1 AND": {_fmt(window[0]): root_window_rows},
        "WHERE device_type = 'roku' AND": {_fmt(window[0]): roku_window_rows},
    }
    for bw in baseline_windows:
        dataset["WHERE 1 AND"][_fmt(bw[0])] = root_baseline_rows
        dataset["WHERE device_type = 'roku' AND"][_fmt(bw[0])] = roku_baseline_rows
    return dataset


async def test_walk_descends_device_type_then_app_version_and_exhausts_dimensions():
    window = (datetime(2026, 3, 2, 18, 0, 0), datetime(2026, 3, 3, 2, 0, 0))
    lookback_weeks = 4
    baseline_windows = tuple(
        (window[0] - timedelta(weeks=i), window[1] - timedelta(weeks=i))
        for i in range(1, lookback_weeks + 1)
    )
    fake = _WalkFakeGateway(_roku_820_dataset(window, baseline_windows))

    result = await walk(
        fake,
        metric_name="rebuffer",
        window=window,
        dimensions=["device_type", "app_version"],
        lookback_weeks=lookback_weeks,
        min_weight_fraction=0.0,
    )

    assert isinstance(result, WalkResult)
    assert [step.dimension for step in result.path] == ["device_type", "app_version"]
    assert [step.value for step in result.path] == ["roku", "8.2.0"]
    assert result.final_slice == Slice().refine("device_type", "roku").refine(
        "app_version", "8.2.0"
    )
    assert result.stop_reason is StopReason.DIMENSIONS_EXHAUSTED
    assert result.metric == "rebuffer"
    assert result.baseline_windows == baseline_windows
    assert result.elapsed_ms >= 0.0

    # Every step's lift is recorded and clears the default min_lift -- that is WHY the
    # walk was willing to descend at all.
    assert all(step.lift >= 1.5 for step in result.path)

    # Requirement 3: never re-split a dimension already fixed in the current slice.
    assert len({step.dimension for step in result.path}) == len(result.path)

    # Every level issues 1 window query + lookback_weeks baseline queries, all batched.
    assert len(result.query_log) == 2 * (1 + lookback_weeks)


async def test_walk_stops_at_max_depth_before_exhausting_dimensions():
    window = (datetime(2026, 3, 2, 18, 0, 0), datetime(2026, 3, 3, 2, 0, 0))
    baseline_windows = tuple(
        (window[0] - timedelta(weeks=i), window[1] - timedelta(weeks=i)) for i in range(1, 5)
    )
    fake = _WalkFakeGateway(_roku_820_dataset(window, baseline_windows))

    result = await walk(
        fake,
        metric_name="rebuffer",
        window=window,
        dimensions=["device_type", "app_version"],
        min_weight_fraction=0.0,
        max_depth=1,
    )

    assert [step.dimension for step in result.path] == ["device_type"]
    assert result.stop_reason is StopReason.MAX_DEPTH
    # Only the first level's queries were issued.
    assert len(result.query_log) == 1 + 4


async def test_walk_rejects_empty_dimensions():
    fake = _WalkFakeGateway({})
    with pytest.raises(ValueError, match="dimensions"):
        await walk(
            fake,
            metric_name="rebuffer",
            window=(datetime(2026, 3, 2), datetime(2026, 3, 3)),
            dimensions=[],
        )


async def test_walk_rejects_negative_max_depth():
    fake = _WalkFakeGateway({})
    with pytest.raises(ValueError, match="max_depth"):
        await walk(
            fake,
            metric_name="rebuffer",
            window=(datetime(2026, 3, 2), datetime(2026, 3, 3)),
            dimensions=["device_type"],
            max_depth=-1,
        )


async def test_walk_max_depth_zero_stops_immediately_with_no_queries():
    fake = _WalkFakeGateway({})
    result = await walk(
        fake,
        metric_name="rebuffer",
        window=(datetime(2026, 3, 2), datetime(2026, 3, 3)),
        dimensions=["device_type"],
        max_depth=0,
    )
    assert result.path == ()
    assert result.stop_reason is StopReason.MAX_DEPTH
    assert result.query_log == ()


def test_default_max_depth_matches_dimension_hierarchy_size():
    from continuity.data.topology import DIMENSION_HIERARCHY

    assert len(DIMENSION_HIERARCHY) == DEFAULT_MAX_DEPTH
