import re

import pytest

from continuity.analysis.metrics import METRICS, Metric, get_metric


def test_metric_names_match_ground_truth_effect_names():
    """incidents.py writes effects with these exact metric names -- baseline/detect
    later must be able to look up a metric by the name a planted incident uses."""
    assert set(METRICS) == {"rebuffer", "startup", "bitrate", "errors"}


def test_get_metric_returns_the_named_metric():
    assert get_metric("rebuffer") is METRICS["rebuffer"]


def test_get_metric_rejects_unknown_name():
    with pytest.raises(KeyError, match="bogus_metric"):
        get_metric("bogus_metric")


@pytest.mark.parametrize("name", ["rebuffer", "startup", "bitrate", "errors"])
def test_every_metric_defines_both_sql_forms_nonempty(name):
    metric = METRICS[name]
    assert isinstance(metric, Metric)
    assert metric.rollup_sql.strip()
    assert metric.raw_sql.strip()
    assert metric.label
    assert metric.unit


def test_rebuffer_is_higher_is_worse():
    assert METRICS["rebuffer"].higher_is_worse is True


def test_startup_is_higher_is_worse():
    assert METRICS["startup"].higher_is_worse is True


def test_errors_is_higher_is_worse():
    assert METRICS["errors"].higher_is_worse is True


def test_bitrate_is_lower_is_worse():
    """Direction matters for later deviation logic: a bitrate drop is degradation,
    not an improvement, so it must NOT share the higher_is_worse flag."""
    assert METRICS["bitrate"].higher_is_worse is False


def test_rebuffer_rollup_sql_is_ratio_of_sums_not_average_of_ratios():
    sql = METRICS["rebuffer"].rollup_sql
    assert "sum(rebuffer_ms)" in sql
    assert "sum(watched_ms)" in sql
    assert "avg(" not in sql.lower()


def test_rebuffer_raw_sql_is_ratio_of_sums_not_average_of_ratios():
    sql = METRICS["rebuffer"].raw_sql
    assert "sum(rebuffer_ms)" in sql
    assert "sum(watched_ms)" in sql
    assert "avg(" not in sql.lower()


def test_rebuffer_denominator_is_guarded_against_division_by_zero():
    for sql in (METRICS["rebuffer"].rollup_sql, METRICS["rebuffer"].raw_sql):
        assert "nullIf(sum(watched_ms), 0)" in sql


def test_errors_denominator_is_guarded_against_division_by_zero():
    sql = METRICS["errors"].rollup_sql
    assert "nullIf(" in sql


def test_rebuffer_rollup_uses_plain_sum_not_a_merge_combinator():
    """rebuffer_ms/watched_ms are SimpleAggregateFunction(sum, ...) columns in the
    rollup: plain sum() combines them correctly, no xMerge needed."""
    sql = METRICS["rebuffer"].rollup_sql
    assert "sumMerge" not in sql
    assert re.search(r"\bsum\(", sql)


def test_startup_rollup_uses_quantilesTDigestMerge_on_the_state_column():
    sql = METRICS["startup"].rollup_sql
    assert "quantilesTDigestMerge" in sql
    assert "startup_q" in sql


def test_startup_raw_uses_plain_quantile_function_on_startup_ms():
    sql = METRICS["startup"].raw_sql
    assert "quantileTDigest" in sql
    assert "startup_ms" in sql
    # startup_ms is only meaningful on 'start' events (see schema.py qoe_rollup_5m_mv)
    assert "event_type" in sql
    assert "'start'" in sql


def test_startup_is_p95():
    assert "0.95" in METRICS["startup"].rollup_sql
    assert "0.95" in METRICS["startup"].raw_sql


def test_bitrate_rollup_uses_avgMerge_on_the_state_column():
    sql = METRICS["bitrate"].rollup_sql
    assert sql.strip() == "avgMerge(bitrate_avg)"


def test_bitrate_raw_uses_plain_avg_filtered_to_heartbeat_events():
    sql = METRICS["bitrate"].raw_sql
    assert "avg" in sql.lower()
    assert "bitrate_kbps" in sql
    assert "'heartbeat'" in sql


def test_no_metric_emits_a_bare_count_against_the_rollup():
    """count() on qoe_rollup_5m returns unmerged part counts, not rows -- it must
    never appear in a rollup expression (see CLAUDE.md hard constraint)."""
    bare_count = re.compile(r"(?<![A-Za-z])count\s*\(")
    for name, metric in METRICS.items():
        assert not bare_count.search(metric.rollup_sql), (
            f"{name}.rollup_sql contains a bare count(): {metric.rollup_sql!r}"
        )


def test_sql_for_selects_raw_form_when_slice_requires_raw_events():
    metric = METRICS["rebuffer"]
    assert metric.sql_for(raw_events=True) == metric.raw_sql
    assert metric.sql_for(raw_events=False) == metric.rollup_sql
