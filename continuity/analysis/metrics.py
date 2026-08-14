"""QoE metric definitions: SQL expressions over the rollup and over raw events.

Pure logic, no SQL execution. Each Metric carries two expression forms because the
two tables store the same quantity differently (see continuity/data/schema.py):

* ``qoe_rollup_5m`` columns are aggregate STATES (or SimpleAggregateFunction sums)
  produced by the materialized view, so they must be read with merge combinators:
  ``uniqMerge``, ``quantilesTDigestMerge``, ``avgMerge``. Plain ``sum()`` is correct
  for the SimpleAggregateFunction columns (starts, errors, watched_ms, rebuffer_ms)
  because sum is associative over already-summed partials.
* ``playback_events`` columns are plain per-event values, read with the ordinary
  aggregate functions (and an ``...If`` filter where the column is only meaningful on
  one event_type, mirroring the materialized view's own filtering).

Never emit a bare ``count()`` against the rollup: it is an AggregatingMergeTree, so
``count()`` returns however many unmerged parts exist at query time, not a row count
(see CLAUDE.md). Every rollup expression here is built from merge-invariant
aggregates instead.

Ratio metrics (rebuffer, errors) are computed as a ratio of sums, never an average of
per-bucket ratios -- the two differ once you aggregate across buckets, and only the
former is correct. Each denominator is guarded with ``nullIf(..., 0)`` so a bucket
with no watch time or no starts yields NULL rather than a division error.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str
    higher_is_worse: bool
    rollup_sql: str
    raw_sql: str

    def sql_for(self, *, raw_events: bool) -> str:
        """The expression form for the table a given Slice requires.

        Pass ``slice_.requires_raw_events`` -- see continuity/analysis/slices.py.
        """
        return self.raw_sql if raw_events else self.rollup_sql


METRICS: dict[str, Metric] = {
    "rebuffer": Metric(
        name="rebuffer",
        label="Rebuffer ratio",
        unit="ratio",
        higher_is_worse=True,
        rollup_sql="sum(rebuffer_ms) / nullIf(sum(watched_ms), 0)",
        raw_sql="sum(rebuffer_ms) / nullIf(sum(watched_ms), 0)",
    ),
    "startup": Metric(
        name="startup",
        label="Startup latency (p95)",
        unit="ms",
        higher_is_worse=True,
        rollup_sql="quantilesTDigestMerge(0.5, 0.95)(startup_q)[2]",
        raw_sql="quantileTDigestIf(0.95)(startup_ms, event_type = 'start')",
    ),
    "bitrate": Metric(
        name="bitrate",
        label="Average bitrate",
        unit="kbps",
        higher_is_worse=False,
        rollup_sql="avgMerge(bitrate_avg)",
        raw_sql="avgIf(bitrate_kbps, event_type = 'heartbeat')",
    ),
    "errors": Metric(
        name="errors",
        label="Error rate",
        unit="errors per start",
        higher_is_worse=True,
        rollup_sql="sum(errors) / nullIf(sum(starts), 0)",
        raw_sql=(
            "sum(toUInt64(event_type = 'error')) / "
            "nullIf(sum(toUInt64(event_type = 'start')), 0)"
        ),
    ),
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(METRICS)}") from None
