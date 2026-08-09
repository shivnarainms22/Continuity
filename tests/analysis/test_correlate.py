"""Unit tests for continuity.analysis.correlate -- pure ranking/scoring/classification
logic plus a fake-gateway check of the async orchestration. No ClickHouse involved.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from continuity.analysis.correlate import (
    ChangeRow,
    DisconfirmingEvidence,
    RankedChange,
    RejectedChange,
    SiblingMeasurement,
    classify_change,
    compute_disconfirming_evidence,
    correlate_changes,
    rank_candidates,
)
from continuity.analysis.slices import Slice
from continuity.gateway.mcp_gateway import QueryResult

_ONSET = datetime(2026, 2, 12, 21, 0, 0)


def _row(
    change_id: int,
    changed_at: datetime,
    *,
    dimension_key: str = "app_version",
    dimension_value: str = "8.2.0",
    change_type: str = "app_release",
    component: str = "roku_app",
    description: str = "test change",
) -> ChangeRow:
    return ChangeRow(
        change_id=change_id,
        changed_at=changed_at,
        change_type=change_type,
        component=component,
        description=description,
        dimension_key=dimension_key,
        dimension_value=dimension_value,
    )


_BLAST_RADIUS = Slice().refine("device_type", "roku").refine("app_version", "8.2.0")


# ---------------------------------------------------------------------------
# THE MANDATORY TEST: a change after onset must never be ranked, no matter how
# perfect its dimensional match, and must be explicitly rejected as "too late".
# ---------------------------------------------------------------------------


def test_a_change_after_onset_is_rejected_even_with_a_perfect_dimensional_match():
    after_onset = _row(1, _ONSET + timedelta(minutes=5))
    before_onset = _row(2, _ONSET - timedelta(hours=1))

    candidates, rejected = rank_candidates(
        [after_onset, before_onset], blast_radius=_BLAST_RADIUS, onset=_ONSET
    )

    candidate_ids = [c.change_id for c in candidates]
    assert 1 not in candidate_ids, "a change after onset must never be ranked"
    assert 2 in candidate_ids

    rejected_ids = {r.change_id: r.reason for r in rejected}
    assert 1 in rejected_ids
    assert "too late" in rejected_ids[1]


def test_a_change_exactly_at_onset_is_accepted_not_rejected():
    at_onset = _row(1, _ONSET)
    candidates, rejected = rank_candidates([at_onset], blast_radius=_BLAST_RADIUS, onset=_ONSET)
    assert [c.change_id for c in candidates] == [1]
    assert rejected == ()


def test_tolerance_accepts_a_change_shortly_after_onset():
    """Tolerance absorbs detection-granularity noise around the boundary -- a change a
    couple of minutes after onset, within tolerance, is still a viable candidate."""
    shortly_after = _row(1, _ONSET + timedelta(minutes=2))
    candidates, rejected = rank_candidates(
        [shortly_after],
        blast_radius=_BLAST_RADIUS,
        onset=_ONSET,
        tolerance=timedelta(minutes=5),
    )
    assert [c.change_id for c in candidates] == [1]
    assert rejected == ()
    # Treated as maximally proximate, not penalised for being technically after onset.
    assert candidates[0].score == pytest.approx(1.0)


def test_a_change_before_the_lookback_horizon_is_rejected_as_outside_window():
    too_early = _row(1, _ONSET - timedelta(hours=10))
    candidates, rejected = rank_candidates(
        [too_early], blast_radius=_BLAST_RADIUS, onset=_ONSET, lookback=timedelta(hours=6)
    )
    assert candidates == ()
    assert len(rejected) == 1
    assert "outside window" in rejected[0].reason


# ---------------------------------------------------------------------------
# Temporal proximity scoring.
# ---------------------------------------------------------------------------


def test_score_decays_linearly_with_distance_from_onset():
    lookback = timedelta(hours=6)
    right_before = _row(1, _ONSET - timedelta(minutes=1))
    halfway = _row(2, _ONSET - timedelta(hours=3))
    at_horizon = _row(3, _ONSET - timedelta(hours=6))

    candidates, _ = rank_candidates(
        [right_before, halfway, at_horizon],
        blast_radius=_BLAST_RADIUS,
        onset=_ONSET,
        lookback=lookback,
    )
    by_id = {c.change_id: c for c in candidates}

    assert by_id[1].score > by_id[2].score > by_id[3].score
    assert by_id[1].score == pytest.approx(1.0, abs=0.01)
    assert by_id[2].score == pytest.approx(0.5, abs=0.01)
    assert by_id[3].score == pytest.approx(0.0, abs=0.01)


def test_temporal_delta_is_positive_for_a_change_before_onset():
    row = _row(1, _ONSET - timedelta(hours=2))
    candidates, _ = rank_candidates([row], blast_radius=_BLAST_RADIUS, onset=_ONSET)
    assert candidates[0].temporal_delta == timedelta(hours=2)


# ---------------------------------------------------------------------------
# Dimensional overlap: match / contradiction / unrelated.
# ---------------------------------------------------------------------------


def test_matching_dimension_scores_higher_than_unrelated_dimension_at_the_same_time():
    matching = _row(
        1, _ONSET - timedelta(hours=1), dimension_key="app_version", dimension_value="8.2.0"
    )
    unrelated = _row(2, _ONSET - timedelta(hours=1), dimension_key="isp", dimension_value="comcast")

    candidates, rejected = rank_candidates(
        [matching, unrelated], blast_radius=_BLAST_RADIUS, onset=_ONSET
    )

    assert rejected == ()
    by_id = {c.change_id: c for c in candidates}
    assert by_id[1].dimensional_overlap is True
    assert by_id[2].dimensional_overlap is False
    assert by_id[1].score > by_id[2].score
    assert candidates[0].change_id == 1


def test_contradicting_dimension_is_rejected_not_merely_down_ranked():
    """The blast radius pins device_type=roku; a change to device_type=firetv targets a
    disjoint population and cannot be the cause of THIS blast radius."""
    contradicting = _row(
        1, _ONSET - timedelta(hours=1), dimension_key="device_type", dimension_value="firetv"
    )

    candidates, rejected = rank_candidates(
        [contradicting], blast_radius=_BLAST_RADIUS, onset=_ONSET
    )

    assert candidates == ()
    assert len(rejected) == 1
    assert "no dimensional overlap" in rejected[0].reason


def test_unrelated_dimension_is_weak_evidence_but_still_ranked():
    unrelated = _row(1, _ONSET - timedelta(hours=1), dimension_key="isp", dimension_value="comcast")
    candidates, rejected = rank_candidates([unrelated], blast_radius=_BLAST_RADIUS, onset=_ONSET)
    assert rejected == ()
    assert len(candidates) == 1
    assert candidates[0].dimensional_overlap is False
    assert 0 < candidates[0].score < 1.0


# ---------------------------------------------------------------------------
# Deterministic ordering.
# ---------------------------------------------------------------------------


def test_multiple_changes_at_the_same_instant_are_ordered_deterministically_by_change_id():
    same_instant = _ONSET - timedelta(hours=1)
    a = _row(5, same_instant)
    b = _row(3, same_instant)
    c = _row(9, same_instant)

    candidates, _ = rank_candidates([a, b, c], blast_radius=_BLAST_RADIUS, onset=_ONSET)

    assert [cand.change_id for cand in candidates] == [3, 5, 9]


def test_ranking_is_stable_across_repeated_calls_with_reordered_input():
    same_instant = _ONSET - timedelta(hours=1)
    rows = [_row(5, same_instant), _row(3, same_instant), _row(9, same_instant)]

    first, _ = rank_candidates(rows, blast_radius=_BLAST_RADIUS, onset=_ONSET)
    second, _ = rank_candidates(list(reversed(rows)), blast_radius=_BLAST_RADIUS, onset=_ONSET)

    assert [c.change_id for c in first] == [c.change_id for c in second]


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_no_changes_in_window_returns_empty_not_an_error():
    candidates, rejected = rank_candidates([], blast_radius=_BLAST_RADIUS, onset=_ONSET)
    assert candidates == ()
    assert rejected == ()


def test_rejected_candidates_carry_the_original_change_fields():
    row = _row(
        1,
        _ONSET + timedelta(hours=1),
        change_type="network_config",
        component="cdn_x",
        description="a plausible-sounding but too-late change",
    )
    _, rejected = rank_candidates([row], blast_radius=_BLAST_RADIUS, onset=_ONSET)
    assert rejected[0].change_type == "network_config"
    assert rejected[0].component == "cdn_x"
    assert rejected[0].description == "a plausible-sounding but too-late change"


def test_classify_change_returns_a_ranked_change_or_a_rejected_change():
    accepted = classify_change(
        _row(1, _ONSET - timedelta(hours=1)), blast_radius=_BLAST_RADIUS, onset=_ONSET
    )
    rejected = classify_change(
        _row(2, _ONSET + timedelta(hours=1)), blast_radius=_BLAST_RADIUS, onset=_ONSET
    )
    assert isinstance(accepted, RankedChange)
    assert isinstance(rejected, RejectedChange)


# ---------------------------------------------------------------------------
# Disconfirming evidence: pure function.
# ---------------------------------------------------------------------------


def test_no_sibling_dimension_reports_nothing_to_check():
    evidence = compute_disconfirming_evidence(
        "title_id", "1", sibling_dimension=None, sibling_measurements=(), higher_is_worse=True
    )
    assert evidence.sibling_dimension is None
    assert evidence.siblings == ()
    assert "nothing to check" in evidence.note or "no dimension" in evidence.note


def test_change_that_touched_no_other_sibling_at_all_is_narrow_and_strong():
    """The "went only to Roku" case: after excluding the blast radius's own value (the
    caller's job -- see correlate_changes._attach_disconfirming_evidence), there is no
    other sibling data at all, so there is nothing that could disconfirm this change."""
    evidence = compute_disconfirming_evidence(
        "app_version",
        "8.2.0",
        sibling_dimension="device_type",
        sibling_measurements=(),
        higher_is_worse=True,
    )
    assert evidence.siblings == ()
    assert "narrow" in evidence.note
    assert "strong" in evidence.note


def test_change_that_touched_every_device_type_but_only_roku_degraded_is_weaker_evidence():
    """A deploy that went to every device type but only Roku degraded -- exactly the
    example from the spec -- must record broad, non-degraded exposure."""
    measurements = [
        SiblingMeasurement(value="roku", metric_value=0.02, baseline_value=0.001),
        SiblingMeasurement(value="firetv", metric_value=0.001, baseline_value=0.001),
        SiblingMeasurement(value="ios", metric_value=0.0011, baseline_value=0.001),
        SiblingMeasurement(value="android", metric_value=0.0009, baseline_value=0.001),
    ]
    evidence = compute_disconfirming_evidence(
        "app_version",
        "8.2.0",
        sibling_dimension="device_type",
        sibling_measurements=measurements,
        higher_is_worse=True,
    )
    assert evidence.siblings_checked == 4
    assert evidence.siblings_degraded == 1
    assert evidence.siblings_not_degraded == 3
    assert "weaker" in evidence.note


def test_change_where_every_sibling_also_degraded_is_broad_and_weak():
    measurements = [
        SiblingMeasurement(value="roku", metric_value=0.02, baseline_value=0.001),
        SiblingMeasurement(value="firetv", metric_value=0.03, baseline_value=0.001),
    ]
    evidence = compute_disconfirming_evidence(
        "app_version",
        "8.2.0",
        sibling_dimension="device_type",
        sibling_measurements=measurements,
        higher_is_worse=True,
    )
    assert evidence.siblings_not_degraded == 0
    assert "every other" in evidence.note


def test_disconfirming_evidence_handles_missing_baseline_without_crashing():
    measurements = [SiblingMeasurement(value="new_device", metric_value=0.5, baseline_value=None)]
    evidence = compute_disconfirming_evidence(
        "app_version",
        "8.2.0",
        sibling_dimension="device_type",
        sibling_measurements=measurements,
        higher_is_worse=True,
    )
    assert evidence.siblings[0].degraded is None
    assert evidence.siblings_checked == 1
    assert evidence.siblings_degraded == 0
    assert evidence.siblings_not_degraded == 0


def test_bitrate_drop_direction_is_respected_for_degraded_determination():
    """bitrate is lower-is-worse: a DROP is degraded, a rise is not."""
    measurements = [
        SiblingMeasurement(value="dropped", metric_value=1000.0, baseline_value=3000.0),
        SiblingMeasurement(value="steady", metric_value=2900.0, baseline_value=3000.0),
    ]
    evidence = compute_disconfirming_evidence(
        "title_id",
        "1",
        sibling_dimension="device_type",
        sibling_measurements=measurements,
        higher_is_worse=False,
    )
    by_value = {s.value: s for s in evidence.siblings}
    assert by_value["dropped"].degraded is True
    assert by_value["steady"].degraded is False


def test_disconfirming_evidence_is_attached_to_the_candidate_type():
    row = _row(1, _ONSET - timedelta(hours=1))
    candidate = classify_change(row, blast_radius=_BLAST_RADIUS, onset=_ONSET)
    assert isinstance(candidate, RankedChange)
    assert isinstance(candidate.disconfirming_evidence, DisconfirmingEvidence)


# ---------------------------------------------------------------------------
# Async orchestration against a fake gateway: one query to fetch candidates,
# regardless of row count, plus disconfirming-evidence enrichment queries.
# ---------------------------------------------------------------------------


class _FakeGateway:
    def __init__(self, change_log_rows: list[dict], sibling_rows: dict[str, list[dict]]) -> None:
        self._change_log_rows = change_log_rows
        self._sibling_rows = sibling_rows
        self.queries: list[str] = []

    async def query(self, sql: str) -> QueryResult:
        self.queries.append(sql)
        if "FROM change_log" in sql:
            rows = self._change_log_rows
        else:
            # Sibling window/baseline queries: pick by which marker literal is present.
            rows = []
            for marker, marker_rows in self._sibling_rows.items():
                if marker in sql:
                    rows = marker_rows
                    break
        columns = list(rows[0].keys()) if rows else []
        return QueryResult(sql=sql, columns=columns, rows=rows)


async def test_correlate_changes_issues_exactly_one_query_to_fetch_candidates():
    onset = datetime(2026, 2, 12, 21, 0, 0)
    change_log_rows = [
        {
            "change_id": 1,
            "changed_at": (onset - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "change_type": "app_release",
            "component": "roku_app",
            "description": "roku 8.2.0 rollout",
            "dimension_key": "app_version",
            "dimension_value": "8.2.0",
        },
        {
            "change_id": 2,
            "changed_at": (onset - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "change_type": "network_config",
            "component": "unrelated",
            "description": "unrelated change",
            "dimension_key": "isp",
            "dimension_value": "comcast",
        },
    ]
    fake = _FakeGateway(change_log_rows, sibling_rows={})

    result = await correlate_changes(
        fake,
        blast_radius=_BLAST_RADIUS,
        anomaly_window=(onset, onset + timedelta(hours=8)),
    )

    change_log_queries = [q for q in fake.queries if "FROM change_log" in q]
    assert len(change_log_queries) == 1, "must fetch candidates with exactly one query"
    assert len(result.candidates) == 2
    assert result.candidates[0].change_id == 1  # matching dimension outranks unrelated
    assert result.sql == change_log_queries[0]
    for candidate in result.candidates:
        assert candidate.disconfirming_evidence is not None


async def test_correlate_changes_returns_empty_candidates_when_no_changes_in_window():
    onset = datetime(2026, 2, 12, 21, 0, 0)
    fake = _FakeGateway(change_log_rows=[], sibling_rows={})

    result = await correlate_changes(
        fake, blast_radius=_BLAST_RADIUS, anomaly_window=(onset, onset + timedelta(hours=8))
    )

    assert result.candidates == ()
    assert result.rejected == ()


async def test_correlate_changes_rejects_invalid_anomaly_window():
    onset = datetime(2026, 2, 12, 21, 0, 0)
    fake = _FakeGateway(change_log_rows=[], sibling_rows={})
    with pytest.raises(ValueError, match="before end"):
        await correlate_changes(
            fake, blast_radius=_BLAST_RADIUS, anomaly_window=(onset, onset - timedelta(hours=1))
        )
