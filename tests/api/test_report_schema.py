"""Unit tests for continuity/api/report_schema.py -- pure JSON reshaping of the same
dataclasses continuity/analysis/cli.py's render_brief walks, plus a fake-gateway check
of fetch_incident_series's async orchestration. No ClickHouse involved: every object
below is constructed directly, exactly like tests/analysis/test_correlate.py and
tests/analysis/test_impact.py already do for the same dataclasses.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from continuity.analysis.cli import (
    IncidentInvestigation,
    InvestigationReport,
    MergedIncident,
    RefinedIncident,
    StageTiming,
)
from continuity.analysis.correlate import (
    CorrelationResult,
    DisconfirmingEvidence,
    RankedChange,
    RejectedChange,
)
from continuity.analysis.detect import AnomalyWindow, DetectionResult
from continuity.analysis.impact import ImpactResult, Methodology
from continuity.analysis.slices import Slice
from continuity.analysis.walk import RefinementStep, StopReason, WalkResult
from continuity.api import report_schema
from continuity.gateway.mcp_gateway import QueryResult

_SLICE = Slice().refine("device_type", "roku").refine("app_version", "8.2.0")
_START = datetime(2026, 2, 12, 18, 0, 0)
_END = datetime(2026, 2, 13, 2, 0, 0)


def _anomaly_window(*, start: datetime, end: datetime, peak_z: float = 8.0) -> AnomalyWindow:
    return AnomalyWindow(
        slice=_SLICE,
        metric="rebuffer",
        start=start,
        end=end,
        peak_z=peak_z,
        peak_value=0.09,
        expected_at_peak=0.02,
        bucket_count=4,
        sql="SELECT 1 -- anomaly window",
    )


def _walk_result() -> WalkResult:
    step = RefinementStep(
        dimension="device_type",
        value="roku",
        share_of_deviation=0.8,
        lift=3.2,
        contribution=0.5,
        weight=1000.0,
        sql="SELECT 1 -- split",
        baseline_sql="SELECT 1 -- baseline",
    )
    return WalkResult(
        metric="rebuffer",
        window=(_START, _END),
        baseline_windows=((_START - timedelta(weeks=1), _END - timedelta(weeks=1)),),
        path=(step,),
        final_slice=_SLICE,
        stop_reason=StopReason.LOW_LIFT,
        stop_detail="device_type=roku has lift 1.1 < min_lift 1.5",
        elapsed_ms=12.3,
        query_log=(),
    )


def _correlation_result(*, with_candidate: bool) -> CorrelationResult:
    evidence = DisconfirmingEvidence(
        dimension_key="app_version",
        dimension_value="8.2.0",
        sibling_dimension="device_type",
        siblings=(),
        note="this change touched no other device_type value in the window",
    )
    candidates = (
        (
            RankedChange(
                change_id=1,
                changed_at=_START - timedelta(hours=3),
                change_type="app_release",
                component="roku_app",
                description="Roku app 8.2.0 rollout increased rebuffer rate",
                dimension_key="app_version",
                dimension_value="8.2.0",
                score=0.9,
                temporal_delta=timedelta(hours=3),
                dimensional_overlap=True,
                disconfirming_evidence=evidence,
                sql="SELECT 1 -- candidates",
            ),
        )
        if with_candidate
        else ()
    )
    rejected = (
        RejectedChange(
            change_id=2,
            changed_at=_START - timedelta(days=2),
            change_type="network_config",
            component="unrelated",
            description="unrelated change",
            dimension_key="isp",
            dimension_value="comcast",
            reason="outside window: changed too long before onset",
            sql="SELECT 1 -- candidates",
        ),
    )
    return CorrelationResult(
        blast_radius=_SLICE,
        onset=_START,
        end=_END,
        lookback=timedelta(hours=6),
        tolerance=timedelta(0),
        candidates=candidates,
        rejected=rejected,
        sql="SELECT 1 -- candidates",
    )


def _impact_result() -> ImpactResult:
    methodology = Methodology(
        base_monthly_churn=Decimal("0.025"),
        base_churn_variation=Decimal("0.40"),
        tenure_multiplier_at_signup=Decimal("2.0"),
        tenure_multiplier_floor=Decimal("0.5"),
        tenure_half_life_days=Decimal("180"),
        severity_multiplier_max=Decimal("3.0"),
        severity_sessions_half_saturation=Decimal("5"),
        severity_qoe_half_saturation=Decimal("2.0"),
        churn_risk_ceiling=Decimal("1.0"),
        qoe_delta_ratio=Decimal("2.5"),
        affected_subscriber_count=42,
        window=(_START, _END),
        slice=_SLICE,
        notes="heuristic, not a trained model",
    )
    return ImpactResult(
        slice=_SLICE,
        window=(_START, _END),
        affected_subscribers=42,
        arr_at_risk_low=Decimal("1000.00"),
        arr_at_risk_expected=Decimal("2000.00"),
        arr_at_risk_high=Decimal("3000.00"),
        methodology=methodology,
        sql="SELECT 1 -- impact",
    )


def _incident_investigation(*, used_fallback: bool, with_candidate: bool) -> IncidentInvestigation:
    population_window = _anomaly_window(start=_START, end=_START + timedelta(hours=2))
    population = MergedIncident(windows=(population_window,), walks=(_walk_result(),))

    refined_windows = (
        population.windows
        if used_fallback
        else (_anomaly_window(start=_START - timedelta(hours=1), end=_END, peak_z=10.0),)
    )
    refined = RefinedIncident(
        population_incident=population,
        refine_detection=DetectionResult(
            slice=_SLICE,
            metric="rebuffer",
            windows=[] if used_fallback else list(refined_windows),
            total_buckets=100,
            anomalous_buckets=0 if used_fallback else 10,
            unknown_buckets=0,
            sql="SELECT 1 -- refine detect",
        ),
        windows=refined_windows,
        used_fallback=used_fallback,
        fallback_reason="thin slice, no signal" if used_fallback else None,
        typical_deviation_ratio=Decimal("2.5"),
        peak_deviation_ratio=Decimal("3.5"),
        severity_sql="SELECT 1 -- severity",
    )
    return IncidentInvestigation(
        incident=refined,
        correlation=_correlation_result(with_candidate=with_candidate),
        impact=_impact_result(),
        qoe_delta_ratio=Decimal("2.5"),
    )


def _report(incidents: tuple[IncidentInvestigation, ...]) -> InvestigationReport:
    detection = DetectionResult(
        slice=Slice(),
        metric="rebuffer",
        windows=[_anomaly_window(start=_START, end=_START + timedelta(hours=2))],
        total_buckets=500,
        anomalous_buckets=20,
        unknown_buckets=5,
        sql="SELECT 1 -- population detect",
    )
    return InvestigationReport(
        metric_name="rebuffer",
        window=(_START, _END),
        description="incident INC-APP-ROKU-820",
        detection=detection,
        incidents=incidents,
        stage_timings=(
            StageTiming("session_startup", 5.0, ()),
            StageTiming("detect", 42.0, ()),
        ),
        total_elapsed_ms=100.0,
    )


# ---------------------------------------------------------------------------
# slice_predicates
# ---------------------------------------------------------------------------


def test_slice_predicates_orders_coarse_to_fine():
    assert report_schema.slice_predicates(_SLICE) == [
        {"dimension": "device_type", "value": "roku"},
        {"dimension": "app_version", "value": "8.2.0"},
    ]


def test_slice_predicates_of_the_whole_population_is_empty():
    assert report_schema.slice_predicates(Slice()) == []


# ---------------------------------------------------------------------------
# serialize_report -- structure, not prose.
# ---------------------------------------------------------------------------


def test_serialize_report_with_no_incidents_is_a_healthy_empty_result():
    report = _report(())
    result = report_schema.serialize_report(report, [])
    assert result["incidents"] == []
    assert result["detection"]["windows_found"] == 1
    assert result["metric"] == "rebuffer"


def test_serialize_report_rejects_mismatched_series_length():
    report = _report((_incident_investigation(used_fallback=False, with_candidate=True),))
    with pytest.raises(ValueError, match="incident_series"):
        report_schema.serialize_report(report, [])


def test_serialize_report_refined_incident_carries_the_true_span_not_the_population_span():
    ir = _incident_investigation(used_fallback=False, with_candidate=True)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    what_happened = result["incidents"][0]["what_happened"]
    assert what_happened["used_fallback"] is False
    assert what_happened["refined_span"] is not None
    assert what_happened["refined_peak_z"] == 10.0
    assert what_happened["typical_multiple"] == pytest.approx(3.5)


def test_serialize_report_fallback_incident_has_no_refined_span():
    ir = _incident_investigation(used_fallback=True, with_candidate=True)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    what_happened = result["incidents"][0]["what_happened"]
    assert what_happened["used_fallback"] is True
    assert what_happened["refined_span"] is None
    assert what_happened["fallback_reason"] == "thin slice, no signal"


def test_serialize_report_who_affected_carries_the_drill_down_path_and_sql():
    ir = _incident_investigation(used_fallback=False, with_candidate=True)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    who_affected = result["incidents"][0]["who_affected"]
    assert who_affected["predicates"] == [
        {"dimension": "device_type", "value": "roku"},
        {"dimension": "app_version", "value": "8.2.0"},
    ]
    assert who_affected["drill_down"] == [
        {
            "dimension": "device_type",
            "value": "roku",
            "share_of_deviation": 0.8,
            "lift": 3.2,
            "weight": 1000.0,
            "sql": "SELECT 1 -- split",
            "baseline_sql": "SELECT 1 -- baseline",
        }
    ]


def test_serialize_report_probable_cause_top_candidate_carries_disconfirming_evidence():
    ir = _incident_investigation(used_fallback=False, with_candidate=True)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    probable_cause = result["incidents"][0]["probable_cause"]
    assert probable_cause["top"]["change_id"] == 1
    assert probable_cause["top"]["disconfirming_evidence"]["sibling_dimension"] == "device_type"
    assert len(probable_cause["rejected"]) == 1


def test_serialize_report_probable_cause_with_no_candidates_is_explicit_not_missing():
    ir = _incident_investigation(used_fallback=False, with_candidate=False)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    assert result["incidents"][0]["probable_cause"]["top"] is None
    recommended = result["incidents"][0]["recommended_action"]
    assert recommended == {
        "has_candidate": False,
        "change_id": None,
        "component": None,
        "description": None,
    }


def test_serialize_report_recommended_action_names_the_top_candidate_change():
    ir = _incident_investigation(used_fallback=False, with_candidate=True)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    recommended = result["incidents"][0]["recommended_action"]
    assert recommended["has_candidate"] is True
    assert recommended["change_id"] == 1
    assert recommended["component"] == "roku_app"


def test_serialize_report_impact_money_is_a_plain_float_not_a_decimal_object():
    ir = _incident_investigation(used_fallback=False, with_candidate=True)
    report = _report((ir,))
    series = {"points": [], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    impact = result["incidents"][0]["impact"]
    assert impact["arr_at_risk_low"] == 1000.0
    assert impact["arr_at_risk_expected"] == 2000.0
    assert impact["arr_at_risk_high"] == 3000.0
    assert isinstance(impact["arr_at_risk_low"], float)


def test_serialize_report_series_carries_the_true_anomaly_windows():
    ir = _incident_investigation(used_fallback=False, with_candidate=True)
    report = _report((ir,))
    series = {"points": [{"bucket": "x"}], "sql": "SELECT 1 -- series", "metric": "rebuffer"}

    result = report_schema.serialize_report(report, [series])

    incident_series = result["incidents"][0]["series"]
    assert incident_series["points"] == [{"bucket": "x"}]
    assert incident_series["sql"] == "SELECT 1 -- series"
    assert len(incident_series["anomaly_windows"]) == 1


# ---------------------------------------------------------------------------
# fetch_incident_series -- async orchestration against a fake gateway.
# ---------------------------------------------------------------------------


class _FakeSeriesGateway:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.queries: list[str] = []

    async def query(self, sql: str) -> QueryResult:
        self.queries.append(sql)
        return QueryResult(sql=sql, columns=["bucket", "value"], rows=self._rows)


def _row(dt: datetime, value: float) -> dict:
    return {"bucket": dt.strftime("%Y-%m-%d %H:%M:%S"), "value": value}


async def test_fetch_incident_series_issues_exactly_one_query():
    fake = _FakeSeriesGateway(rows=[])
    result = await report_schema.fetch_incident_series(
        fake, slice_=Slice(), metric_name="rebuffer", span=(_START, _START + timedelta(hours=1))
    )
    assert len(fake.queries) == 1
    assert result["sql"] == fake.queries[0]
    assert result["metric"] == "rebuffer"


async def test_fetch_incident_series_marks_a_bucket_with_no_comparison_history_as_unknown():
    span = (_START, _START + timedelta(minutes=5))
    fake = _FakeSeriesGateway(rows=[_row(_START, 0.09)])  # no comparison-week history at all

    result = await report_schema.fetch_incident_series(
        fake, slice_=Slice(), metric_name="rebuffer", span=span, padding=timedelta(0)
    )

    bucket = next(p for p in result["points"] if p["bucket"] == report_schema.iso(_START))
    assert bucket["status"] == "unknown"
    assert bucket["expected"] is None
    assert bucket["lower"] is None
    assert bucket["upper"] is None


async def test_fetch_incident_series_computes_a_baseline_band_with_enough_history():
    span = (_START, _START + timedelta(minutes=5))
    comparison_values = [0.019, 0.020, 0.021, 0.020]  # not all identical -> a real, nonzero MAD
    rows = [_row(_START, 0.09)]
    for week, value in enumerate(comparison_values, start=1):
        rows.append(_row(_START - timedelta(weeks=week), value))
    fake = _FakeSeriesGateway(rows=rows)

    result = await report_schema.fetch_incident_series(
        fake, slice_=Slice(), metric_name="rebuffer", span=span, padding=timedelta(0)
    )

    bucket = next(p for p in result["points"] if p["bucket"] == report_schema.iso(_START))
    assert bucket["status"] == "anomalous"
    assert bucket["expected"] == pytest.approx(0.02)
    assert bucket["lower"] is not None
    assert bucket["upper"] is not None
    assert bucket["lower"] < bucket["expected"] < bucket["upper"]
