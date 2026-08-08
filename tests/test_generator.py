import time
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from continuity.data import topology
from continuity.data.catalog import generate_subscribers, generate_titles
from continuity.data.generator import (
    CHANGE_LOG_COLUMNS,
    PLAYBACK_EVENTS_COLUMNS,
    change_log_rows,
    generate,
)
from continuity.data.incidents import ChangeLogEntry, Effect, PlantedIncident

WINDOW_START = datetime(2026, 8, 3, 0, 0, tzinfo=UTC)  # Monday
SATURDAY_START = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)  # Saturday, same week

TITLES = generate_titles(np.random.default_rng(1), 20, as_of=WINDOW_START.date())
SUBSCRIBERS = generate_subscribers(np.random.default_rng(2), 100, as_of=WINDOW_START.date())


def _collect(**kwargs) -> dict[str, np.ndarray]:
    """Run generate() to exhaustion and concatenate every batch into one column dict."""
    kwargs.setdefault("titles", TITLES)
    kwargs.setdefault("subscribers", SUBSCRIBERS)
    kwargs.setdefault("incidents", ())
    kwargs.setdefault("batch_size", 20_000)
    columns: dict[str, list[np.ndarray]] = {c: [] for c in PLAYBACK_EVENTS_COLUMNS}
    batch_count = 0
    for batch in generate(**kwargs):
        batch_count += 1
        for c in PLAYBACK_EVENTS_COLUMNS:
            columns[c].append(np.asarray(batch[c], dtype=object))
    merged = {
        c: (np.concatenate(v) if v else np.array([], dtype=object)) for c, v in columns.items()
    }
    merged["_batch_count"] = batch_count
    return merged


def _roku_820_incident(
    start: datetime, hours: int, *, affected_fraction: float = 1.0
) -> PlantedIncident:
    return PlantedIncident(
        incident_id="INC-APP-ROKU-820",
        kind="device_app_fault",
        start=start,
        end=start + timedelta(hours=hours),
        predicate={"device_type": "roku", "app_version": "8.2.0"},
        affected_fraction=affected_fraction,
        effects=(Effect(metric="rebuffer", multiplier=4.5),),
        change=ChangeLogEntry(
            change_id=1,
            changed_at=start - timedelta(hours=1),
            change_type="app_release",
            component="roku_app",
            description="test",
            dimension_key="app_version",
            dimension_value="8.2.0",
        ),
    )


def _decoy_incident(start: datetime, hours: int, *, title_id: int) -> PlantedIncident:
    return PlantedIncident(
        incident_id=f"DECOY-PREMIERE-{title_id}",
        kind="decoy_premiere",
        start=start,
        end=start + timedelta(hours=hours),
        predicate={"title_id": str(title_id)},
        affected_fraction=1.0,
        effects=(),
        volume_multiplier=6.0,
        change=None,
        is_decoy=True,
    )


# --- Contract (1): weekday volume ------------------------------------------------


def test_weekend_session_volume_exceeds_midweek_volume():
    """expected_sessions() must be multiplied by weekday_factor or a Saturday shows the
    same volume as a Tuesday despite carrying higher load -- an incoherent model."""
    tuesday = _collect(
        seed=42, window_start=WINDOW_START + timedelta(days=1), days=1, sessions_per_day=4000
    )
    saturday = _collect(seed=42, window_start=SATURDAY_START, days=1, sessions_per_day=4000)

    tuesday_starts = int((tuesday["event_type"] == "start").sum())
    saturday_starts = int((saturday["event_type"] == "start").sum())
    assert saturday_starts > tuesday_starts


# --- Contract (2): event-type field discipline -----------------------------------


def test_startup_ms_is_nonzero_only_on_start_events():
    data = _collect(seed=1, window_start=WINDOW_START, days=1, sessions_per_day=1000)
    startup = data["startup_ms"].astype(np.int64)
    event_type = data["event_type"]

    assert (startup[event_type != "start"] == 0).all()
    assert (startup[event_type == "start"] > 0).any()


def test_bitrate_kbps_is_nonzero_only_on_heartbeat_events():
    data = _collect(seed=1, window_start=WINDOW_START, days=1, sessions_per_day=1000)
    bitrate = data["bitrate_kbps"].astype(np.int64)
    event_type = data["event_type"]

    assert (bitrate[event_type != "heartbeat"] == 0).all()
    assert (bitrate[event_type == "heartbeat"] > 0).any()


# --- Contract (3): determinism ---------------------------------------------------


def test_generation_is_byte_identical_for_the_same_seed():
    kwargs = dict(seed=777, window_start=WINDOW_START, days=1, sessions_per_day=1500)
    first = _collect(**kwargs)
    second = _collect(**kwargs)

    for col in PLAYBACK_EVENTS_COLUMNS:
        assert list(first[col]) == list(second[col]), f"column {col} differs across runs"


def test_generation_differs_for_a_different_seed():
    a = _collect(seed=1, window_start=WINDOW_START, days=1, sessions_per_day=1500)
    b = _collect(seed=2, window_start=WINDOW_START, days=1, sessions_per_day=1500)
    assert list(a["session_id"]) != list(b["session_id"])


# --- Contract (4): incident effects are measurable -------------------------------


def test_incident_matching_sessions_show_roughly_4_5x_rebuffer_of_baseline():
    """A session matching INC-APP-ROKU-820's predicate inside its window must show
    roughly 4.5x the rebuffer of an equivalent session outside it."""
    incident = _roku_820_incident(WINDOW_START, hours=24)
    data = _collect(
        seed=9, window_start=WINDOW_START, days=1, sessions_per_day=20_000, incidents=(incident,)
    )

    matching = (data["device_type"] == "roku") & (data["app_version"] == "8.2.0")
    matching_total, other_total = _rebuffer_totals_by_session_group(
        data["session_id"], data["rebuffer_ms"].astype(np.int64), matching
    )

    assert matching_total, "no matching sessions generated -- test setup is broken"
    assert other_total, "no baseline sessions generated -- test setup is broken"

    ratio = np.mean(matching_total) / np.mean(other_total)
    assert 3.0 <= ratio <= 6.5, f"expected roughly 4.5x rebuffer, got {ratio:.2f}x"


def _rebuffer_totals_by_session_group(
    session_id: np.ndarray, rebuffer_ms: np.ndarray, row_matches: np.ndarray
) -> tuple[list[float], list[float]]:
    """Single hash-based pass grouping rebuffer_ms by session, split by group membership.

    A row's `session_id` repeats across every event of that session, so one pass with a
    plain dict is O(n). `np.isin` over an object array of UUIDs is O(n*m) and unusable
    at this scale -- do not reintroduce it here.
    """
    totals: dict = {}
    is_matching_session: dict = {}
    for sid, amount, matches in zip(session_id, rebuffer_ms, row_matches, strict=True):
        totals[sid] = totals.get(sid, 0) + amount
        is_matching_session[sid] = bool(matches) or is_matching_session.get(sid, False)

    matching_totals = [totals[sid] for sid, m in is_matching_session.items() if m]
    other_totals = [totals[sid] for sid, m in is_matching_session.items() if not m]
    return matching_totals, other_totals


def test_session_matching_no_incident_predicate_has_baseline_qoe():
    """No incidents at all -> no session anywhere should carry an inflated rebuffer."""
    data = _collect(seed=11, window_start=WINDOW_START, days=1, sessions_per_day=5000, incidents=())
    rebuffer_ms = data["rebuffer_ms"].astype(np.int64)
    # Baseline rebuffer events are small exponential draws; none should resemble the
    # 4.5x-inflated magnitude an incident would produce.
    assert rebuffer_ms.max() < 15_000


def test_decoy_window_has_elevated_volume_with_qoe_inside_normal_bounds():
    """The decoy predicate carries no effects: volume should spike but QoE must stay
    within the same range as ordinary baseline traffic."""
    premiere_title_id = int(TITLES[0].title_id)
    decoy_start = WINDOW_START + timedelta(days=1)
    decoy = _decoy_incident(decoy_start, hours=24, title_id=premiere_title_id)

    data = _collect(
        seed=21, window_start=WINDOW_START, days=2, sessions_per_day=5000, incidents=(decoy,)
    )
    event_time = data["event_time"]
    title_id = data["title_id"].astype(np.int64)
    event_type = data["event_type"]

    in_decoy_day = (event_time >= np.datetime64(decoy_start.replace(tzinfo=None))) & (
        event_time < np.datetime64((decoy_start + timedelta(hours=24)).replace(tzinfo=None))
    )
    before_decoy_day = ~in_decoy_day

    premiere_starts_during = int(
        ((title_id == premiere_title_id) & (event_type == "start") & in_decoy_day).sum()
    )
    premiere_starts_before = int(
        ((title_id == premiere_title_id) & (event_type == "start") & before_decoy_day).sum()
    )
    assert premiere_starts_during > premiere_starts_before

    rebuffer_ms = data["rebuffer_ms"].astype(np.int64)
    premiere_mask = title_id == premiere_title_id
    other_mask = title_id != premiere_title_id
    if rebuffer_ms[premiere_mask].sum() > 0 and rebuffer_ms[other_mask].sum() > 0:
        premiere_avg = rebuffer_ms[premiere_mask].mean()
        other_avg = rebuffer_ms[other_mask].mean()
        assert premiere_avg < other_avg * 2.0


def test_encode_incident_reduces_bitrate_for_matching_title_only():
    """Cross-checks contract (5): if title_id predicate matching were broken (e.g. by
    comparing an int array to a str value) this effect would never apply."""
    title_id = int(TITLES[1].title_id)
    incident = PlantedIncident(
        incident_id=f"INC-ENCODE-{title_id}",
        kind="encode_fault",
        start=WINDOW_START,
        end=WINDOW_START + timedelta(hours=24),
        predicate={"title_id": str(title_id)},
        affected_fraction=1.0,
        effects=(Effect(metric="bitrate", multiplier=0.45),),
        change=ChangeLogEntry(
            change_id=3,
            changed_at=WINDOW_START - timedelta(hours=4),
            change_type="encode_pipeline",
            component="transcoder",
            description="test",
            dimension_key="title_id",
            dimension_value=str(title_id),
        ),
    )
    data = _collect(
        seed=5, window_start=WINDOW_START, days=1, sessions_per_day=10_000, incidents=(incident,)
    )
    bitrate = data["bitrate_kbps"].astype(np.int64)
    event_type = data["event_type"]
    row_title = data["title_id"].astype(np.int64)

    heartbeats = event_type == "heartbeat"
    matching = heartbeats & (row_title == title_id)
    other = heartbeats & (row_title != title_id)

    assert bitrate[matching].mean() < bitrate[other].mean()


# --- Contract (5): predicate values are strings ----------------------------------


def test_incident_matches_requires_predicate_values_as_strings_including_title_id():
    """incidents.PlantedIncident.matches() compares raw equality -- an int title_id
    from the generator would never match a string predicate value."""
    incident = PlantedIncident(
        incident_id="INC-ENCODE-9",
        kind="encode_fault",
        start=WINDOW_START,
        end=WINDOW_START + timedelta(hours=1),
        predicate={"title_id": "9"},
        affected_fraction=1.0,
        effects=(Effect(metric="bitrate", multiplier=0.45),),
    )
    when = WINDOW_START + timedelta(minutes=1)
    assert incident.matches({"title_id": 9}, when) is False
    assert incident.matches({"title_id": "9"}, when) is True


# --- Acceptance (4): dimension values only ever come from topology.py -----------


def test_generated_dimension_values_are_members_of_topology_and_pop_matches_cdn():
    data = _collect(seed=3, window_start=WINDOW_START, days=1, sessions_per_day=1500)

    assert set(data["cdn"].tolist()) <= set(topology.CDNS)
    assert set(data["device_type"].tolist()) <= set(topology.DEVICE_TYPES)
    for device in set(data["device_type"].tolist()):
        allowed = set(topology.app_versions_for(device))
        mask = data["device_type"] == device
        assert set(data["app_version"][mask].tolist()) <= allowed

    for cdn in set(data["cdn"].tolist()):
        allowed_pops = set(topology.pops_for(cdn))
        mask = data["cdn"] == cdn
        assert set(data["pop"][mask].tolist()) <= allowed_pops


# --- Edge cases -------------------------------------------------------------------


def test_days_zero_produces_no_batches_without_raising():
    batches = list(
        generate(
            seed=1,
            window_start=WINDOW_START,
            days=0,
            sessions_per_day=1000,
            titles=TITLES,
            subscribers=SUBSCRIBERS,
            incidents=(),
        )
    )
    assert batches == []


def test_sessions_per_day_zero_produces_no_batches_without_raising():
    batches = list(
        generate(
            seed=1,
            window_start=WINDOW_START,
            days=2,
            sessions_per_day=0,
            titles=TITLES,
            subscribers=SUBSCRIBERS,
            incidents=(),
        )
    )
    assert batches == []


def test_empty_incidents_tuple_works():
    data = _collect(seed=1, window_start=WINDOW_START, days=1, sessions_per_day=500, incidents=())
    assert len(data["event_time"]) > 0


def test_negative_days_raises():
    with pytest.raises(ValueError):
        list(
            generate(
                seed=1,
                window_start=WINDOW_START,
                days=-1,
                sessions_per_day=100,
                titles=TITLES,
                subscribers=SUBSCRIBERS,
            )
        )


def test_negative_sessions_per_day_raises():
    with pytest.raises(ValueError):
        list(
            generate(
                seed=1,
                window_start=WINDOW_START,
                days=1,
                sessions_per_day=-1,
                titles=TITLES,
                subscribers=SUBSCRIBERS,
            )
        )


def test_sessions_requested_with_empty_titles_raises():
    with pytest.raises(ValueError):
        list(
            generate(
                seed=1,
                window_start=WINDOW_START,
                days=1,
                sessions_per_day=100,
                titles=[],
                subscribers=SUBSCRIBERS,
            )
        )


def test_sessions_requested_with_empty_subscribers_raises():
    with pytest.raises(ValueError):
        list(
            generate(
                seed=1,
                window_start=WINDOW_START,
                days=1,
                sessions_per_day=100,
                titles=TITLES,
                subscribers=[],
            )
        )


# --- Session event-sequence coherence --------------------------------------------


def test_session_produces_one_start_n_heartbeat_and_one_end():
    data = _collect(seed=1, window_start=WINDOW_START, days=1, sessions_per_day=200)
    session_id = data["session_id"]
    event_type = data["event_type"]
    event_time = data["event_time"]

    some_session = session_id[0]
    mask = session_id == some_session
    types = event_type[mask]
    times = event_time[mask]
    order = np.argsort(times)
    ordered_types = types[order].tolist()

    assert ordered_types[0] == "start"
    assert ordered_types[-1] == "end"
    assert ordered_types.count("heartbeat") >= 1
    assert set(ordered_types) <= {"start", "heartbeat", "rebuffer", "error", "end"}
    assert ordered_types.count("start") == 1
    assert ordered_types.count("end") == 1


def test_row_values_have_schema_compatible_types():
    """Checks the raw batch as generate() yields it (before the object-dtype coercion
    `_collect` applies for convenience elsewhere in this file)."""
    batch = next(
        generate(
            seed=1,
            window_start=WINDOW_START,
            days=1,
            sessions_per_day=100,
            titles=TITLES,
            subscribers=SUBSCRIBERS,
            incidents=(),
        )
    )
    assert isinstance(batch["session_id"][0], uuid.UUID)
    assert isinstance(int(batch["subscriber_id"][0]), int)
    assert isinstance(int(batch["title_id"][0]), int)
    assert isinstance(batch["device_type"][0], str)
    assert isinstance(batch["event_type"][0], str)
    assert np.issubdtype(batch["event_time"].dtype, np.datetime64)


# --- Performance / memory ---------------------------------------------------------


def test_small_generation_completes_quickly():
    started = time.perf_counter()
    list(
        generate(
            seed=1,
            window_start=WINDOW_START,
            days=1,
            sessions_per_day=500,
            titles=TITLES,
            subscribers=SUBSCRIBERS,
            incidents=(),
        )
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0


def test_large_request_yields_more_than_one_batch():
    batches = list(
        generate(
            seed=1,
            window_start=WINDOW_START,
            days=2,
            sessions_per_day=8000,
            titles=TITLES,
            subscribers=SUBSCRIBERS,
            incidents=(),
            batch_size=5_000,
        )
    )
    assert len(batches) > 1
    for batch in batches[:-1]:
        assert len(batch["event_time"]) == 5_000


# --- Session length: variable, right-skewed heartbeat counts ---------------------


def _session_event_counts(
    session_id: np.ndarray, event_type: np.ndarray
) -> tuple[dict, dict, dict]:
    """Single O(n) hash-based pass -- see the np.isin warning on
    `_rebuffer_totals_by_session_group`."""
    heartbeat_counts: dict = {}
    start_counts: dict = {}
    end_counts: dict = {}
    for sid, et in zip(session_id, event_type, strict=True):
        if et == "heartbeat":
            heartbeat_counts[sid] = heartbeat_counts.get(sid, 0) + 1
        elif et == "start":
            start_counts[sid] = start_counts.get(sid, 0) + 1
        elif et == "end":
            end_counts[sid] = end_counts.get(sid, 0) + 1
    return heartbeat_counts, start_counts, end_counts


def test_session_heartbeat_counts_genuinely_vary_and_are_right_skewed():
    """A fixed heartbeat count would give every session identical watched_ms, which is
    both a visible synthetic-data tell and makes 'watch-time lost' a trivial constant
    multiple of session count. The distribution must have real spread and skew right
    (median below mean), not just take more than one distinct value by accident."""
    data = _collect(seed=31, window_start=WINDOW_START, days=1, sessions_per_day=4000)
    heartbeat_counts, _, _ = _session_event_counts(data["session_id"], data["event_type"])
    counts = np.array(list(heartbeat_counts.values()))

    assert len(set(counts.tolist())) >= 5, "heartbeat counts barely vary across sessions"
    assert counts.min() >= 1, "minimum of 1 heartbeat must be respected"
    assert np.median(counts) < np.mean(counts), "distribution should be right-skewed"


def test_movie_sessions_run_longer_than_series_sessions_on_average():
    """Content type should influence session length where it is cheap to do so."""
    rng = np.random.default_rng(99)
    titles = generate_titles(rng, 200, as_of=WINDOW_START.date())
    movie_ids = {t.title_id for t in titles if t.content_type == "movie"}
    series_ids = {t.title_id for t in titles if t.content_type == "series"}
    assert movie_ids and series_ids, "test fixture needs both content types present"

    data = _collect(
        seed=32, window_start=WINDOW_START, days=1, sessions_per_day=6000, titles=titles
    )
    heartbeat_counts, _, _ = _session_event_counts(data["session_id"], data["event_type"])
    title_by_session: dict = {}
    for sid, title in zip(data["session_id"], data["title_id"].astype(np.int64), strict=True):
        title_by_session.setdefault(sid, title)

    movie_lengths = [
        count for sid, count in heartbeat_counts.items() if title_by_session[sid] in movie_ids
    ]
    series_lengths = [
        count for sid, count in heartbeat_counts.items() if title_by_session[sid] in series_ids
    ]
    assert movie_lengths and series_lengths
    assert np.mean(movie_lengths) > np.mean(series_lengths)


def test_every_session_still_has_exactly_one_start_and_one_end():
    data = _collect(seed=33, window_start=WINDOW_START, days=1, sessions_per_day=3000)
    _, start_counts, end_counts = _session_event_counts(data["session_id"], data["event_type"])

    all_sessions = set(data["session_id"].tolist())
    assert set(start_counts) == all_sessions
    assert set(end_counts) == all_sessions
    assert all(count == 1 for count in start_counts.values())
    assert all(count == 1 for count in end_counts.values())


# --- change_log_rows ---------------------------------------------------------------


def test_change_log_rows_are_generated_from_incidents_with_a_change_entry():
    incident_with_change = _roku_820_incident(WINDOW_START, hours=8)
    decoy = _decoy_incident(WINDOW_START, hours=5, title_id=999)

    rows = change_log_rows((incident_with_change, decoy))
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(CHANGE_LOG_COLUMNS)
    assert row["change_id"] == incident_with_change.change.change_id
    assert row["dimension_value"] == "8.2.0"


def test_change_log_rows_empty_when_no_incident_has_a_change():
    decoy = _decoy_incident(WINDOW_START, hours=5, title_id=999)
    assert change_log_rows((decoy,)) == []
