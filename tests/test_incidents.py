import json
from datetime import UTC, datetime, timedelta

import pytest

from continuity.data import topology
from continuity.data.incidents import (
    ChangeLogEntry,
    Effect,
    PlantedIncident,
    build_incidents,
    write_ground_truth,
)

WINDOW_START = datetime(2026, 8, 1, tzinfo=UTC)
DAYS = 56  # matches continuity.data.load.DEFAULT_DAYS: 4+ prior weeks per incident
PREMIERE_TITLE_ID = 4001
ENCODE_TITLE_ID = 4002


@pytest.fixture
def incidents() -> tuple[PlantedIncident, ...]:
    return build_incidents(
        WINDOW_START,
        DAYS,
        premiere_title_id=PREMIERE_TITLE_ID,
        encode_title_id=ENCODE_TITLE_ID,
    )


def _by_id(incidents: tuple[PlantedIncident, ...], incident_id: str) -> PlantedIncident:
    for inc in incidents:
        if inc.incident_id == incident_id:
            return inc
    raise AssertionError(f"no incident with id {incident_id!r}")


def test_build_incidents_returns_the_four_planted_incidents(incidents):
    ids = {inc.incident_id for inc in incidents}
    assert ids == {
        "INC-APP-ROKU-820",
        "INC-POP-NW-ATL-2",
        f"INC-ENCODE-{ENCODE_TITLE_ID}",
        f"DECOY-PREMIERE-{PREMIERE_TITLE_ID}",
    }


def test_roku_820_predicate_is_not_separable_by_either_dimension_alone(incidents):
    """Load-bearing property (a): neither device_type=roku nor app_version=8.2.0 alone
    identifies the incident's true blast radius -- both dimensions are required."""
    inc = _by_id(incidents, "INC-APP-ROKU-820")
    assert inc.predicate == {"device_type": "roku", "app_version": "8.2.0"}

    roku_versions = topology.app_versions_for("roku")
    assert "8.2.0" in roku_versions
    assert set(roku_versions) - {"8.2.0"}, "roku must run other app_versions too"

    other_devices_on_820 = [
        device
        for device in topology.DEVICE_TYPES
        if device != "roku" and "8.2.0" in topology.app_versions_for(device)
    ]
    assert other_devices_on_820, "8.2.0 must also ship on non-roku devices"


def test_incident_windows_are_relative_to_window_start(incidents):
    """Incidents are anchored to the END of the window (days - offset), not the start --
    see build_incidents' docstring. This gives every incident at least 4 prior weeks of
    same-weekday history for the week-over-week baseline, regardless of window length."""
    app = _by_id(incidents, "INC-APP-ROKU-820")
    assert app.start == WINDOW_START + timedelta(days=DAYS - 14, hours=18)
    assert app.end == app.start + timedelta(hours=8)

    pop = _by_id(incidents, "INC-POP-NW-ATL-2")
    assert pop.start == WINDOW_START + timedelta(days=DAYS - 11, hours=2)
    assert pop.end == pop.start + timedelta(hours=6)

    encode = _by_id(incidents, f"INC-ENCODE-{ENCODE_TITLE_ID}")
    assert encode.start == WINDOW_START + timedelta(days=DAYS - 8, hours=9)
    assert encode.end == encode.start + timedelta(hours=30)

    decoy = _by_id(incidents, f"DECOY-PREMIERE-{PREMIERE_TITLE_ID}")
    assert decoy.start == WINDOW_START + timedelta(days=DAYS - 6, hours=20)
    assert decoy.end == decoy.start + timedelta(hours=5)


def test_every_incident_has_at_least_four_prior_same_weekday_weeks_within_the_window():
    """The coupled reason incidents moved to end-of-window offsets: with the default
    week-over-week baseline (K=4 weeks lookback), every incident's start must have at
    least 4 same-weekday samples strictly before it and still inside [window_start,
    window_start + days)."""
    incidents = build_incidents(
        WINDOW_START, DAYS, premiere_title_id=PREMIERE_TITLE_ID, encode_title_id=ENCODE_TITLE_ID
    )
    for inc in incidents:
        earliest_needed = inc.start - timedelta(weeks=4)
        assert earliest_needed >= WINDOW_START, (
            f"{inc.incident_id} starts at {inc.start} but only has "
            f"{(inc.start - WINDOW_START).days // 7} prior weeks within the window"
        )


def test_matches_true_for_full_predicate_match_within_window(incidents):
    app = _by_id(incidents, "INC-APP-ROKU-820")
    when = app.start + timedelta(hours=1)
    assert app.matches({"device_type": "roku", "app_version": "8.2.0"}, when) is True


def test_matches_false_before_window_start(incidents):
    app = _by_id(incidents, "INC-APP-ROKU-820")
    when = app.start - timedelta(seconds=1)
    assert app.matches({"device_type": "roku", "app_version": "8.2.0"}, when) is False


def test_matches_false_after_window_end(incidents):
    app = _by_id(incidents, "INC-APP-ROKU-820")
    when = app.end + timedelta(seconds=1)
    assert app.matches({"device_type": "roku", "app_version": "8.2.0"}, when) is False


def test_matches_true_at_exact_start_boundary(incidents):
    app = _by_id(incidents, "INC-APP-ROKU-820")
    assert app.matches({"device_type": "roku", "app_version": "8.2.0"}, app.start) is True


def test_matches_false_at_exact_end_boundary(incidents):
    """The window is half-open [start, end) -- the end instant itself is outside it."""
    app = _by_id(incidents, "INC-APP-ROKU-820")
    assert app.matches({"device_type": "roku", "app_version": "8.2.0"}, app.end) is False


def test_matches_false_when_any_predicate_key_mismatches(incidents):
    app = _by_id(incidents, "INC-APP-ROKU-820")
    when = app.start + timedelta(hours=1)
    assert app.matches({"device_type": "firetv", "app_version": "8.2.0"}, when) is False
    assert app.matches({"device_type": "roku", "app_version": "8.1.4"}, when) is False


def test_matches_false_when_predicate_key_absent_from_dims(incidents):
    app = _by_id(incidents, "INC-APP-ROKU-820")
    when = app.start + timedelta(hours=1)
    assert app.matches({"device_type": "roku"}, when) is False


def test_every_real_incident_has_effects_and_a_prior_change_entry(incidents):
    real = [inc for inc in incidents if not inc.is_decoy]
    assert len(real) == 3
    for inc in real:
        assert len(inc.effects) >= 1
        assert inc.change is not None
        assert inc.change.changed_at < inc.start


def test_decoy_has_no_effects_elevated_volume_no_change_and_is_flagged(incidents):
    decoy = _by_id(incidents, f"DECOY-PREMIERE-{PREMIERE_TITLE_ID}")
    assert decoy.effects == ()
    assert decoy.volume_multiplier > 1
    assert decoy.change is None
    assert decoy.is_decoy is True


def test_predicate_keys_are_valid_dimensions_or_title_id(incidents):
    allowed = set(topology.DIMENSION_HIERARCHY) | {"title_id"}
    for inc in incidents:
        assert inc.predicate, f"{inc.incident_id} has an empty predicate"
        for key in inc.predicate:
            assert key in allowed, f"{inc.incident_id} predicate key {key!r} is not real"


def test_affected_fraction_is_within_zero_exclusive_one_inclusive(incidents):
    for inc in incidents:
        assert 0 < inc.affected_fraction <= 1


def test_write_ground_truth_round_trips_through_json(tmp_path, incidents):
    path = tmp_path / "ground_truth.json"
    write_ground_truth(incidents, path, seed=20260908, days=DAYS)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 20260908
    assert payload["days"] == DAYS

    rebuilt = []
    for row in payload["incidents"]:
        change = row["change"]
        rebuilt.append(
            PlantedIncident(
                incident_id=row["incident_id"],
                kind=row["kind"],
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
                predicate=row["predicate"],
                affected_fraction=row["affected_fraction"],
                effects=tuple(Effect(**e) for e in row["effects"]),
                volume_multiplier=row["volume_multiplier"],
                change=(
                    ChangeLogEntry(
                        change_id=change["change_id"],
                        changed_at=datetime.fromisoformat(change["changed_at"]),
                        change_type=change["change_type"],
                        component=change["component"],
                        description=change["description"],
                        dimension_key=change["dimension_key"],
                        dimension_value=change["dimension_value"],
                    )
                    if change is not None
                    else None
                ),
                is_decoy=row["is_decoy"],
            )
        )

    assert tuple(rebuilt) == incidents


def test_ground_truth_json_contains_the_true_predicates(tmp_path, incidents):
    path = tmp_path / "ground_truth.json"
    write_ground_truth(incidents, path, seed=20260908, days=DAYS)

    payload = json.loads(path.read_text(encoding="utf-8"))
    serialised_predicates = {row["incident_id"]: row["predicate"] for row in payload["incidents"]}
    for inc in incidents:
        assert serialised_predicates[inc.incident_id] == inc.predicate
