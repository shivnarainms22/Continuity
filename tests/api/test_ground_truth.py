"""Unit tests for continuity.api.ground_truth -- pure file I/O, no ClickHouse needed."""

from __future__ import annotations

import json

import pytest

from continuity.api.ground_truth import GroundTruthError, load_incident_summaries

_SAMPLE = {
    "incidents": [
        {
            "incident_id": "INC-TEST-1",
            "kind": "device_app_fault",
            "start": "2026-03-01T00:00:00+00:00",
            "end": "2026-03-01T04:00:00+00:00",
            "predicate": {"device_type": "roku"},
            "is_decoy": False,
        },
        {
            "incident_id": "DECOY-TEST-2",
            "kind": "decoy_premiere",
            "start": "2026-03-05T00:00:00+00:00",
            "end": "2026-03-05T04:00:00+00:00",
            "predicate": {"title_id": "3"},
            "is_decoy": True,
        },
    ]
}


def test_returns_id_window_predicate_kind_and_decoy_flag_for_every_incident(tmp_path):
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(_SAMPLE), encoding="utf-8")

    summaries = load_incident_summaries(path)

    assert summaries == [
        {
            "id": "INC-TEST-1",
            "window": {"start": "2026-03-01T00:00:00+00:00", "end": "2026-03-01T04:00:00+00:00"},
            "predicate": {"device_type": "roku"},
            "kind": "device_app_fault",
            "is_decoy": False,
        },
        {
            "id": "DECOY-TEST-2",
            "window": {"start": "2026-03-05T00:00:00+00:00", "end": "2026-03-05T04:00:00+00:00"},
            "predicate": {"title_id": "3"},
            "kind": "decoy_premiere",
            "is_decoy": True,
        },
    ]


def test_missing_predicate_defaults_to_empty_dict(tmp_path):
    payload = {
        "incidents": [
            {"incident_id": "X", "start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00"}
        ]
    }
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summaries = load_incident_summaries(path)

    assert summaries[0]["predicate"] == {}
    assert summaries[0]["is_decoy"] is False
    assert summaries[0]["kind"] is None


def test_missing_file_raises_ground_truth_error(tmp_path):
    with pytest.raises(GroundTruthError, match="not found"):
        load_incident_summaries(tmp_path / "does-not-exist.json")


def test_malformed_json_raises_ground_truth_error(tmp_path):
    path = tmp_path / "ground_truth.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(GroundTruthError, match="not valid JSON"):
        load_incident_summaries(path)


def test_missing_incidents_key_raises_ground_truth_error(tmp_path):
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps({"seed": 1}), encoding="utf-8")

    with pytest.raises(GroundTruthError, match="'incidents' list"):
        load_incident_summaries(path)


def test_empty_incidents_list_returns_empty_list(tmp_path):
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps({"incidents": []}), encoding="utf-8")

    assert load_incident_summaries(path) == []
