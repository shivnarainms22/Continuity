"""Planted incident ground truth.

Pure apart from `write_ground_truth`: no network, no ClickHouse, no global random state.
This module is the ground truth the entire evaluation harness scores the agent against.
Constraint (see CLAUDE.md): ground truth NEVER enters ClickHouse. `write_ground_truth`
writes only to a local JSON file, read solely by the eval harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Effect:
    metric: str
    multiplier: float


@dataclass(frozen=True)
class ChangeLogEntry:
    change_id: int
    changed_at: datetime
    change_type: str
    component: str
    description: str
    dimension_key: str
    dimension_value: str


@dataclass(frozen=True)
class PlantedIncident:
    incident_id: str
    kind: str
    start: datetime
    end: datetime
    predicate: dict[str, str]  # the TRUE blast radius
    affected_fraction: float  # share of matching sessions actually hit
    effects: tuple[Effect, ...] = field(default_factory=tuple)
    volume_multiplier: float = 1.0
    change: ChangeLogEntry | None = None
    is_decoy: bool = False

    def matches(self, dims: dict[str, str], when: datetime) -> bool:
        """True iff `when` falls in the half-open window [start, end) and every
        predicate key is present in `dims` with the matching value."""
        if not (self.start <= when < self.end):
            return False
        return all(dims.get(key) == value for key, value in self.predicate.items())


def build_incidents(
    window_start: datetime,
    days: int,
    *,
    premiere_title_id: int,
    encode_title_id: int,
) -> tuple[PlantedIncident, ...]:
    """The four planted incidents: three real, one decoy.

    INC-APP-ROKU-820 is scoped to device_type=roku AND app_version=8.2.0. Neither
    dimension alone identifies it -- 8.2.0 also ships on firetv/ios/android, and roku
    also runs 8.0.9/8.1.4 -- so a correct investigation must drill down on both.
    """
    app_start = window_start + timedelta(days=12, hours=18)
    app_incident = PlantedIncident(
        incident_id="INC-APP-ROKU-820",
        kind="device_app_fault",
        start=app_start,
        end=app_start + timedelta(hours=8),
        predicate={"device_type": "roku", "app_version": "8.2.0"},
        affected_fraction=0.85,
        effects=(Effect(metric="rebuffer", multiplier=4.5),),
        change=ChangeLogEntry(
            change_id=1,
            changed_at=app_start - timedelta(hours=3),
            change_type="app_release",
            component="roku_app",
            description="Roku app 8.2.0 rollout increased rebuffer rate",
            dimension_key="app_version",
            dimension_value="8.2.0",
        ),
    )

    pop_start = window_start + timedelta(days=15, hours=2)
    pop_incident = PlantedIncident(
        incident_id="INC-POP-NW-ATL-2",
        kind="pop_fault",
        start=pop_start,
        end=pop_start + timedelta(hours=6),
        predicate={"cdn": "cdn_northwind", "pop": "nw-atl-2"},
        affected_fraction=0.9,
        effects=(
            Effect(metric="startup", multiplier=3.2),
            Effect(metric="rebuffer", multiplier=2.0),
        ),
        change=ChangeLogEntry(
            change_id=2,
            changed_at=pop_start - timedelta(hours=1),
            change_type="network_config",
            component="cdn_northwind_pop",
            description="BGP reroute at nw-atl-2 degraded startup latency",
            dimension_key="pop",
            dimension_value="nw-atl-2",
        ),
    )

    encode_start = window_start + timedelta(days=18, hours=9)
    encode_incident = PlantedIncident(
        incident_id=f"INC-ENCODE-{encode_title_id}",
        kind="encode_fault",
        start=encode_start,
        end=encode_start + timedelta(hours=30),
        predicate={"title_id": str(encode_title_id)},
        affected_fraction=1.0,
        effects=(
            Effect(metric="bitrate", multiplier=0.45),
            Effect(metric="rebuffer", multiplier=2.5),
        ),
        change=ChangeLogEntry(
            change_id=3,
            changed_at=encode_start - timedelta(hours=4),
            change_type="encode_pipeline",
            component="transcoder",
            description=f"Re-encode of title {encode_title_id} shipped with lower bitrate ladder",
            dimension_key="title_id",
            dimension_value=str(encode_title_id),
        ),
    )

    decoy_start = window_start + timedelta(days=20, hours=20)
    decoy_incident = PlantedIncident(
        incident_id=f"DECOY-PREMIERE-{premiere_title_id}",
        kind="decoy_premiere",
        start=decoy_start,
        end=decoy_start + timedelta(hours=5),
        predicate={"title_id": str(premiere_title_id)},
        affected_fraction=1.0,
        effects=(),
        volume_multiplier=6.0,
        change=None,
        is_decoy=True,
    )

    return (app_incident, pop_incident, encode_incident, decoy_incident)


def write_ground_truth(
    incidents: tuple[PlantedIncident, ...],
    path: Path,
    *,
    seed: int,
    days: int,
) -> None:
    """Serialize ground truth to JSON. The only I/O in this module. Never call this
    with a ClickHouse-backed path -- ground truth must never enter the database."""
    payload = {
        "seed": seed,
        "days": days,
        "incidents": [_incident_to_dict(inc) for inc in incidents],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _incident_to_dict(incident: PlantedIncident) -> dict:
    return {
        "incident_id": incident.incident_id,
        "kind": incident.kind,
        "start": incident.start.isoformat(),
        "end": incident.end.isoformat(),
        "predicate": incident.predicate,
        "affected_fraction": incident.affected_fraction,
        "effects": [{"metric": e.metric, "multiplier": e.multiplier} for e in incident.effects],
        "volume_multiplier": incident.volume_multiplier,
        "change": _change_to_dict(incident.change),
        "is_decoy": incident.is_decoy,
    }


def _change_to_dict(change: ChangeLogEntry | None) -> dict | None:
    if change is None:
        return None
    return {
        "change_id": change.change_id,
        "changed_at": change.changed_at.isoformat(),
        "change_type": change.change_type,
        "component": change.component,
        "description": change.description,
        "dimension_key": change.dimension_key,
        "dimension_value": change.dimension_value,
    }
