"""Reads `data/ground_truth.json` for the incident list `/api/incidents` serves.

This is a read-only reshaping of the same file `continuity/analysis/cli.py`'s own
`--incident` flag reads (see its `_load_ground_truth`) -- into the small summary shape
the UI needs: id, window, predicate, decoy flag. No dates are hardcoded anywhere; every
value in the returned summaries comes straight from the file on disk.

Deliberately does not import from `continuity.analysis.cli` even though that module has
an equivalent loader: that loader raises `InvestigationInputError`, a type owned by the
analysis package, and importing analysis-internal exception types into the API layer for
a two-line JSON read is not worth the coupling. The two loaders read the same file and
would need to change together only if the ground-truth schema itself changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Matches continuity/analysis/cli.py::DEFAULT_GROUND_TRUTH_PATH -- both resolve relative
# to the process's working directory, which is the repo root in every documented way
# this project is run (uv run uvicorn ..., or WORKDIR /app in the container).
DEFAULT_GROUND_TRUTH_PATH = Path("data/ground_truth.json")


class GroundTruthError(RuntimeError):
    """The ground truth file is missing or malformed. Never swallowed into an empty list --
    an empty incident feed and a broken file look identical to a user unless this raises."""


def load_incident_summaries(path: Path = DEFAULT_GROUND_TRUTH_PATH) -> list[dict[str, Any]]:
    """The incident feed `/api/incidents` returns: id, window, predicate, decoy flag.

    Ground truth's `predicate` (the blast-radius dimensions) is exposed here for the UI
    to display -- unlike the deterministic pipeline's own `walk()`, which must never be
    given the predicate as a shortcut. This endpoint only lists incidents; it does not
    run or bias the investigation.
    """
    if not path.exists():
        raise GroundTruthError(f"Ground truth file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GroundTruthError(f"Ground truth file {path} is not valid JSON: {exc}") from exc

    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        raise GroundTruthError(f"Ground truth file {path} has no 'incidents' list.")

    return [
        {
            "id": row["incident_id"],
            "window": {"start": row["start"], "end": row["end"]},
            "predicate": row.get("predicate", {}),
            "is_decoy": bool(row.get("is_decoy", False)),
        }
        for row in incidents
    ]
