"""Synthetic title and subscriber catalogs.

Pure module: no I/O, no network, no global random state. All randomness is drawn from
an injected `numpy.random.Generator` so a fixed seed reproduces byte-identical output --
the eval harness depends on that.

Money is represented as `decimal.Decimal`, not `float`. `monthly_arpu` feeds a revenue
calculation downstream and float drift (e.g. 15.99 not round-tripping exactly) is
unacceptable for money; `Decimal` with a fixed quantization matches the ClickHouse
`Decimal(8, 2)` column exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from continuity.data.topology import GEO

PLAN_ARPU: dict[str, Decimal] = {
    "basic": Decimal("8.99"),
    "standard": Decimal("15.99"),
    "premium": Decimal("22.99"),
}

_PLAN_NAMES: tuple[str, ...] = ("basic", "standard", "premium")
_PLAN_WEIGHTS: tuple[float, ...] = (0.35, 0.45, 0.20)

_GENRES: tuple[str, ...] = (
    "drama",
    "comedy",
    "action",
    "documentary",
    "thriller",
    "animation",
    "reality",
)
_CONTENT_TYPES: tuple[str, ...] = ("movie", "series", "special")

# Tenure is drawn from an exponential distribution truncated to this many days, so the
# subscriber base skews toward recent signups. This matters for sub-project 2: the churn
# heuristic weights low-tenure subscribers as higher risk, and a uniform tenure spread
# would make that heuristic meaningless.
_TENURE_MAX_DAYS = 1500
_TENURE_SCALE_DAYS = 300.0

# Titles are backdated up to this many days before `as_of`-equivalent generation.
_TITLE_MAX_AGE_DAYS = 2000
_TITLE_SCALE_DAYS = 400.0
_PREMIERE_PROBABILITY = 0.08


@dataclass(frozen=True)
class Title:
    title_id: int
    name: str
    genre: str
    content_type: str
    release_date: date
    is_premiere: bool


@dataclass(frozen=True)
class Subscriber:
    subscriber_id: int
    plan: str
    monthly_arpu: Decimal
    signup_date: date
    tenure_days: int
    country: str
    region: str


def generate_titles(rng: np.random.Generator, count: int, *, as_of: date) -> list[Title]:
    """Generate `count` titles with unique ids, deterministic under a fixed `rng` seed.

    `release_date` is never after `as_of`. A minority of titles are flagged
    `is_premiere` (recent release, elevated draw).

    `as_of` is required rather than defaulting to today: a wall-clock default would make
    output depend on the day it ran, and the eval harness requires regeneration from a
    fixed seed to be byte-identical. This mirrors `generate_subscribers`.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []

    genres = rng.choice(_GENRES, size=count)
    content_types = rng.choice(_CONTENT_TYPES, size=count)
    # Exponential-shaped age so most titles are relatively recent, matching a real catalog.
    ages_days = np.minimum(
        rng.exponential(scale=_TITLE_SCALE_DAYS, size=count).astype(int),
        _TITLE_MAX_AGE_DAYS,
    )
    premiere_draw = rng.random(count) < _PREMIERE_PROBABILITY

    titles = []
    for i in range(count):
        age_days = int(ages_days[i])
        is_premiere = bool(premiere_draw[i])
        # A premiere title is recent by definition -- clamp its age to the last 14 days.
        if is_premiere:
            age_days = min(age_days, 14)
        titles.append(
            Title(
                title_id=i + 1,
                name=f"Title {i + 1:05d}",
                genre=str(genres[i]),
                content_type=str(content_types[i]),
                release_date=as_of - timedelta(days=age_days),
                is_premiere=is_premiere,
            )
        )
    return titles


def generate_subscribers(rng: np.random.Generator, count: int, *, as_of: date) -> list[Subscriber]:
    """Generate `count` subscribers with unique ids, deterministic under a fixed `rng` seed.

    Plan mix is roughly 35/45/20 basic/standard/premium. `monthly_arpu` always matches the
    plan exactly (see `PLAN_ARPU`). `tenure_days` is drawn from an exponential distribution
    skewed toward newer subscribers and is always consistent with
    `(as_of - signup_date).days`. `(country, region)` pairs are drawn from
    `continuity.data.topology.GEO` so downstream joins line up.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []

    plans = rng.choice(_PLAN_NAMES, size=count, p=_PLAN_WEIGHTS)
    tenure_days = np.minimum(
        rng.exponential(scale=_TENURE_SCALE_DAYS, size=count).astype(int),
        _TENURE_MAX_DAYS,
    )
    geo_indices = rng.integers(0, len(GEO), size=count)

    subscribers = []
    for i in range(count):
        plan = str(plans[i])
        tenure = int(tenure_days[i])
        country, region = GEO[int(geo_indices[i])]
        subscribers.append(
            Subscriber(
                subscriber_id=i + 1,
                plan=plan,
                monthly_arpu=PLAN_ARPU[plan],
                signup_date=as_of - timedelta(days=tenure),
                tenure_days=tenure,
                country=country,
                region=region,
            )
        )
    return subscribers
