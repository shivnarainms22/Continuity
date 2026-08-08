"""Diurnal/weekly seasonality and load-correlated QoE degradation.

Pure functions only -- no I/O, no global random state, no randomness of any kind here.
Any randomness needed downstream must come from an injected `numpy.random.Generator`.

`degrade` is the load -> QoE coupling that makes rebuffering genuinely worse at peak
concurrency, exactly as it is in real streaming systems. This is deliberate: it is what
makes a naive fixed-threshold anomaly detector fire every single night at 21:00, which is
the real false-positive problem ops teams have, and it is what forces the
seasonality-aware baseline in the next sub-project to be real work rather than
decoration. Do not flatten this curve.
"""

from __future__ import annotations

import math
from datetime import datetime

# Traffic peaks at prime time (21:00) and troughs twelve hours away, mid-morning (09:00).
# The baseline keeps every raw weight strictly positive; the cosine term drives a 5x raw
# peak/trough spread, which `degrade` below turns into a ~2x QoE difference.
_PEAK_HOUR = 21
_DIURNAL_BASELINE = 1.5


def _raw_diurnal(hour: int) -> float:
    return _DIURNAL_BASELINE + math.cos(2 * math.pi * (hour - _PEAK_HOUR) / 24)


_raw_diurnal_weights = tuple(_raw_diurnal(h) for h in range(24))
_raw_diurnal_total = sum(_raw_diurnal_weights)

DIURNAL_WEIGHTS: tuple[float, ...] = tuple(w / _raw_diurnal_total for w in _raw_diurnal_weights)


def diurnal_weight(hour: int) -> float:
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be in 0..23, got {hour}")
    return DIURNAL_WEIGHTS[hour]


# Mon=0 .. Sun=6, matching datetime.weekday(). Weekend evenings draw more concurrent
# viewers than a midweek evening; Friday is a smaller step toward the weekend.
WEEKDAY_FACTORS: tuple[float, ...] = (0.95, 0.95, 0.95, 0.95, 1.05, 1.25, 1.25)


def weekday_factor(weekday: int) -> float:
    if not 0 <= weekday <= 6:
        raise ValueError(f"weekday must be in 0..6 (Mon..Sun), got {weekday}")
    return WEEKDAY_FACTORS[weekday]


def expected_sessions(bucket: datetime, sessions_per_day: int) -> float:
    """Expected new sessions in the 5-minute bucket starting at `bucket`.

    Only hour-of-day shapes the intra-day curve. DIURNAL_WEIGHTS sums to 1.0 over 24
    hours and each hour holds exactly 12 five-minute buckets, so summing this function
    over all 288 buckets of a day recovers `sessions_per_day` (up to float rounding).

    CONTRACT WITH THE GENERATOR (continuity/data/generator.py):

    This deliberately applies only the intra-day shape, NOT `weekday_factor`, so that
    summing over a day's 288 buckets lands exactly on `sessions_per_day` whichever
    weekday is chosen. `sessions_per_day` therefore means "volume on a nominal weekday".

    The caller MUST multiply the result by `weekday_factor(bucket.weekday())`. Skipping
    that leaves the model incoherent: `load_factor` already includes the weekday term, so
    a Saturday would show 25% higher load -- and therefore worse QoE -- while carrying
    exactly the same session volume as a Tuesday. Load represents concurrency, and
    concurrency comes from volume; the two must move together.
    """
    if sessions_per_day < 0:
        raise ValueError(f"sessions_per_day must be >= 0, got {sessions_per_day}")
    return sessions_per_day * diurnal_weight(bucket.hour) / 12


_MAX_RAW_LOAD = max(DIURNAL_WEIGHTS) * max(WEEKDAY_FACTORS)


def load_factor(bucket: datetime) -> float:
    """Normalised concurrency in [0.0, 1.0]. 1.0 is peak hour on the highest-traffic day."""
    raw = diurnal_weight(bucket.hour) * weekday_factor(bucket.weekday())
    return raw / _MAX_RAW_LOAD


def degrade(base: float, load: float, *, alpha: float = 1.2, beta: float = 2.0) -> float:
    """Load-coupled QoE degradation: base * (1 + alpha * load ** beta).

    Equals `base` at load=0, is monotonically non-decreasing in load, and never returns
    less than `base`. This coupling is the point of the module -- do not flatten it.
    """
    if base < 0:
        raise ValueError(f"base must be >= 0, got {base}")
    if not 0.0 <= load <= 1.0:
        raise ValueError(f"load must be in [0.0, 1.0], got {load}")
    return base * (1 + alpha * load**beta)
