"""Synthetic playback-event stream assembly.

Pure module: no I/O, no network, no ClickHouse. All randomness comes from a single
`numpy.random.Generator` seeded by the caller, consumed in a fixed order (bucket by
bucket, dimension by dimension), so the same seed reproduces byte-identical output --
the eval harness depends on that.

Vectorised with NumPy per 5-minute bucket. The only Python-level loops are over small,
bounded collections (device types, CDNs, active incidents, event-type groups) -- never
over individual sessions or events -- so this scales to tens of millions of rows without
building multi-million-element Python lists.

CONTRACT WITH seasonality.expected_sessions(): that function applies only the intra-day
shape and deliberately omits the weekday term (see its docstring). This module supplies
the missing `weekday_factor(bucket.weekday())` multiplier, because `load_factor` already
includes the weekday term -- omitting it here would make a Saturday show higher load and
worse QoE than a Tuesday on identical session volume.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta

import numpy as np

from continuity.data import seasonality, topology
from continuity.data.catalog import Subscriber, Title
from continuity.data.incidents import PlantedIncident

# Column order matches continuity/data/schema.py's PLAYBACK_EVENTS DDL exactly.
PLAYBACK_EVENTS_COLUMNS: tuple[str, ...] = (
    "event_time",
    "session_id",
    "subscriber_id",
    "title_id",
    "device_type",
    "os_version",
    "app_version",
    "cdn",
    "pop",
    "isp",
    "country",
    "region",
    "event_type",
    "watched_ms",
    "rebuffer_ms",
    "startup_ms",
    "bitrate_kbps",
    "error_code",
)

# Column order matches continuity/data/schema.py's CHANGE_LOG DDL exactly.
CHANGE_LOG_COLUMNS: tuple[str, ...] = (
    "change_id",
    "changed_at",
    "change_type",
    "component",
    "description",
    "dimension_key",
    "dimension_value",
)

BUCKETS_PER_DAY = 288
BUCKET_MINUTES = 5
BUCKET_MS = BUCKET_MINUTES * 60 * 1000
HEARTBEAT_INTERVAL_MS = 30_000
HEARTBEATS_PER_SESSION = 6
SESSION_DURATION_MS = (HEARTBEATS_PER_SESSION + 1) * HEARTBEAT_INTERVAL_MS

# Baseline QoE, before the load coupling and any incident effect are applied.
BASE_REBUFFER_PROB = 0.12
BASE_REBUFFER_MEAN_MS = 800.0
BASE_STARTUP_MEAN_MS = 900.0
BASE_BITRATE_KBPS = 4500.0
BASE_ERROR_PROB = 0.01

ERROR_CODES: tuple[str, ...] = ("ERR_DECODE", "ERR_NETWORK", "ERR_DRM", "ERR_TIMEOUT")

_DEVICE_TYPE_ARR = np.array(topology.DEVICE_TYPES, dtype=object)
_DEVICE_WEIGHT_ARR = np.array([topology.DEVICE_WEIGHTS[d] for d in topology.DEVICE_TYPES])
_DEVICE_WEIGHT_ARR = _DEVICE_WEIGHT_ARR / _DEVICE_WEIGHT_ARR.sum()
_CDN_ARR = np.array(topology.CDNS, dtype=object)
_ISP_ARR = np.array(topology.ISPS, dtype=object)
_GEO_COUNTRY_ARR = np.array([c for c, _ in topology.GEO], dtype=object)
_GEO_REGION_ARR = np.array([r for _, r in topology.GEO], dtype=object)
_ERROR_CODE_ARR = np.array(ERROR_CODES, dtype=object)

_UINT32_MAX = np.iinfo(np.uint32).max


def change_log_rows(incidents: Sequence[PlantedIncident]) -> list[dict]:
    """One row per incident that carries a `ChangeLogEntry`. The decoy carries none."""
    rows = []
    for incident in incidents:
        change = incident.change
        if change is None:
            continue
        rows.append(
            {
                "change_id": change.change_id,
                "changed_at": change.changed_at,
                "change_type": change.change_type,
                "component": change.component,
                "description": change.description,
                "dimension_key": change.dimension_key,
                "dimension_value": change.dimension_value,
            }
        )
    return rows


def generate(
    *,
    seed: int,
    window_start: datetime,
    days: int,
    sessions_per_day: int,
    titles: Sequence[Title],
    subscribers: Sequence[Subscriber],
    incidents: Sequence[PlantedIncident] = (),
    batch_size: int = 50_000,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield column-oriented batches of `playback_events` rows.

    Each batch is a dict keyed by `PLAYBACK_EVENTS_COLUMNS`, values are NumPy arrays of
    equal length, ready for a `clickhouse-connect` insert. Batches are flushed as soon
    as `batch_size` rows have accumulated, so memory stays bounded regardless of how
    many days/sessions are requested.
    """
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")
    if sessions_per_day < 0:
        raise ValueError(f"sessions_per_day must be >= 0, got {sessions_per_day}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if sessions_per_day > 0 and (not titles or not subscribers):
        raise ValueError("titles and subscribers must be non-empty when sessions_per_day > 0")

    rng = np.random.default_rng(seed)
    title_ids = np.array([t.title_id for t in titles], dtype=np.uint32)
    subscriber_ids = np.array([s.subscriber_id for s in subscribers], dtype=np.uint32)
    incidents = tuple(incidents)

    buffer: dict[str, list[np.ndarray]] = {c: [] for c in PLAYBACK_EVENTS_COLUMNS}
    buffered_rows = 0

    for i in range(days * BUCKETS_PER_DAY):
        bucket = window_start + timedelta(minutes=BUCKET_MINUTES * i)
        rows = _generate_bucket(rng, bucket, sessions_per_day, title_ids, subscriber_ids, incidents)
        if rows is None:
            continue
        for c in PLAYBACK_EVENTS_COLUMNS:
            buffer[c].append(rows[c])
        buffered_rows += len(rows["event_time"])

        while buffered_rows >= batch_size:
            merged = {c: np.concatenate(buffer[c]) for c in PLAYBACK_EVENTS_COLUMNS}
            yield {c: merged[c][:batch_size] for c in PLAYBACK_EVENTS_COLUMNS}
            buffer = {c: [merged[c][batch_size:]] for c in PLAYBACK_EVENTS_COLUMNS}
            buffered_rows = len(buffer["event_time"][0])

    if buffered_rows > 0:
        yield {c: np.concatenate(buffer[c]) for c in PLAYBACK_EVENTS_COLUMNS}


def _base_session_count(rng: np.random.Generator, bucket: datetime, sessions_per_day: int) -> int:
    if sessions_per_day <= 0:
        return 0
    expected = seasonality.expected_sessions(bucket, sessions_per_day) * seasonality.weekday_factor(
        bucket.weekday()
    )
    if expected <= 0:
        return 0
    return int(rng.poisson(expected))


def _generate_bucket(
    rng: np.random.Generator,
    bucket: datetime,
    sessions_per_day: int,
    title_ids: np.ndarray,
    subscriber_ids: np.ndarray,
    incidents: tuple[PlantedIncident, ...],
) -> dict[str, np.ndarray] | None:
    base_n = _base_session_count(rng, bucket, sessions_per_day)
    active = [inc for inc in incidents if inc.start <= bucket < inc.end]

    # Volume-boosting incidents (currently: the decoy) add extra sessions scoped to
    # their predicate on top of the baseline count, rather than distorting it.
    extra_slices: list[tuple[int, int]] = []  # (extra_n, forced_title_id)
    if len(title_ids) > 0:
        for inc in active:
            if inc.volume_multiplier != 1.0 and set(inc.predicate) == {"title_id"}:
                forced_title = int(inc.predicate["title_id"])
                extra = base_n / len(title_ids) * (inc.volume_multiplier - 1.0)
                extra_n = int(round(extra))
                if extra_n > 0:
                    extra_slices.append((extra_n, forced_title))

    total_n = base_n + sum(n for n, _ in extra_slices)
    if total_n <= 0:
        return None

    device_type = rng.choice(_DEVICE_TYPE_ARR, size=total_n, p=_DEVICE_WEIGHT_ARR)
    os_version = np.empty(total_n, dtype=object)
    app_version = np.empty(total_n, dtype=object)
    for device in topology.DEVICE_TYPES:
        mask = device_type == device
        count = int(mask.sum())
        if count:
            os_version[mask] = rng.choice(
                np.array(topology.os_versions_for(device), dtype=object), size=count
            )
            app_version[mask] = rng.choice(
                np.array(topology.app_versions_for(device), dtype=object), size=count
            )

    cdn = rng.choice(_CDN_ARR, size=total_n)
    pop = np.empty(total_n, dtype=object)
    for cdn_name in topology.CDNS:
        mask = cdn == cdn_name
        count = int(mask.sum())
        if count:
            pop[mask] = rng.choice(np.array(topology.pops_for(cdn_name), dtype=object), size=count)

    isp = rng.choice(_ISP_ARR, size=total_n)
    geo_idx = rng.integers(0, len(topology.GEO), size=total_n)
    country = _GEO_COUNTRY_ARR[geo_idx]
    region = _GEO_REGION_ARR[geo_idx]

    title_idx = rng.integers(0, len(title_ids), size=total_n)
    title_id = title_ids[title_idx].copy()
    offset = base_n
    for extra_n, forced_title in extra_slices:
        title_id[offset : offset + extra_n] = forced_title
        offset += extra_n

    subscriber_idx = rng.integers(0, len(subscriber_ids), size=total_n)
    subscriber_id = subscriber_ids[subscriber_idx]

    dim_arrays: dict[str, np.ndarray] = {
        "device_type": device_type,
        "os_version": os_version,
        "app_version": app_version,
        "cdn": cdn,
        "pop": pop,
        "isp": isp,
        "country": country,
        "region": region,
        "title_id": title_id.astype(str),
    }

    rebuffer_mult = np.ones(total_n)
    startup_mult = np.ones(total_n)
    bitrate_mult = np.ones(total_n)
    effect_targets = {"rebuffer": rebuffer_mult, "startup": startup_mult, "bitrate": bitrate_mult}

    for inc in active:
        if not inc.effects:
            continue
        mask = np.ones(total_n, dtype=bool)
        for key, value in inc.predicate.items():
            if key not in dim_arrays:
                raise ValueError(f"unknown predicate key {key!r}")
            mask &= dim_arrays[key] == value
        if not mask.any():
            continue
        affected = mask & (rng.random(total_n) < inc.affected_fraction)
        if not affected.any():
            continue
        for effect in inc.effects:
            target = effect_targets.get(effect.metric)
            if target is None:
                raise ValueError(f"unknown effect metric {effect.metric!r}")
            target[affected] *= effect.multiplier

    load = seasonality.load_factor(bucket)

    rebuffer_prob = min(max(seasonality.degrade(BASE_REBUFFER_PROB, load), 0.0), 1.0)
    rebuffer_mean = seasonality.degrade(BASE_REBUFFER_MEAN_MS, load) * rebuffer_mult
    stall = rng.random(total_n) < rebuffer_prob
    rebuffer_amount = np.zeros(total_n)
    if stall.any():
        rebuffer_amount[stall] = rng.exponential(rebuffer_mean[stall])
    rebuffer_amount = _to_uint32(rebuffer_amount)

    startup_mean = seasonality.degrade(BASE_STARTUP_MEAN_MS, load) * startup_mult
    startup_amount = _to_uint32(rng.exponential(startup_mean))

    bitrate_base = BASE_BITRATE_KBPS / seasonality.degrade(1.0, load)
    bitrate_noise = rng.normal(1.0, 0.05, size=total_n)
    bitrate_amount = _to_uint32(np.clip(bitrate_base * bitrate_mult * bitrate_noise, 100, None))

    error_flag = rng.random(total_n) < BASE_ERROR_PROB
    error_choice = rng.choice(_ERROR_CODE_ARR, size=total_n)

    bucket_np = np.datetime64(bucket.replace(tzinfo=None), "ms")
    start_offset = rng.integers(0, BUCKET_MS, size=total_n).astype("timedelta64[ms]")
    start_time = bucket_np + start_offset
    end_time = start_time + np.timedelta64(SESSION_DURATION_MS, "ms")

    session_id = _make_session_ids(rng, total_n)

    groups: list[dict[str, np.ndarray]] = []

    empty_ms = np.zeros(total_n, dtype=np.uint32)
    empty_error = np.full(total_n, "", dtype=object)

    groups.append(
        _row_group(
            event_time=start_time,
            session_id=session_id,
            subscriber_id=subscriber_id,
            title_id=title_id,
            device_type=device_type,
            os_version=os_version,
            app_version=app_version,
            cdn=cdn,
            pop=pop,
            isp=isp,
            country=country,
            region=region,
            event_type=np.full(total_n, "start", dtype=object),
            watched_ms=empty_ms,
            rebuffer_ms=empty_ms,
            startup_ms=startup_amount,
            bitrate_kbps=empty_ms,
            error_code=empty_error,
        )
    )

    hb_n = total_n * HEARTBEATS_PER_SESSION
    hb_offsets = np.tile(
        (np.arange(1, HEARTBEATS_PER_SESSION + 1) * HEARTBEAT_INTERVAL_MS).astype(
            "timedelta64[ms]"
        ),
        total_n,
    )
    groups.append(
        _row_group(
            event_time=np.repeat(start_time, HEARTBEATS_PER_SESSION) + hb_offsets,
            session_id=np.repeat(session_id, HEARTBEATS_PER_SESSION),
            subscriber_id=np.repeat(subscriber_id, HEARTBEATS_PER_SESSION),
            title_id=np.repeat(title_id, HEARTBEATS_PER_SESSION),
            device_type=np.repeat(device_type, HEARTBEATS_PER_SESSION),
            os_version=np.repeat(os_version, HEARTBEATS_PER_SESSION),
            app_version=np.repeat(app_version, HEARTBEATS_PER_SESSION),
            cdn=np.repeat(cdn, HEARTBEATS_PER_SESSION),
            pop=np.repeat(pop, HEARTBEATS_PER_SESSION),
            isp=np.repeat(isp, HEARTBEATS_PER_SESSION),
            country=np.repeat(country, HEARTBEATS_PER_SESSION),
            region=np.repeat(region, HEARTBEATS_PER_SESSION),
            event_type=np.full(hb_n, "heartbeat", dtype=object),
            watched_ms=np.full(hb_n, HEARTBEAT_INTERVAL_MS, dtype=np.uint32),
            rebuffer_ms=np.zeros(hb_n, dtype=np.uint32),
            startup_ms=np.zeros(hb_n, dtype=np.uint32),
            bitrate_kbps=np.repeat(bitrate_amount, HEARTBEATS_PER_SESSION),
            error_code=np.full(hb_n, "", dtype=object),
        )
    )

    groups.append(
        _row_group(
            event_time=end_time,
            session_id=session_id,
            subscriber_id=subscriber_id,
            title_id=title_id,
            device_type=device_type,
            os_version=os_version,
            app_version=app_version,
            cdn=cdn,
            pop=pop,
            isp=isp,
            country=country,
            region=region,
            event_type=np.full(total_n, "end", dtype=object),
            watched_ms=empty_ms,
            rebuffer_ms=empty_ms,
            startup_ms=empty_ms,
            bitrate_kbps=empty_ms,
            error_code=empty_error,
        )
    )

    if stall.any():
        n_stall = int(stall.sum())
        rebuffer_offset = rng.integers(0, SESSION_DURATION_MS, size=n_stall).astype(
            "timedelta64[ms]"
        )
        groups.append(
            _row_group(
                event_time=start_time[stall] + rebuffer_offset,
                session_id=session_id[stall],
                subscriber_id=subscriber_id[stall],
                title_id=title_id[stall],
                device_type=device_type[stall],
                os_version=os_version[stall],
                app_version=app_version[stall],
                cdn=cdn[stall],
                pop=pop[stall],
                isp=isp[stall],
                country=country[stall],
                region=region[stall],
                event_type=np.full(n_stall, "rebuffer", dtype=object),
                watched_ms=np.zeros(n_stall, dtype=np.uint32),
                rebuffer_ms=rebuffer_amount[stall],
                startup_ms=np.zeros(n_stall, dtype=np.uint32),
                bitrate_kbps=np.zeros(n_stall, dtype=np.uint32),
                error_code=np.full(n_stall, "", dtype=object),
            )
        )

    if error_flag.any():
        n_error = int(error_flag.sum())
        error_offset = rng.integers(0, SESSION_DURATION_MS, size=n_error).astype("timedelta64[ms]")
        groups.append(
            _row_group(
                event_time=start_time[error_flag] + error_offset,
                session_id=session_id[error_flag],
                subscriber_id=subscriber_id[error_flag],
                title_id=title_id[error_flag],
                device_type=device_type[error_flag],
                os_version=os_version[error_flag],
                app_version=app_version[error_flag],
                cdn=cdn[error_flag],
                pop=pop[error_flag],
                isp=isp[error_flag],
                country=country[error_flag],
                region=region[error_flag],
                event_type=np.full(n_error, "error", dtype=object),
                watched_ms=np.zeros(n_error, dtype=np.uint32),
                rebuffer_ms=np.zeros(n_error, dtype=np.uint32),
                startup_ms=np.zeros(n_error, dtype=np.uint32),
                bitrate_kbps=np.zeros(n_error, dtype=np.uint32),
                error_code=error_choice[error_flag],
            )
        )

    return {c: np.concatenate([g[c] for g in groups]) for c in PLAYBACK_EVENTS_COLUMNS}


def _row_group(**columns: np.ndarray) -> dict[str, np.ndarray]:
    return columns


def _to_uint32(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, _UINT32_MAX).astype(np.uint32)


def _make_session_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=object)
    raw = rng.bytes(n * 16)
    return np.array([uuid.UUID(bytes=raw[i * 16 : (i + 1) * 16]) for i in range(n)], dtype=object)
