"""Slice: an immutable dimension predicate that renders itself to a SQL WHERE clause.

Pure logic, no I/O, no SQL execution -- this module only builds strings. It is the
single place that decides two things every downstream query depends on:

1. Whether a value is safe to interpolate into SQL (dimension names are checked
   against an allowlist; string values are escaped so they cannot break out of their
   quoted literal).
2. Whether a predicate can be answered by the ``qoe_rollup_5m`` rollup or must fall
   back to ``playback_events`` -- the rollup deliberately excludes ``title_id`` (see
   continuity/data/schema.py), so any slice naming it requires raw events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from continuity.data.topology import DIMENSION_HIERARCHY

TITLE_ID_DIMENSION = "title_id"

# The dimension universe a Slice may ever predicate on. title_id is not part of the
# drill-down hierarchy (it is excluded from the rollup for cardinality reasons) but is
# still a valid, allowlisted predicate that forces the raw-events table.
ALLOWED_DIMENSIONS: frozenset[str] = frozenset({*DIMENSION_HIERARCHY, TITLE_ID_DIMENSION})

ROLLUP_TABLE = "qoe_rollup_5m"
RAW_EVENTS_TABLE = "playback_events"

# Dimensions that must be read from RAW_EVENTS_TABLE regardless of which table the
# enclosing Slice's own predicates would otherwise allow -- title_id is excluded from
# the rollup for cardinality reasons (see continuity/data/schema.py). A future
# raw-events-only dimension is added here, not by special-casing its name at each
# call site that groups dimensions by table (see `dimension_required_table`).
RAW_ONLY_DIMENSIONS: frozenset[str] = frozenset({TITLE_ID_DIMENSION})


class InvalidSliceError(ValueError):
    """Raised when a Slice predicate names an unknown dimension or an unsafe value."""


def _sort_key(dimension: str) -> int:
    """Coarse-to-fine hierarchy order, with title_id (outside the hierarchy) last."""
    try:
        return DIMENSION_HIERARCHY.index(dimension)
    except ValueError:
        return len(DIMENSION_HIERARCHY)


def _escape_string_literal(value: str) -> str:
    """Escape a value for a single-quoted ClickHouse string literal.

    Backslash must be escaped before the single quote -- reversing the order would
    double-escape the backslashes the quote-escaping step introduces. Once escaped,
    the value cannot contain an unescaped ``'`` and therefore cannot terminate the
    literal early, no matter what characters (``;``, ``--``, ``/*``) it contains.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _render_predicate(dimension: str, value: str) -> str:
    if dimension == TITLE_ID_DIMENSION:
        if not value.isdigit():
            raise InvalidSliceError(
                f"title_id must be a non-negative integer, got {value!r}."
            )
        return f"{dimension} = {value}"
    return f"{dimension} = '{_escape_string_literal(value)}'"


def _validate(dimension: str, value: str) -> None:
    if dimension not in ALLOWED_DIMENSIONS:
        raise InvalidSliceError(
            f"Unknown dimension {dimension!r}. Known: {sorted(ALLOWED_DIMENSIONS)}"
        )
    if value == "":
        raise InvalidSliceError(f"Empty value for dimension {dimension!r}.")


@dataclass(frozen=True)
class Slice:
    """An immutable dimension predicate, e.g. {device_type: roku, app_version: 8.2.0}.

    Hashable and equality-comparable by predicate content so it can be used as a dict
    key (baselines, caches). Construct with ``Slice()`` for the whole population, then
    ``.refine(dimension, value)`` to narrow -- each call returns a new Slice, the
    original is untouched.
    """

    predicates: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for dimension, value in self.predicates:
            _validate(dimension, value)
            _render_predicate(dimension, value)  # raises on an unsafe value eagerly

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Dimension names present in this slice, coarse-to-fine hierarchy order."""
        return tuple(sorted((d for d, _ in self.predicates), key=_sort_key))

    @property
    def requires_raw_events(self) -> bool:
        """True iff this slice cannot be answered by qoe_rollup_5m (names title_id)."""
        return any(d == TITLE_ID_DIMENSION for d, _ in self.predicates)

    @property
    def required_table(self) -> str:
        return RAW_EVENTS_TABLE if self.requires_raw_events else ROLLUP_TABLE

    def refine(self, dimension: str, value: str | int) -> Slice:
        """Return a NEW slice with `dimension` set to `value`, overriding any prior
        value for that dimension. The original slice is unchanged."""
        _validate(dimension, str(value))
        remaining = {d: v for d, v in self.predicates if d != dimension}
        remaining[dimension] = str(value)
        return Slice(frozenset(remaining.items()))

    def where_sql(self) -> str:
        """A valid SQL WHERE-clause body. The empty slice (whole population) renders
        the tautology "1" rather than an empty string, so callers can always do
        `WHERE {slice.where_sql()}` regardless of predicate count."""
        if not self.predicates:
            return "1"
        ordered = sorted(self.predicates, key=lambda kv: _sort_key(kv[0]))
        return " AND ".join(_render_predicate(d, v) for d, v in ordered)

    def __str__(self) -> str:
        if not self.predicates:
            return "(all)"
        ordered = sorted(self.predicates, key=lambda kv: _sort_key(kv[0]))
        return " / ".join(v for _, v in ordered)


def dimension_required_table(slice_: Slice, dimension: str) -> str:
    """Which table answers a GROUP BY on `dimension` within `slice_`'s predicate.

    RAW_EVENTS_TABLE when either the slice's own predicates force it (`slice_` already
    pins a raw-only dimension, e.g. title_id) or `dimension` itself does (it is in
    `RAW_ONLY_DIMENSIONS`) -- ROLLUP_TABLE otherwise. Callers that batch several
    dimensions into one query (see continuity/analysis/split.py) must group by this
    value first and never UNION ALL a rollup-backed arm with a raw-events-backed one:
    mixing them forces the cheap rollup arms to share the expensive raw-events arm's
    memory budget for the query's whole lifetime.
    """
    if slice_.requires_raw_events or dimension in RAW_ONLY_DIMENSIONS:
        return RAW_EVENTS_TABLE
    return ROLLUP_TABLE
