import pytest

from continuity.analysis.slices import (
    RAW_EVENTS_TABLE,
    ROLLUP_TABLE,
    InvalidSliceError,
    Slice,
)


def test_empty_slice_renders_valid_where_clause_matching_whole_population():
    assert Slice().where_sql() == "1"


def test_empty_slice_has_no_dimensions():
    assert Slice().dimensions == ()


def test_empty_slice_str_is_readable():
    assert str(Slice()) == "(all)"


def test_single_predicate_renders_equality_clause():
    s = Slice().refine("device_type", "roku")
    assert s.where_sql() == "device_type = 'roku'"


def test_multi_predicate_renders_all_clauses_joined_with_and():
    s = Slice().refine("device_type", "roku").refine("app_version", "8.2.0")
    where = s.where_sql()
    assert "device_type = 'roku'" in where
    assert "app_version = '8.2.0'" in where
    assert " AND " in where


def test_multi_predicate_orders_clauses_by_hierarchy_not_alphabetically():
    # device_type precedes app_version in DIMENSION_HIERARCHY
    s = Slice().refine("app_version", "8.2.0").refine("device_type", "roku")
    where = s.where_sql()
    assert where.index("device_type") < where.index("app_version")


def test_refine_returns_new_slice_and_leaves_original_unchanged():
    base = Slice().refine("device_type", "roku")
    refined = base.refine("app_version", "8.2.0")
    assert base.dimensions == ("device_type",)
    assert refined.dimensions == ("device_type", "app_version")
    assert base is not refined


def test_refine_overrides_existing_dimension_value():
    s = Slice().refine("device_type", "roku").refine("device_type", "firetv")
    assert s.where_sql() == "device_type = 'firetv'"


def test_dimensions_property_lists_hierarchy_order():
    s = Slice().refine("app_version", "8.2.0").refine("device_type", "roku")
    assert s.dimensions == ("device_type", "app_version")


def test_str_renders_readable_value_sequence_in_hierarchy_order():
    s = Slice().refine("app_version", "8.2.0").refine("device_type", "roku")
    assert str(s) == "roku / 8.2.0"


def test_unknown_dimension_is_rejected():
    with pytest.raises(InvalidSliceError, match="bogus_dim"):
        Slice().refine("bogus_dim", "x")


def test_unknown_dimension_error_lists_allowed_dimensions():
    with pytest.raises(InvalidSliceError) as exc:
        Slice().refine("bogus_dim", "x")
    assert "device_type" in str(exc.value)
    assert "title_id" in str(exc.value)


@pytest.mark.parametrize(
    "payload",
    [
        "x' OR 1=1 --",
        "x'; DROP TABLE playback_events; --",
        "x' UNION SELECT password FROM subscribers --",
        "back\\slash",
        "x' OR '1'='1",
    ],
)
def test_injection_payloads_are_safely_escaped_not_executable(payload):
    """An injection payload must render as an inert string literal: it must stay
    inside a single quoted literal and must not terminate that literal early."""
    s = Slice().refine("device_type", payload)
    where = s.where_sql()
    # The clause must be exactly one equality against one quoted literal for
    # device_type -- no unescaped quote may close the literal early.
    assert where.startswith("device_type = '")
    assert where.endswith("'")
    inner = where[len("device_type = '") : -1]
    # every literal single quote inside must be escaped (preceded by a backslash)
    import re

    unescaped_quotes = re.findall(r"(?<!\\)'", inner)
    assert unescaped_quotes == []


def test_injection_payload_cannot_break_out_to_add_a_second_clause():
    s = Slice().refine("device_type", "roku'; DROP TABLE playback_events; --")
    where = s.where_sql()
    assert where.count(" AND ") == 0
    assert "DROP TABLE" in where  # present, but inert -- inside the escaped literal


def test_title_id_predicate_renders_as_unquoted_integer():
    s = Slice().refine("title_id", "5821")
    assert s.where_sql() == "title_id = 5821"


def test_title_id_predicate_accepts_int_value():
    s = Slice().refine("title_id", 5821)
    assert s.where_sql() == "title_id = 5821"


def test_title_id_predicate_rejects_non_numeric_value():
    with pytest.raises(InvalidSliceError):
        Slice().refine("title_id", "5821' OR 1=1 --")


def test_empty_value_is_rejected():
    with pytest.raises(InvalidSliceError):
        Slice().refine("device_type", "")


def test_slice_is_hashable_and_usable_as_dict_key():
    a = Slice().refine("device_type", "roku")
    b = Slice().refine("device_type", "roku")
    cache = {a: "cached"}
    assert cache[b] == "cached"


def test_slices_with_same_predicates_are_equal():
    a = Slice().refine("device_type", "roku").refine("app_version", "8.2.0")
    b = Slice().refine("app_version", "8.2.0").refine("device_type", "roku")
    assert a == b
    assert hash(a) == hash(b)


def test_slice_without_title_id_requires_rollup_table():
    s = Slice().refine("device_type", "roku").refine("app_version", "8.2.0")
    assert s.requires_raw_events is False
    assert s.required_table == ROLLUP_TABLE


def test_slice_with_title_id_requires_raw_events_table():
    """qoe_rollup_5m deliberately excludes title_id -- any slice naming it cannot be
    served by the rollup and must fall back to playback_events."""
    s = Slice().refine("title_id", "5821")
    assert s.requires_raw_events is True
    assert s.required_table == RAW_EVENTS_TABLE


def test_slice_with_title_id_and_other_dimensions_still_requires_raw_events():
    s = Slice().refine("device_type", "roku").refine("title_id", "5821")
    assert s.requires_raw_events is True
    assert s.required_table == RAW_EVENTS_TABLE
