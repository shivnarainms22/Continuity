import pytest

from continuity.data.topology import (
    CDNS,
    DEVICE_TYPES,
    DEVICE_WEIGHTS,
    DIMENSION_HIERARCHY,
    GEO,
    ISPS,
    app_versions_for,
    os_versions_for,
    pops_for,
)


def test_hierarchy_is_ordered_coarse_to_fine():
    assert DIMENSION_HIERARCHY[0] == "cdn"
    assert DIMENSION_HIERARCHY.index("cdn") < DIMENSION_HIERARCHY.index("pop")
    assert DIMENSION_HIERARCHY.index("device_type") < DIMENSION_HIERARCHY.index("app_version")


def test_every_cdn_has_pops():
    assert all(pops_for(c) for c in CDNS)


def test_pops_are_globally_unique_across_cdns():
    """A PoP name must identify its CDN unambiguously or drill-down attributes wrongly."""
    seen = [p for c in CDNS for p in pops_for(c)]
    assert len(seen) == len(set(seen))


def test_app_versions_are_device_appropriate():
    assert app_versions_for("roku") != app_versions_for("ios")
    assert all(v for v in app_versions_for("roku"))


def test_pops_for_unknown_cdn_raises_naming_value_and_valid_options():
    with pytest.raises(ValueError, match="bogus_cdn") as exc:
        pops_for("bogus_cdn")
    for cdn in CDNS:
        assert cdn in str(exc.value)


def test_os_versions_for_unknown_device_raises_naming_value_and_valid_options():
    with pytest.raises(ValueError, match="bogus_device") as exc:
        os_versions_for("bogus_device")
    for device in DEVICE_TYPES:
        assert device in str(exc.value)


def test_app_versions_for_unknown_device_raises_naming_value_and_valid_options():
    with pytest.raises(ValueError, match="bogus_device") as exc:
        app_versions_for("bogus_device")
    for device in DEVICE_TYPES:
        assert device in str(exc.value)


def test_device_weights_cover_exactly_the_device_types():
    assert set(DEVICE_WEIGHTS) == set(DEVICE_TYPES)


def test_device_weights_sum_to_one():
    assert sum(DEVICE_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_device_type_has_os_versions():
    assert all(os_versions_for(d) for d in DEVICE_TYPES)


def test_every_device_type_has_app_versions():
    assert all(app_versions_for(d) for d in DEVICE_TYPES)


def test_no_dimension_value_contains_sql_quote_or_backslash():
    """Values are interpolated into SQL literals downstream."""
    everything = [
        *CDNS,
        *DEVICE_TYPES,
        *ISPS,
        *(p for c in CDNS for p in pops_for(c)),
        *(country for country, region in GEO),
        *(region for country, region in GEO),
        *(v for d in DEVICE_TYPES for v in os_versions_for(d)),
        *(v for d in DEVICE_TYPES for v in app_versions_for(d)),
    ]
    assert not any("'" in v or "\\" in v for v in everything)


def test_an_app_version_is_shared_across_multiple_device_types():
    """A planted incident scoped to 'roku AND app 8.2.0' must not be separable by
    splitting on device_type alone -- at least one app_version string has to appear
    under 2+ device types, forcing the drill-down to actually work."""
    counts: dict[str, int] = {}
    for device in DEVICE_TYPES:
        for version in app_versions_for(device):
            counts[version] = counts.get(version, 0) + 1
    assert any(count >= 2 for count in counts.values())
