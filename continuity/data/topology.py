"""The dimension universe. This IS the drill-down hierarchy.

Pure data, no I/O, no imports from anywhere else in the project. The subtle logic here
must stay unit-testable without infrastructure.
"""

from __future__ import annotations

# Coarse to fine. The Scope stage in sub-project 2 walks this order.
DIMENSION_HIERARCHY: tuple[str, ...] = (
    "cdn",
    "pop",
    "isp",
    "country",
    "region",
    "device_type",
    "os_version",
    "app_version",
)

CDNS: tuple[str, ...] = ("cdn_meridian", "cdn_northwind", "cdn_solstice")

POPS_BY_CDN: dict[str, tuple[str, ...]] = {
    "cdn_meridian": ("mer-iad-1", "mer-ord-1", "mer-dfw-1", "mer-lax-1", "mer-lhr-1"),
    "cdn_northwind": ("nw-atl-2", "nw-sea-1", "nw-jfk-3", "nw-fra-1"),
    "cdn_solstice": ("sol-den-1", "sol-mia-1", "sol-sjc-2", "sol-yyz-1"),
}

ISPS: tuple[str, ...] = (
    "comcast",
    "charter",
    "att",
    "verizon",
    "cox",
    "tmobile",
    "bt",
    "deutsche_telekom",
)

GEO: tuple[tuple[str, str], ...] = (
    ("US", "us_northeast"),
    ("US", "us_southeast"),
    ("US", "us_midwest"),
    ("US", "us_west"),
    ("CA", "ca_east"),
    ("GB", "gb_south"),
    ("DE", "de_west"),
)

DEVICE_TYPES: tuple[str, ...] = (
    "roku",
    "firetv",
    "samsung_tv",
    "lg_tv",
    "ios",
    "android",
    "web",
)

_OS_VERSIONS: dict[str, tuple[str, ...]] = {
    "roku": ("roku_os_13.0", "roku_os_14.0", "roku_os_14.1"),
    "firetv": ("fireos_7", "fireos_8"),
    "samsung_tv": ("tizen_6.5", "tizen_7.0"),
    "lg_tv": ("webos_23", "webos_24"),
    "ios": ("ios_17.5", "ios_18.2"),
    "android": ("android_14", "android_15"),
    "web": ("chrome_129", "safari_18", "firefox_131"),
}

# Load-bearing: the 8.2.0 line ships on roku, firetv, ios and android, so a fault scoped
# to "roku AND 8.2.0" is not separable by splitting on either dimension alone -- the
# hierarchical drill-down has to do real work to find it. Roku additionally carries a
# legacy 8.0.9 build, so its version set is not identical to any other platform's.
_APP_VERSIONS: dict[str, tuple[str, ...]] = {
    "roku": ("8.0.9", "8.1.4", "8.2.0"),
    "firetv": ("8.1.4", "8.2.0"),
    "samsung_tv": ("8.1.2", "8.1.4"),
    "lg_tv": ("8.1.2", "8.1.4"),
    "ios": ("8.1.4", "8.2.0"),
    "android": ("8.1.4", "8.2.0"),
    "web": ("web_2026.7", "web_2026.8"),
}

# Device population weights -- TV platforms dominate watch time in real streaming data.
DEVICE_WEIGHTS: dict[str, float] = {
    "roku": 0.24,
    "firetv": 0.18,
    "samsung_tv": 0.15,
    "lg_tv": 0.09,
    "ios": 0.13,
    "android": 0.13,
    "web": 0.08,
}


def pops_for(cdn: str) -> tuple[str, ...]:
    try:
        return POPS_BY_CDN[cdn]
    except KeyError:
        raise ValueError(f"Unknown CDN {cdn!r}. Known: {sorted(POPS_BY_CDN)}") from None


def os_versions_for(device_type: str) -> tuple[str, ...]:
    try:
        return _OS_VERSIONS[device_type]
    except KeyError:
        raise ValueError(f"Unknown device {device_type!r}. Known: {sorted(_OS_VERSIONS)}") from None


def app_versions_for(device_type: str) -> tuple[str, ...]:
    try:
        return _APP_VERSIONS[device_type]
    except KeyError:
        raise ValueError(
            f"Unknown device {device_type!r}. Known: {sorted(_APP_VERSIONS)}"
        ) from None
