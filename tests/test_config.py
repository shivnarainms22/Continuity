import pytest

from continuity.config import ClickHouseConfig

_REQUIRED = ("CLICKHOUSE_HOST", "CLICKHOUSE_PASSWORD")
_OPTIONAL = ("CLICKHOUSE_PORT", "CLICKHOUSE_USER", "CLICKHOUSE_DATABASE", "CLICKHOUSE_SECURE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Tests must not inherit the developer's real .env."""
    for name in (*_REQUIRED, *_OPTIONAL):
        monkeypatch.delenv(name, raising=False)


def test_from_env_reads_all_fields(monkeypatch):
    for key, value in {
        "CLICKHOUSE_HOST": "ch.example.com",
        "CLICKHOUSE_PORT": "8443",
        "CLICKHOUSE_USER": "reader",
        "CLICKHOUSE_PASSWORD": "pw",
        "CLICKHOUSE_DATABASE": "continuity",
        "CLICKHOUSE_SECURE": "true",
    }.items():
        monkeypatch.setenv(key, value)

    cfg = ClickHouseConfig.from_env()

    assert cfg.host == "ch.example.com"
    assert cfg.port == 8443
    assert cfg.user == "reader"
    assert cfg.password == "pw"
    assert cfg.database == "continuity"
    assert cfg.secure is True


def test_defaults_apply_when_optional_vars_absent(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "localhost")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "pw")

    cfg = ClickHouseConfig.from_env()

    assert cfg.port == 8123
    assert cfg.user == "default"
    assert cfg.database == "continuity"
    assert cfg.secure is False


def test_missing_host_raises_with_actionable_message(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "pw")
    with pytest.raises(ValueError, match="CLICKHOUSE_HOST"):
        ClickHouseConfig.from_env()


def test_missing_password_raises_with_actionable_message(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "localhost")
    with pytest.raises(ValueError, match="CLICKHOUSE_PASSWORD"):
        ClickHouseConfig.from_env()


def test_blank_value_is_treated_as_missing(monkeypatch):
    """An exported-but-empty var is a configuration mistake, not a valid value."""
    monkeypatch.setenv("CLICKHOUSE_HOST", "   ")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "pw")
    with pytest.raises(ValueError, match="CLICKHOUSE_HOST"):
        ClickHouseConfig.from_env()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("TRUE", True), ("1", True),
        ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False),
        ("no", False), ("off", False),
    ],
)
def test_secure_parses_common_boolean_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "p")
    monkeypatch.setenv("CLICKHOUSE_SECURE", raw)

    assert ClickHouseConfig.from_env().secure is expected


def test_unrecognised_boolean_raises_rather_than_defaulting(monkeypatch):
    """Silently reading 'flase' as False would send credentials over plaintext."""
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "p")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "flase")

    with pytest.raises(ValueError, match="CLICKHOUSE_SECURE"):
        ClickHouseConfig.from_env()


@pytest.mark.parametrize("raw", ["notanumber", "80.5", "", "  "])
def test_non_integer_port_raises(monkeypatch, raw):
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "p")
    monkeypatch.setenv("CLICKHOUSE_PORT", raw)

    with pytest.raises(ValueError, match="CLICKHOUSE_PORT"):
        ClickHouseConfig.from_env()


@pytest.mark.parametrize("port", ["0", "65536", "-1", "99999"])
def test_out_of_range_port_raises(monkeypatch, port):
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "p")
    monkeypatch.setenv("CLICKHOUSE_PORT", port)

    with pytest.raises(ValueError, match="CLICKHOUSE_PORT"):
        ClickHouseConfig.from_env()


def test_repr_does_not_leak_password(monkeypatch):
    """This object is logged during agent runs and appears in demo screenshots."""
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "hunter2")

    rendered = repr(ClickHouseConfig.from_env())

    assert "hunter2" not in rendered
    assert "***" in rendered


def test_str_does_not_leak_password(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "hunter2")

    assert "hunter2" not in str(ClickHouseConfig.from_env())
