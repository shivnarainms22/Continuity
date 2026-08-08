from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from continuity.data.catalog import (
    PLAN_ARPU,
    Subscriber,
    Title,
    generate_subscribers,
    generate_titles,
)
from continuity.data.topology import GEO

AS_OF = date(2026, 8, 8)


# --- generate_titles ---------------------------------------------------------


def test_generate_titles_returns_requested_count():
    rng = np.random.default_rng(1)
    titles = generate_titles(rng, 50, as_of=AS_OF)
    assert len(titles) == 50
    assert all(isinstance(t, Title) for t in titles)


def test_generate_titles_ids_are_unique():
    rng = np.random.default_rng(1)
    titles = generate_titles(rng, 200, as_of=AS_OF)
    assert len({t.title_id for t in titles}) == 200


def test_generate_titles_is_deterministic_for_same_seed():
    titles_a = generate_titles(np.random.default_rng(42), 100, as_of=AS_OF)
    titles_b = generate_titles(np.random.default_rng(42), 100, as_of=AS_OF)
    assert titles_a == titles_b


def test_generate_titles_different_seeds_differ():
    titles_a = generate_titles(np.random.default_rng(1), 100, as_of=AS_OF)
    titles_b = generate_titles(np.random.default_rng(2), 100, as_of=AS_OF)
    assert titles_a != titles_b


def test_generate_titles_some_are_premieres():
    rng = np.random.default_rng(7)
    titles = generate_titles(rng, 500, as_of=AS_OF)
    assert any(t.is_premiere for t in titles)
    assert any(not t.is_premiere for t in titles)


def test_generate_titles_release_date_never_in_future():
    reference = date(2026, 8, 8)
    rng = np.random.default_rng(7)
    titles = generate_titles(rng, 500, as_of=AS_OF)
    assert all(t.release_date <= reference for t in titles)


def test_generate_titles_zero_count_returns_empty_list():
    assert generate_titles(np.random.default_rng(1), 0, as_of=AS_OF) == []


def test_generate_titles_count_one_works():
    titles = generate_titles(np.random.default_rng(1), 1, as_of=AS_OF)
    assert len(titles) == 1


def test_generate_titles_negative_count_raises():
    with pytest.raises(ValueError):
        generate_titles(np.random.default_rng(1), -1, as_of=AS_OF)


# --- generate_subscribers ----------------------------------------------------


def test_generate_subscribers_returns_requested_count():
    rng = np.random.default_rng(1)
    subs = generate_subscribers(rng, 50, as_of=AS_OF)
    assert len(subs) == 50
    assert all(isinstance(s, Subscriber) for s in subs)


def test_generate_subscribers_ids_are_unique():
    rng = np.random.default_rng(1)
    subs = generate_subscribers(rng, 300, as_of=AS_OF)
    assert len({s.subscriber_id for s in subs}) == 300


def test_generate_subscribers_is_deterministic_for_same_seed():
    subs_a = generate_subscribers(np.random.default_rng(42), 200, as_of=AS_OF)
    subs_b = generate_subscribers(np.random.default_rng(42), 200, as_of=AS_OF)
    assert subs_a == subs_b


def test_generate_subscribers_different_seeds_differ():
    subs_a = generate_subscribers(np.random.default_rng(1), 200, as_of=AS_OF)
    subs_b = generate_subscribers(np.random.default_rng(2), 200, as_of=AS_OF)
    assert subs_a != subs_b


def test_generate_subscribers_arpu_matches_plan_exactly():
    rng = np.random.default_rng(3)
    subs = generate_subscribers(rng, 1000, as_of=AS_OF)
    for s in subs:
        assert s.monthly_arpu == PLAN_ARPU[s.plan]
        assert isinstance(s.monthly_arpu, Decimal)


def test_generate_subscribers_plan_mix_is_roughly_35_45_20():
    rng = np.random.default_rng(9)
    subs = generate_subscribers(rng, 20_000, as_of=AS_OF)
    counts = {"basic": 0, "standard": 0, "premium": 0}
    for s in subs:
        counts[s.plan] += 1
    n = len(subs)
    assert 0.30 <= counts["basic"] / n <= 0.40
    assert 0.40 <= counts["standard"] / n <= 0.50
    assert 0.15 <= counts["premium"] / n <= 0.25


def test_generate_subscribers_tenure_never_negative():
    rng = np.random.default_rng(5)
    subs = generate_subscribers(rng, 2000, as_of=AS_OF)
    assert all(s.tenure_days >= 0 for s in subs)


def test_generate_subscribers_tenure_matches_signup_date_arithmetic():
    rng = np.random.default_rng(5)
    subs = generate_subscribers(rng, 2000, as_of=AS_OF)
    for s in subs:
        assert (AS_OF - s.signup_date).days == s.tenure_days


def test_generate_subscribers_tenure_is_skewed_toward_new_subscribers():
    """The churn heuristic weights low tenure as high risk; a uniform distribution
    would make that heuristic meaningless, so the median must sit well below the
    midpoint of the observed range."""
    rng = np.random.default_rng(11)
    subs = generate_subscribers(rng, 20_000, as_of=AS_OF)
    tenures = sorted(s.tenure_days for s in subs)
    median = tenures[len(tenures) // 2]
    midpoint = (tenures[0] + tenures[-1]) / 2
    assert median < midpoint * 0.6


def test_generate_subscribers_geo_pairs_are_from_topology():
    rng = np.random.default_rng(13)
    subs = generate_subscribers(rng, 5000, as_of=AS_OF)
    geo_set = set(GEO)
    assert all((s.country, s.region) in geo_set for s in subs)


def test_generate_subscribers_zero_count_returns_empty_list():
    assert generate_subscribers(np.random.default_rng(1), 0, as_of=AS_OF) == []


def test_generate_subscribers_count_one_works():
    subs = generate_subscribers(np.random.default_rng(1), 1, as_of=AS_OF)
    assert len(subs) == 1


def test_generate_subscribers_negative_count_raises():
    with pytest.raises(ValueError):
        generate_subscribers(np.random.default_rng(1), -1, as_of=AS_OF)
