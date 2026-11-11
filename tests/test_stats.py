"""Pure statistics helpers against textbook values (no simulator)."""

import math

import pytest
from scipy import stats as sps

from flybrain.stats import compare, holm, summarize, verdict


def test_summary_mean_sample_sd_and_t_interval():
    s = summarize([1, 2, 3, 4, 5])
    assert s.n == 5 and s.mean == 3 and s.sd == pytest.approx(math.sqrt(2.5))
    half = 2.7764451051977934 * math.sqrt(2.5) / math.sqrt(5)  # t(0.975, 4) * se
    assert (s.ci_low, s.ci_high) == (pytest.approx(3 - half), pytest.approx(3 + half))


def test_sample_sd_uses_n_minus_one():
    assert summarize([2, 4, 4, 4, 5, 5, 7, 9]).sd == pytest.approx(math.sqrt(32 / 7))


def test_one_trial_has_no_spread_and_no_interval():
    s = summarize([7.0])
    assert s.n == 1 and s.mean == 7.0 and s.sd is None and s.ci_low is None and s.ci_high is None


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        summarize([])


def test_welch_matches_known_values():
    c = compare([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])  # second minus first
    assert c.delta == 3 and c.t == pytest.approx(1.8973665961010275) and c.df == pytest.approx(5.882352941176471)
    assert c.p == pytest.approx(0.10753119493062728)
    assert c.cohens_d == pytest.approx(3 / 2.5)  # pooled SD = sqrt((2.5 + 10) / 2)
    half = sps.t.ppf(0.975, c.df) * math.sqrt(0.5 + 2.0)
    assert (c.delta_ci_low, c.delta_ci_high) == (pytest.approx(3 - half), pytest.approx(3 + half))


def test_welch_is_antisymmetric_in_sign_only():
    up, down = compare([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), compare([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])
    assert down.delta == -up.delta and down.p == pytest.approx(up.p) and down.cohens_d == pytest.approx(-up.cohens_d)


def test_identical_trials_are_not_different():
    c = compare([5, 5, 5], [5, 5, 5])
    assert c.delta == 0 and c.p == 1.0 and c.cohens_d == 0


def test_no_spread_but_a_different_mean_is_flagged_not_nan():
    c = compare([60, 60, 60], [0, 0, 0])
    assert c.delta == -60 and c.p == 0.0 and c.cohens_d is None  # d is undefined without spread


def test_too_few_trials_give_no_test():
    c = compare([60.0], [10.0, 12.0])
    assert c.delta == pytest.approx(-49.0) and c.p is None and c.delta_ci_low is None and c.cohens_d is None


def test_holm_adjustment_is_monotone_and_keeps_order():
    assert holm([0.01, 0.04, 0.03]) == [pytest.approx(0.03), pytest.approx(0.06), pytest.approx(0.06)]
    assert holm([0.5]) == [0.5]
    assert holm([0.4, 0.9]) == [pytest.approx(0.8), 0.9] and holm([]) == []
    assert holm([0.3, 0.3, 0.3]) == [pytest.approx(0.9)] * 3  # capped at 1 and never below a smaller p
    assert max(holm([0.9, 0.8, 0.7])) <= 1.0


def test_holm_skips_missing_p_values():
    assert holm([0.01, None, 0.04]) == [pytest.approx(0.02), None, pytest.approx(0.04)]


def test_verdict_is_honest_about_noise_and_the_model():
    sig = verdict(0.003, 5)
    assert "significant at p<0.05" in sig and "5 trials" in sig and "model" in sig.lower()
    assert "not distinguishable from noise with 5 trials" in verdict(0.4, 5)
    assert "at least 2 trials" in verdict(None, 1)


def test_verdict_names_the_multiple_comparison_correction():
    text = verdict(0.01, 3, tests=8)
    assert "8 comparisons" in text and "Holm" in text
