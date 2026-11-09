from evals.leaderboard.stats import (
    IMPROVED,
    NO_CLEAR_CHANGE,
    OVERALL,
    POSSIBLE_DRIFT,
    REGRESSED,
    compare,
    drift_check,
    label_for,
    summarize,
)

CATEGORIES = {f"t{i}": "long" for i in range(60)}


def test_labels():
    assert label_for(0.01, 0.2) == IMPROVED
    assert label_for(-0.2, -0.01) == REGRESSED
    assert label_for(-0.1, 0.1) == NO_CLEAR_CHANGE


def test_large_consistent_gain_is_improved_and_reversal_is_regressed():
    low = {t: 0.2 + (i % 3) * 0.1 for i, t in enumerate(CATEGORIES)}
    high = {t: v + 0.4 + (i % 2) * 0.1 for i, (t, v) in enumerate(low.items())}
    up = compare(high, low, CATEGORIES, iterations=2000)["long"]
    down = compare(low, high, CATEGORIES, iterations=2000)["long"]
    assert up["label"] == IMPROVED
    assert down["label"] == REGRESSED
    assert up["low"] < up["difference"]


def test_identical_runs_show_no_clear_change():
    values = {t: float(i % 2) for i, t in enumerate(CATEGORIES)}
    assert compare(values, dict(values), CATEGORIES, iterations=500)["long"]["label"] == NO_CLEAR_CHANGE


def test_noise_is_not_labeled_as_a_change():
    new = {t: float(i % 2) for i, t in enumerate(CATEGORIES)}
    old = {t: float((i + 1) % 2) for i, t in enumerate(CATEGORIES)}
    assert compare(new, old, CATEGORIES, iterations=2000)["long"]["label"] == NO_CLEAR_CHANGE


def test_correction_widens_the_interval():
    new = {t: float(i % 2) for i, t in enumerate(CATEGORIES)}
    old = {t: 0.5 for t in CATEGORIES}
    one = compare(new, old, CATEGORIES, 1, iterations=4000)["long"]
    many = compare(new, old, CATEGORIES, 8, iterations=4000)["long"]
    assert many["corrected_high"] - many["corrected_low"] > one["high"] - one["low"]


def test_overall_weights_categories_equally():
    categories = {"a1": "long", "a2": "long", "a3": "long", "b1": "grounding"}
    means = {t: 0.0 for t in categories}
    means["b1"] = 1.0
    result = compare(means, {t: 0.0 for t in categories}, categories, iterations=50)
    assert result[OVERALL]["difference"] == 0.5
    assert result["long"]["tasks"] == 3
    assert result[OVERALL]["tasks"] == 4


def test_single_category_has_no_overall():
    assert OVERALL not in compare({"t0": 1.0}, {"t0": 0.0}, {"t0": "long"}, iterations=10)


def test_only_shared_and_known_tasks_are_compared():
    result = compare({"a": 1.0, "b": 1.0, "x": 1.0}, {"a": 0.0, "c": 0.0, "x": 0.0}, {"a": "long", "b": "long"}, iterations=10)
    assert result["long"]["tasks"] == 1


def test_no_shared_tasks_gives_empty_result():
    assert compare({"a": 1.0}, {"b": 0.0}, {"a": "long", "b": "long"}) == {}


def test_summarize_reports_score_and_interval():
    means = {t: float(i % 2) for i, t in enumerate(CATEGORIES)}
    result = summarize(means, CATEGORIES, iterations=500)["long"]
    assert result["score"] == 0.5
    assert result["low"] < 0.5 < result["high"]
    assert result["tasks"] == 60


def test_drift_check_finds_no_drift_between_identical_runs():
    run = {"sonnet": {t: float(i % 2) for i, t in enumerate(CATEGORIES)}}
    drift = drift_check(run, run, iterations=200)
    assert drift["sonnet"]["difference"] == 0
    assert drift["sonnet"]["label"] == NO_CLEAR_CHANGE


def test_drift_check_flags_a_large_shift():
    before = {"sonnet": {t: 1.0 for t in CATEGORIES}}
    after = {"sonnet": {t: 0.0 for t in CATEGORIES}}
    assert drift_check(before, after, iterations=500)["sonnet"]["label"] == POSSIBLE_DRIFT


def test_overall_is_a_reserved_category_name():
    import pytest

    categories = {"a": OVERALL, "b": "long"}
    with pytest.raises(ValueError, match="reserved"):
        summarize({"a": 0.0, "b": 1.0}, categories)
    with pytest.raises(ValueError, match="reserved"):
        compare({"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, categories)
