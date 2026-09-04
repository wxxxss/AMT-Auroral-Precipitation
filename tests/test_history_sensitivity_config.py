import pytest

from data.dataset_v4 import lag_minutes_for_history, sw_dim_for_history


def test_history_dimensions_match_manuscript_runs():
    expected = {60: 68, 90: 92, 120: 116, 180: 164, 240: 212}
    assert {minutes: sw_dim_for_history(minutes) for minutes in expected} == expected


def test_history_lags_are_five_minute_steps():
    assert lag_minutes_for_history(60) == list(range(5, 61, 5))
    assert lag_minutes_for_history(240)[-1] == 240


def test_unsupported_history_rejected():
    with pytest.raises(ValueError):
        sw_dim_for_history(150)
