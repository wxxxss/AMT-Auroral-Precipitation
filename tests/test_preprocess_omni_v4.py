import numpy as np
import pandas as pd

from data.preprocess_omni_v4 import (
    add_history_lags,
    backward_match_ssj,
    regularize_omni_5min,
    stratified_chronological_split,
)


def _omni_frame(times):
    n = len(times)
    return pd.DataFrame(
        {
            "utc": pd.to_datetime(times),
            "Bx": np.arange(n, dtype=float),
            "By": np.arange(n, dtype=float) + 1.0,
            "Bz": np.arange(n, dtype=float) - 2.0,
            "Vx": np.full(n, -400.0),
            "Vy": np.full(n, 10.0),
            "Vz": np.full(n, -5.0),
            "P_dyn": np.full(n, 2.0),
        }
    )


def test_regularize_omni_applies_manuscript_qc_and_short_gap_interpolation():
    df = _omni_frame(
        ["2014-01-01 00:00", "2014-01-01 00:05", "2014-01-01 00:10"]
    )
    df.loc[1, "Bx"] = 600.0

    out = regularize_omni_5min(df, interpolation_limit_steps=6)

    assert out["utc"].tolist() == list(
        pd.to_datetime(["2014-01-01 00:00", "2014-01-01 00:05", "2014-01-01 00:10"])
    )
    assert np.isclose(out.loc[1, "Bx"], 1.0)
    assert np.isclose(out.loc[1, "By"], 2.0)


def test_regularize_omni_does_not_partially_fill_gaps_longer_than_30_minutes():
    # Seven missing 5-min records separate these two observations (35 min of
    # missing data). The manuscript permits interpolation only for complete
    # gaps of up to 30 min, so none of the seven missing Bx values may be filled.
    df = _omni_frame(["2014-01-01 00:00", "2014-01-01 00:40"])

    out = regularize_omni_5min(df, interpolation_limit_steps=6)

    interior = out.loc[
        (out["utc"] > pd.Timestamp("2014-01-01 00:00"))
        & (out["utc"] < pd.Timestamp("2014-01-01 00:40")),
        "Bx",
    ]
    assert len(interior) == 7
    assert interior.isna().all()


def test_add_history_lags_uses_5_minute_steps_and_drops_incomplete_history():
    df = _omni_frame(pd.date_range("2014-01-01", periods=4, freq="5min"))

    out = add_history_lags(df, history_minutes=10)

    assert len(out) == 2
    assert {"Bx_lag_5", "Bx_lag_10", "P_dyn_lag_5", "P_dyn_lag_10"}.issubset(out.columns)
    assert out.iloc[0]["Bx"] == 2.0
    assert out.iloc[0]["Bx_lag_5"] == 1.0
    assert out.iloc[0]["Bx_lag_10"] == 0.0


def test_backward_match_never_uses_future_omni_and_respects_tolerance():
    omni = _omni_frame(["2014-01-01 00:00", "2014-01-01 00:05"])
    ssj = pd.DataFrame(
        {
            "utc": pd.to_datetime(["2014-01-01 00:04", "2014-01-01 00:16"]),
            "mlat": [70.0, 70.0],
            "mlt": [12.0, 12.0],
        }
    )

    out = backward_match_ssj(ssj, omni, tolerance="10min", drop_unmatched=False)

    assert out.loc[0, "Bx"] == 0.0
    assert np.isnan(out.loc[1, "Bx"])


def test_stratified_chronological_split_keeps_2013_only_in_validation():
    rows = []
    for year in (2009, 2010, 2011, 2012, 2013):
        for cls in (0, 1, 2, 3):
            for i in range(10):
                rows.append(
                    {
                        "utc": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(minutes=i),
                        "aurora_type": cls,
                    }
                )
    df = pd.DataFrame(rows)

    train, val = stratified_chronological_split(
        df,
        val_year=2013,
        train_fraction=0.5,
        val_fraction=0.5,
        random_state=42,
    )

    assert set(train["utc"].dt.year) == {2009, 2010, 2011, 2012}
    assert set(val["utc"].dt.year) == {2013}
    assert train["aurora_type"].value_counts().to_dict() == {0: 20, 1: 20, 2: 20, 3: 20}
    assert val["aurora_type"].value_counts().to_dict() == {0: 5, 1: 5, 2: 5, 3: 5}
