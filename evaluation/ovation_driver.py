"""OVATION-Prime driver utilities used in the revised manuscript.

The four-hour weighted Newell coupling implementation follows the weighting
rule used by ``auroramaps.util.calc_avg_solarwind_predstorm``: solar wind is
first aggregated to hourly means, the Newell coupling function is evaluated
for each hour, and the current hour plus the preceding three hours are
combined with weights ``a, 0.65, 0.65^2, 0.65^3``. Here ``a`` is the fraction
of the current hour that has elapsed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_newell_coupling(by: float, bz: float, speed: float) -> float:
    """Return the Newell coupling function for scalar By, Bz and speed."""
    bt = math.sqrt(by * by + bz * bz)
    if bt == 0 or not all(np.isfinite([by, bz, speed])):
        return float("nan")
    clock = math.atan2(by, 0.001 if bz == 0 else bz)
    if bt * math.cos(clock) * bz < 0:
        clock += math.pi
    return (
        speed ** (4.0 / 3.0)
        * abs(math.sin(clock / 2.0)) ** (8.0 / 3.0)
        * bt ** (2.0 / 3.0)
    )


def build_ovation_weighted_ec(df_omni: pd.DataFrame, target_times) -> np.ndarray:
    """Compute the four-hour weighted OVATION coupling driver.

    Parameters
    ----------
    df_omni:
        Dataframe with ``utc``, ``Bx``, ``By``, ``Bz``, ``Vx``, ``Vy``, and
        ``Vz`` columns. The dataframe should cover at least three full hours
        before every requested target time.
    target_times:
        Iterable of timestamps for which the driver should be evaluated.

    Returns
    -------
    numpy.ndarray
        Weighted Newell coupling values. Targets without a complete preceding
        three-hour history are returned as NaN.
    """
    required = {"utc", "Bx", "By", "Bz", "Vx", "Vy", "Vz"}
    missing = sorted(required.difference(df_omni.columns))
    if missing:
        raise ValueError(f"Missing OMNI columns: {missing}")

    sw = df_omni[list(required)].copy()
    sw["utc"] = pd.to_datetime(sw["utc"])
    sw = sw.sort_values("utc").drop_duplicates("utc", keep="first")
    sw["V"] = np.sqrt(sw["Vx"] ** 2 + sw["Vy"] ** 2 + sw["Vz"] ** 2)

    hourly = sw.set_index("utc")[["Bx", "By", "Bz", "V"]].resample("1h").mean()
    hourly["ec"] = np.asarray(
        [calculate_newell_coupling(by, bz, speed) for by, bz, speed in zip(hourly["By"], hourly["Bz"], hourly["V"])],
        dtype=float,
    )

    targets = pd.DatetimeIndex(pd.to_datetime(list(target_times)))
    target_hours = targets.floor("h")
    pos = hourly.index.get_indexer(target_hours)
    valid = pos >= 3
    result = np.full(len(targets), np.nan, dtype=float)
    if not valid.any():
        return result

    fraction = ((targets - target_hours) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    weights = np.column_stack(
        [
            fraction,
            np.full(len(targets), 0.65, dtype=float),
            np.full(len(targets), 0.65**2, dtype=float),
            np.full(len(targets), 0.65**3, dtype=float),
        ]
    )

    ec_hourly = hourly["ec"].to_numpy(dtype=float)
    p = pos[valid]
    ec4 = np.column_stack(
        [ec_hourly[p], ec_hourly[p - 1], ec_hourly[p - 2], ec_hourly[p - 3]]
    )
    result[valid] = np.nansum(ec4 * weights[valid], axis=1) / np.sum(weights[valid], axis=1)
    return result
