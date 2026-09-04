"""Feature construction and PyTorch dataset for the final AMT model.

The production manuscript model uses a 120-min solar-wind history, yielding a
116-dimensional driver vector plus nine spatial/temporal skip features. The
same implementation also supports the controlled 60/90/120/180/240-min
history-sensitivity experiment by changing only the exposed lag horizon.

Raw data are not bundled with this repository; callers provide a dataframe
containing the time-matched OMNI and DMSP/SSJ records.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


OMNI_BASE_VARS = ["Bx", "By", "Bz", "Vx", "Vy", "Vz", "P_dyn"]
DERIVED_SCALAR_NAMES = ["Newell_log", "Ey_conv", "sin_clock", "Bz_south_log"]
CURRENT_DERIVED_NAMES = [
    "Newell_log",
    "Ey_conv",
    "sin_clock",
    "cos_clock",
    "Bz_south_log",
    "Bt",
    "Bmag",
    "Vmag",
    "Akasofu_log",
]
SKIP_FEATURE_NAMES = [
    "mlat_scaled",
    "sin_mlt",
    "cos_mlt",
    "dipole_tilt",
    "cos_sza",
    "sin_doy",
    "cos_doy",
    "sin_hour",
    "cos_hour",
]
TARGET_FLOOR = 1e-6
CADENCE_MINUTES = 5
PRODUCTION_HISTORY_MINUTES = 120
SUPPORTED_HISTORY_MINUTES = (60, 90, 120, 180, 240)


def lag_minutes_for_history(history_minutes: int) -> list[int]:
    history_minutes = int(history_minutes)
    if history_minutes not in SUPPORTED_HISTORY_MINUTES:
        raise ValueError(
            f"history_minutes must be one of {SUPPORTED_HISTORY_MINUTES}; "
            f"got {history_minutes}"
        )
    return list(range(CADENCE_MINUTES, history_minutes + 1, CADENCE_MINUTES))


def sw_dim_for_history(history_minutes: int) -> int:
    """Return solar-wind input dimension for the controlled history horizon."""
    return 20 + 4 * len(lag_minutes_for_history(history_minutes))


def compute_derived_block(Bx, By, Bz, Vx, Vy, Vz, P_dyn=None, eps=1e-6):
    """Return physically motivated scalar descriptors for one time slice."""
    Bx = np.asarray(Bx, dtype=np.float32)
    By = np.asarray(By, dtype=np.float32)
    Bz = np.asarray(Bz, dtype=np.float32)
    Vx = np.asarray(Vx, dtype=np.float32)
    Vy = np.asarray(Vy, dtype=np.float32)
    Vz = np.asarray(Vz, dtype=np.float32)

    Bt = np.sqrt(By * By + Bz * Bz) + eps
    Bmag = np.sqrt(Bx * Bx + By * By + Bz * Bz)
    Vmag = np.sqrt(Vx * Vx + Vy * Vy + Vz * Vz) + eps
    sin_clock = By / Bt
    cos_clock = Bz / Bt
    half_angle = np.sqrt(np.clip((Bt - Bz) / (2.0 * Bt), 0.0, 1.0))

    newell_raw = (
        np.power(Vmag, 4.0 / 3.0)
        * np.power(Bt, 2.0 / 3.0)
        * np.power(half_angle, 8.0 / 3.0)
    )
    newell_log = np.log1p(newell_raw)
    ey_conv = -Vx * Bz
    bz_south_log = np.log1p(np.maximum(-Bz, 0.0))
    akasofu_raw = Vmag * Bmag * Bmag * (half_angle**4)
    akasofu_log = np.log1p(akasofu_raw)

    out = {
        "Newell_log": newell_log.astype(np.float32),
        "Ey_conv": ey_conv.astype(np.float32),
        "sin_clock": sin_clock.astype(np.float32),
        "cos_clock": cos_clock.astype(np.float32),
        "Bz_south_log": bz_south_log.astype(np.float32),
        "Bt": Bt.astype(np.float32),
        "Bmag": Bmag.astype(np.float32),
        "Vmag": Vmag.astype(np.float32),
        "Akasofu_log": akasofu_log.astype(np.float32),
    }
    if P_dyn is not None:
        out["P_dyn"] = np.asarray(P_dyn, dtype=np.float32)
    return out


def _utc_to_days_since_j2000(utc_like):
    ns = np.asarray(pd.to_datetime(utc_like).values, dtype="datetime64[ns]").astype(np.int64)
    j2000_ns = pd.Timestamp("2000-01-01 12:00:00").value
    return (ns - j2000_ns) / (86400.0 * 1e9)


def compute_dipole_tilt_rad(utc_series):
    """Approximate geomagnetic dipole tilt in radians for each UTC."""
    d_days = _utc_to_days_since_j2000(utc_series)
    L_deg = (280.460 + 0.9856474 * d_days) % 360.0
    g_rad = np.radians((357.528 + 0.9856003 * d_days) % 360.0)
    lam_rad = np.radians(L_deg + 1.915 * np.sin(g_rad) + 0.020 * np.sin(2 * g_rad))
    eps_rad = np.radians(23.439 - 0.0000004 * d_days)
    decl_rad = np.arcsin(np.sin(eps_rad) * np.sin(lam_rad))
    ra_rad = np.arctan2(np.cos(eps_rad) * np.sin(lam_rad), np.cos(lam_rad))
    gmst_hours = (18.697374558 + 24.06570982441908 * d_days) % 24.0
    gmst_rad = np.radians(gmst_hours * 15.0)

    pole_lat_rad = np.radians(80.5)
    pole_lon_rad = np.radians(-72.2)
    hour_angle = gmst_rad + pole_lon_rad - ra_rad
    sin_tilt = (
        np.sin(pole_lat_rad) * np.sin(decl_rad)
        + np.cos(pole_lat_rad) * np.cos(decl_rad) * np.cos(hour_angle)
    )
    return np.arcsin(np.clip(sin_tilt, -1.0, 1.0)).astype(np.float32)


def compute_cos_sza(mlat_deg, mlt, utc_series):
    """Approximate cosine of the solar zenith angle at magnetic coordinates."""
    utc = pd.to_datetime(utc_series)
    doy = utc.dt.dayofyear.to_numpy(dtype=np.float32)
    phase = 2.0 * np.pi * (doy - 172.0) / 365.25
    solar_decl = np.radians(23.44) * np.cos(phase)
    mlt_angle = 2.0 * np.pi * (np.asarray(mlt, dtype=np.float32) - 12.0) / 24.0
    mlat_rad = np.radians(np.asarray(mlat_deg, dtype=np.float32))
    return (
        np.sin(mlat_rad) * np.sin(solar_decl)
        + np.cos(mlat_rad) * np.cos(solar_decl) * np.cos(mlt_angle)
    ).astype(np.float32)


def build_sw_feature_names(history_minutes=PRODUCTION_HISTORY_MINUTES):
    """Return AMT solar-wind feature names for a supported history horizon."""
    lag_minutes = lag_minutes_for_history(history_minutes)
    cols = list(OMNI_BASE_VARS)
    cols.extend(CURRENT_DERIVED_NAMES)
    for name in DERIVED_SCALAR_NAMES:
        cols.extend(f"{name}_lag_{m}" for m in lag_minutes)
    cols.extend(
        ["Newell_log_avg_1h", "Ey_conv_avg_1h", "Bz_south_log_avg_1h", "dPdyn_dt"]
    )
    return cols


class AuroraMultiTaskDataset_V4(Dataset):
    """Dataset used for AMT training and inference.

    Parameters
    ----------
    df:
        Dataframe containing ``utc``, ``mlat``, ``mlt``, current OMNI values,
        raw 5-min OMNI lag columns through the selected history horizon, and the
        DMSP/SSJ target columns ``aurora_type``, ``ele_energy_flux``, and
        ``ion_energy_flux``.
    is_train:
        If True, fit a StandardScaler when ``scaler_path`` does not exist.
    scaler_path:
        Serialized StandardScaler path. In validation/inference mode this file
        must already exist.
    history_minutes:
        One of 60, 90, 120, 180, or 240. Production AMT uses 120 min.
    """

    def __init__(
        self,
        df,
        is_train=True,
        scaler_path=None,
        history_minutes=PRODUCTION_HISTORY_MINUTES,
    ):
        self.history_minutes = int(history_minutes)
        self.lag_minutes = lag_minutes_for_history(self.history_minutes)
        self.df = df.copy()
        self._prepare_targets()
        self._prepare_spatial_features()
        self._prepare_derived_features()

        self.sw_cols = build_sw_feature_names(self.history_minutes)
        self.skip_cols = list(SKIP_FEATURE_NAMES)
        missing_sw = [c for c in self.sw_cols if c not in self.df.columns]
        missing_skip = [c for c in self.skip_cols if c not in self.df.columns]
        if missing_sw or missing_skip:
            raise ValueError(f"Missing features: sw={missing_sw}, skip={missing_skip}")
        expected_dim = sw_dim_for_history(self.history_minutes)
        if len(self.sw_cols) != expected_dim:
            raise RuntimeError(
                f"Expected {expected_dim} solar-wind features for "
                f"{self.history_minutes} min, got {len(self.sw_cols)}"
            )

        self._fit_or_load_scaler(is_train=is_train, scaler_path=scaler_path)
        self.target_cols = [
            "target_diffuse_log",
            "target_mono_log",
            "target_broadband_log",
            "target_ion_log",
        ]
        self.X_sw_tensor = torch.tensor(self.df[self.sw_cols].to_numpy(), dtype=torch.float32)
        self.X_skip_tensor = torch.tensor(self.df[self.skip_cols].to_numpy(), dtype=torch.float32)
        self.Y_tensor = torch.tensor(self.df[self.target_cols].to_numpy(), dtype=torch.float32)

        aurora_type = self.df["aurora_type"].fillna(0).astype(int).to_numpy()
        ion_active = (
            self.df["ion_energy_flux"].fillna(0.0).to_numpy() > 1e-5
        ).astype(np.float32)
        self.aurora_type_tensor = torch.tensor(aurora_type, dtype=torch.long)
        self.ion_active_tensor = torch.tensor(ion_active, dtype=torch.float32)

    def _prepare_targets(self):
        required = {"aurora_type", "ele_energy_flux", "ion_energy_flux"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing target columns: {sorted(missing)}")

        self.df["target_diffuse"] = np.where(
            self.df["aurora_type"] == 1, self.df["ele_energy_flux"], 0.0
        )
        self.df["target_mono"] = np.where(
            self.df["aurora_type"] == 2, self.df["ele_energy_flux"], 0.0
        )
        self.df["target_broadband"] = np.where(
            self.df["aurora_type"] == 3, self.df["ele_energy_flux"], 0.0
        )
        self.df["target_ion"] = self.df["ion_energy_flux"]
        for col in ["target_diffuse", "target_mono", "target_broadband", "target_ion"]:
            values = self.df[col].fillna(0.0).to_numpy(dtype=float)
            self.df[f"{col}_log"] = np.log10(values + TARGET_FLOOR)

    def _prepare_spatial_features(self):
        if not {"mlat", "mlt"}.issubset(self.df.columns):
            raise ValueError("Input dataframe must contain mlat and mlt")
        self.df["sin_mlt"] = np.sin(self.df["mlt"] * np.pi / 12.0)
        self.df["cos_mlt"] = np.cos(self.df["mlt"] * np.pi / 12.0)
        self.df["mlat_scaled"] = (self.df["mlat"] - 50.0) / 40.0

    def _prepare_derived_features(self):
        df = self.df
        required_current = set(OMNI_BASE_VARS + ["utc"])
        missing_current = required_current.difference(df.columns)
        if missing_current:
            raise ValueError(f"Missing current OMNI/time columns: {sorted(missing_current)}")

        curr = compute_derived_block(
            *(df[c].to_numpy(dtype=np.float32) for c in OMNI_BASE_VARS[:6])
        )
        for key, value in curr.items():
            df[key] = value

        for minute in self.lag_minutes:
            needed = [
                f"{v}_lag_{minute}" for v in ("Bx", "By", "Bz", "Vx", "Vy", "Vz")
            ]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise ValueError(f"Missing raw OMNI lag columns for {minute} min: {missing}")
            lag = compute_derived_block(*(df[c].to_numpy(dtype=np.float32) for c in needed))
            for name in DERIVED_SCALAR_NAMES:
                df[f"{name}_lag_{minute}"] = lag[name]

        # All supported histories are at least 60 min, so these aggregate features
        # are defined identically for every controlled sensitivity run.
        for name in ["Newell_log", "Ey_conv", "Bz_south_log"]:
            one_hour = [name] + [f"{name}_lag_{m}" for m in range(5, 56, 5)]
            df[f"{name}_avg_1h"] = df[one_hour].mean(axis=1).astype(np.float32)

        if "P_dyn_lag_5" not in df.columns:
            raise ValueError("Missing P_dyn_lag_5 required for dPdyn_dt")
        df["dPdyn_dt"] = ((df["P_dyn"] - df["P_dyn_lag_5"]) / 5.0).astype(np.float32)

        utc = pd.to_datetime(df["utc"])
        raw_doy = utc.dt.dayofyear.to_numpy(dtype=np.float32)
        season_doy = raw_doy.copy()
        is_south = (
            (df["src_hemi"].astype(str).str.upper() == "S").to_numpy()
            if "src_hemi" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        if is_south.any():
            season_doy[is_south] = ((raw_doy[is_south] - 1.0 + 182.0) % 365.0) + 1.0

        df["dipole_tilt"] = compute_dipole_tilt_rad(df["utc"])
        if is_south.any():
            df.loc[is_south, "dipole_tilt"] = -df.loc[is_south, "dipole_tilt"].to_numpy()

        doy_phase = 2.0 * np.pi * (season_doy - 172.0) / 365.25
        solar_decl = np.radians(23.44) * np.cos(doy_phase)
        mlt_angle = 2.0 * np.pi * (df["mlt"].to_numpy(dtype=np.float32) - 12.0) / 24.0
        mlat_rad = np.radians(df["mlat"].to_numpy(dtype=np.float32))
        df["cos_sza"] = (
            np.sin(mlat_rad) * np.sin(solar_decl)
            + np.cos(mlat_rad) * np.cos(solar_decl) * np.cos(mlt_angle)
        ).astype(np.float32)

        hour = (utc.dt.hour + utc.dt.minute / 60.0).to_numpy(dtype=np.float32)
        df["sin_doy"] = np.sin(2.0 * np.pi * season_doy / 365.25).astype(np.float32)
        df["cos_doy"] = np.cos(2.0 * np.pi * season_doy / 365.25).astype(np.float32)
        df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
        df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
        self.df = df

    def _fit_or_load_scaler(self, is_train, scaler_path):
        self.scaler = StandardScaler()
        scaler_path = Path(scaler_path) if scaler_path is not None else None
        if scaler_path is not None and scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            expected = getattr(self.scaler, "n_features_in_", len(self.sw_cols))
            if expected != len(self.sw_cols):
                raise ValueError(
                    f"Scaler expects {expected} features but AMT requires {len(self.sw_cols)}"
                )
            self.df[self.sw_cols] = self.scaler.transform(self.df[self.sw_cols].to_numpy())
            return

        if not is_train:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.df[self.sw_cols] = self.scaler.fit_transform(self.df[self.sw_cols].to_numpy())
        if scaler_path is not None:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.X_sw_tensor[idx],
            self.X_skip_tensor[idx],
            self.Y_tensor[idx],
            self.aurora_type_tensor[idx],
            self.ion_active_tensor[idx],
        )
