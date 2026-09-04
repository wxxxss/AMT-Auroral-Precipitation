#!/usr/bin/env python3
"""DMSP/SSJ preprocessing used by the revised AMT manuscript.

This public module implements the manuscript-described SSJ target pipeline:
classification of electron spectra into background/diffuse/monoenergetic/
broadband populations, conversion of electron and ion total energy flux to
``erg cm^-2 s^-1``, quality control, the |MLAT|=50--90 deg analysis domain,
and the hemispheric folding convention used by AMT.

Southern-Hemisphere seasonal-phase, dipole-tilt, and illumination corrections
are applied later during AMT feature construction in ``data.dataset_v4``. This
module records the source hemisphere and leaves MLT unchanged so those feature
corrections can be applied without altering the real UTC used for OMNI matching.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ERG_PER_EV = 1.602e-12
BACKGROUND_PEAK_MAX = 1.0e5
HIGH_FLUX_THRESHOLD = 2.0e8
MONO_AVG_ENERGY_MIN_EV = 1000.0
BROADBAND_AVG_ENERGY_MIN_EV = 100.0
DIFFUSE_AVG_ENERGY_MAX_EV = 1.0e4
MONO_SIDE_FRACTION = 0.4
TOTAL_FLUX_MAX = 1000.0
MLAT_MIN = 50.0
MLAT_MAX = 90.0


def classify_aurora_spectra(
    ele_diff_flux: np.ndarray,
    ele_avg_energy: np.ndarray,
) -> np.ndarray:
    """Classify SSJ electron spectra using the manuscript criteria.

    Returns integer labels ``0=background, 1=diffuse, 2=monoenergetic,
    3=broadband`` and ``-1`` for spectra that satisfy no retained class.

    The manuscript specifies a two-sided spectral decrease for monoenergetic
    precipitation. The working SSJ classifier operationalizes that condition
    by requiring the peak to be away from the two edge channels and by finding
    a value below 40% of the peak within the two adjacent channels on each side.
    """
    spectra = np.asarray(ele_diff_flux, dtype=float)
    avg_energy = np.asarray(ele_avg_energy, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("ele_diff_flux must be a 2-D array")
    if avg_energy.ndim != 1 or avg_energy.shape[0] != spectra.shape[0]:
        raise ValueError("ele_avg_energy must be 1-D with one value per spectrum")
    if spectra.shape[1] < 5:
        raise ValueError("at least five differential-energy channels are required")

    n = spectra.shape[0]
    labels = np.full(n, -1, dtype=np.int8)
    finite_spectrum = np.all(np.isfinite(spectra), axis=1)
    finite_avg = np.isfinite(avg_energy)
    usable = finite_spectrum & finite_avg
    if not usable.any():
        return labels

    peak_vals = np.full(n, np.nan, dtype=float)
    peak_idxs = np.zeros(n, dtype=int)
    peak_vals[usable] = np.max(spectra[usable], axis=1)
    peak_idxs[usable] = np.argmax(spectra[usable], axis=1)

    background = usable & (peak_vals < BACKGROUND_PEAK_MAX)
    labels[background] = 0
    non_background = usable & ~background

    mono_candidate = (
        non_background
        & (peak_vals >= HIGH_FLUX_THRESHOLD)
        & (avg_energy >= MONO_AVG_ENERGY_MIN_EV)
        & (peak_idxs >= 2)
        & (peak_idxs <= spectra.shape[1] - 3)
    )
    mono_rows = np.flatnonzero(mono_candidate)
    if mono_rows.size:
        p = peak_idxs[mono_rows]
        peak = peak_vals[mono_rows]
        left_min = np.minimum(spectra[mono_rows, p - 1], spectra[mono_rows, p - 2])
        right_min = np.minimum(spectra[mono_rows, p + 1], spectra[mono_rows, p + 2])
        is_mono = (left_min < MONO_SIDE_FRACTION * peak) & (
            right_min < MONO_SIDE_FRACTION * peak
        )
        labels[mono_rows[is_mono]] = 2

    broadband = (
        non_background
        & (labels != 2)
        & ((spectra >= HIGH_FLUX_THRESHOLD).sum(axis=1) >= 3)
        & (avg_energy >= BROADBAND_AVG_ENERGY_MIN_EV)
    )
    labels[broadband] = 3

    diffuse = (
        non_background
        & (labels == -1)
        & (avg_energy < DIFFUSE_AVG_ENERGY_MAX_EV)
    )
    labels[diffuse] = 1
    return labels


def preprocess_ssj_arrays(
    *,
    epoch,
    mlat,
    mlt,
    ele_total_energy_flux_ev,
    ion_total_energy_flux_ev,
    ele_diff_energy_flux,
    ele_avg_energy,
    fold_hemispheres: bool = True,
) -> pd.DataFrame:
    """Convert, quality-control, classify, and spatially filter SSJ arrays."""
    utc = pd.to_datetime(np.asarray(epoch))
    mlat_arr = np.asarray(mlat, dtype=float)
    mlt_arr = np.asarray(mlt, dtype=float)
    ele_ev = np.asarray(ele_total_energy_flux_ev, dtype=float)
    ion_ev = np.asarray(ion_total_energy_flux_ev, dtype=float)
    diff = np.asarray(ele_diff_energy_flux, dtype=float)
    avg = np.asarray(ele_avg_energy, dtype=float)

    n = len(utc)
    one_d = [mlat_arr, mlt_arr, ele_ev, ion_ev, avg]
    if any(arr.ndim != 1 or len(arr) != n for arr in one_d):
        raise ValueError("all one-dimensional SSJ arrays must have the same length")
    if diff.ndim != 2 or diff.shape[0] != n:
        raise ValueError("ele_diff_energy_flux must contain one spectrum per timestamp")

    ele_flux = ele_ev * np.pi * ERG_PER_EV
    ion_flux = ion_ev * np.pi * ERG_PER_EV
    total_flux = ele_flux + ion_flux

    qc = (
        np.isfinite(total_flux)
        & np.isfinite(ele_flux)
        & np.isfinite(ion_flux)
        & (total_flux >= 0.0)
        & (total_flux <= TOTAL_FLUX_MAX)
        & np.isfinite(mlat_arr)
        & np.isfinite(mlt_arr)
    )
    if not qc.any():
        return pd.DataFrame(
            columns=[
                "utc",
                "mlat",
                "mlt",
                "total_energy_flux",
                "ele_energy_flux",
                "ion_energy_flux",
                "aurora_type",
                "src_hemi",
            ]
        )

    labels = classify_aurora_spectra(diff[qc], avg[qc])
    df = pd.DataFrame(
        {
            "utc": utc[qc],
            "mlat": mlat_arr[qc],
            "mlt": mlt_arr[qc],
            "total_energy_flux": total_flux[qc],
            "ele_energy_flux": ele_flux[qc],
            "ion_energy_flux": ion_flux[qc],
            "aurora_type": labels,
        }
    )

    in_domain = df["mlat"].abs().between(MLAT_MIN, MLAT_MAX, inclusive="both")
    df = df.loc[in_domain & (df["aurora_type"] >= 0)].copy()
    if df.empty:
        df["src_hemi"] = pd.Series(dtype="object")
        return df.reset_index(drop=True)

    df["src_hemi"] = np.where(df["mlat"] >= 0.0, "N", "S")
    if fold_hemispheres:
        df["mlat"] = df["mlat"].abs()
    return df.reset_index(drop=True)


def read_ssj_cdf(path: str | Path, *, fold_hemispheres: bool = True) -> pd.DataFrame:
    """Read and preprocess one DMSP SSJ/5 CDF file."""
    try:
        import cdflib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cdflib is required to read raw DMSP/SSJ CDF files") from exc

    cdf = cdflib.CDF(str(path))
    return preprocess_ssj_arrays(
        epoch=cdflib.cdfepoch.to_datetime(cdf.varget("Epoch")),
        mlat=cdf.varget("SC_AACGM_LAT"),
        mlt=cdf.varget("SC_AACGM_LTIME"),
        ele_total_energy_flux_ev=cdf.varget("ELE_TOTAL_ENERGY_FLUX"),
        ion_total_energy_flux_ev=cdf.varget("ION_TOTAL_ENERGY_FLUX"),
        ele_diff_energy_flux=cdf.varget("ELE_DIFF_ENERGY_FLUX"),
        ele_avg_energy=cdf.varget("ELE_AVG_ENERGY"),
        fold_hemispheres=fold_hemispheres,
    )


def build_ssj_parquet(
    cdf_root: str | Path,
    *,
    satellites: tuple[str, ...] | list[str],
    start_year: int,
    end_year: int,
    fold_hemispheres: bool = True,
) -> pd.DataFrame:
    """Preprocess a ``satellite/year/*.cdf`` DMSP tree into one dataframe."""
    root = Path(cdf_root)
    parts: list[pd.DataFrame] = []
    for satellite in satellites:
        sat = str(satellite).lower()
        for year in range(int(start_year), int(end_year) + 1):
            year_dir = root / sat / str(year)
            if not year_dir.is_dir():
                continue
            for cdf_path in sorted(year_dir.glob("*.cdf")):
                df = read_ssj_cdf(cdf_path, fold_hemispheres=fold_hemispheres)
                if df.empty:
                    continue
                df["satellite"] = sat.upper()
                parts.append(df)
    if not parts:
        raise FileNotFoundError(
            f"No usable DMSP/SSJ CDF records found under {root} for "
            f"{list(satellites)}, {start_year}--{end_year}"
        )
    return pd.concat(parts, ignore_index=True).sort_values("utc").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdf-root", required=True, help="Root containing f16/f17/f18 year directories")
    p.add_argument("--satellites", nargs="+", default=["f16", "f17", "f18"])
    p.add_argument("--start-year", type=int, default=2009)
    p.add_argument("--end-year", type=int, default=2014)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--no-fold",
        action="store_true",
        help="Keep signed MLAT instead of applying the manuscript hemispheric fold",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    df = build_ssj_parquet(
        args.cdf_root,
        satellites=args.satellites,
        start_year=args.start_year,
        end_year=args.end_year,
        fold_hemispheres=not args.no_fold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"wrote {len(df):,} DMSP/SSJ rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
