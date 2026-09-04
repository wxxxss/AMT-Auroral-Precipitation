import numpy as np

from data.preprocess_ssj import (
    ERG_PER_EV,
    classify_aurora_spectra,
    preprocess_ssj_arrays,
)


def test_classification_rules_match_manuscript_thresholds():
    spectra = np.full((4, 19), 1.0e6, dtype=float)
    avg_energy = np.array([500.0, 1500.0, 500.0, 500.0])

    # Background: peak < 1e5.
    spectra[0, :] = 5.0e4

    # Monoenergetic: narrow >=2e8 peak with two-sided decrease.
    spectra[1, :] = 1.0e6
    spectra[1, 9] = 3.0e8

    # Broadband: >=3 channels >=2e8 and not already monoenergetic.
    spectra[2, 7:10] = 2.5e8

    # Remaining valid non-background sample with average energy <1e4 -> diffuse.
    spectra[3, :] = 2.0e6

    labels = classify_aurora_spectra(spectra, avg_energy)

    assert labels.tolist() == [0, 2, 3, 1]


def test_preprocess_converts_flux_qc_filters_domain_and_folds_hemisphere():
    epoch = np.array(["2014-01-01T00:00", "2014-01-01T00:01", "2014-01-01T00:02"], dtype="datetime64[m]")
    mlat = np.array([70.0, -65.0, 40.0])
    mlt = np.array([12.0, 23.0, 5.0])
    ele_total_ev = np.array([1.0e11, 2.0e11, 1.0e11])
    ion_total_ev = np.array([1.0e10, 2.0e10, 1.0e10])
    diff = np.full((3, 19), 2.0e6)
    avg = np.full(3, 500.0)

    out = preprocess_ssj_arrays(
        epoch=epoch,
        mlat=mlat,
        mlt=mlt,
        ele_total_energy_flux_ev=ele_total_ev,
        ion_total_energy_flux_ev=ion_total_ev,
        ele_diff_energy_flux=diff,
        ele_avg_energy=avg,
        fold_hemispheres=True,
    )

    assert len(out) == 2
    assert out["mlat"].tolist() == [70.0, 65.0]
    assert out["mlt"].tolist() == [12.0, 23.0]
    assert out["src_hemi"].tolist() == ["N", "S"]
    assert out["aurora_type"].tolist() == [1, 1]
    assert np.isclose(out.loc[0, "ele_energy_flux"], 1.0e11 * np.pi * ERG_PER_EV)
    assert np.isclose(out.loc[0, "ion_energy_flux"], 1.0e10 * np.pi * ERG_PER_EV)
