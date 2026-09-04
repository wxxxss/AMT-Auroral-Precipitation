import numpy as np

from evaluation.evaluate_hp_timing import (
    EARTH_RADIUS_CM,
    hemispheric_power_gw,
    spherical_area_elements_cm2,
)


def test_spherical_area_elements_have_grid_shape_and_positive_area():
    mlat = np.linspace(50.0, 90.0, 80)
    mlt = np.linspace(0.0, 24.0, 144)
    area = spherical_area_elements_cm2(mlat, mlt)

    assert area.shape == (80, 144)
    assert np.all(area >= 0.0)
    assert np.isclose(area[-1].max(), 0.0, atol=1e-6 * EARTH_RADIUS_CM**2)


def test_uniform_flux_is_integrated_with_erg_per_second_to_gw_conversion():
    mlat = np.linspace(50.0, 90.0, 80)
    mlt = np.linspace(0.0, 24.0, 144)
    area = spherical_area_elements_cm2(mlat, mlt)
    flux = np.ones_like(area)

    hp = hemispheric_power_gw(flux, area)

    assert np.isclose(hp, area.sum() * 1.0e-16)
