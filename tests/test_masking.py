import numpy as np
from astropy.wcs import WCS

from extractor.masking import build_catalog_star_mask, recover_empirical_psf_sigma


def _simple_tan_wcs(crval=(30.0, 60.0), crpix=(150.5, 100.5), scale_deg_per_px=2.0 / 3600.0):
    """Minimal TAN WCS for tests -- no SIP, matches the plate scale
    convention used elsewhere in this project's simple test fixtures."""
    w = WCS(naxis=2)
    w.wcs.crval = crval
    w.wcs.crpix = crpix
    w.wcs.cdelt = [-scale_deg_per_px, scale_deg_per_px]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_build_catalog_star_mask_places_disk_at_correct_pixel():
    shape = (200, 300)
    wcs = _simple_tan_wcs()
    # a star exactly at the reference pixel (CRVAL/CRPIX by construction)
    ra, dec = np.array([30.0]), np.array([60.0])
    mag = np.array([10.0])

    mask = build_catalog_star_mask(shape, wcs, ra, dec, mag, radius_px=5.0, mag_cut=13.0)

    assert mask.shape == shape
    # CRPIX is 1-indexed in the FITS/WCS convention; all_world2pix(...,0) returns 0-indexed
    expected_x, expected_y = 149.5, 99.5
    assert mask[int(round(expected_y)), int(round(expected_x))]
    # a point far outside the disk radius should not be masked
    assert not mask[0, 0]
    # roughly the right number of pixels for a radius-5 disk (pi*r^2 ~ 78.5)
    assert 60 <= mask.sum() <= 100


def test_build_catalog_star_mask_respects_magnitude_cut():
    shape = (200, 300)
    wcs = _simple_tan_wcs()
    ra, dec = np.array([30.0]), np.array([60.0])

    bright_mask = build_catalog_star_mask(shape, wcs, ra, dec, np.array([10.0]), radius_px=5.0, mag_cut=13.0)
    faint_mask = build_catalog_star_mask(shape, wcs, ra, dec, np.array([15.0]), radius_px=5.0, mag_cut=13.0)

    assert bright_mask.sum() > 0
    assert faint_mask.sum() == 0


def test_build_catalog_star_mask_empty_catalog_returns_all_false():
    shape = (50, 60)
    wcs = _simple_tan_wcs()
    mask = build_catalog_star_mask(shape, wcs, np.array([]), np.array([]), np.array([]), radius_px=5.0)
    assert mask.shape == shape
    assert not mask.any()


def _synthetic_star_field_image(shape=(300, 300), positions=((100, 100), (200, 150), (150, 220)),
                                 sigma=2.5, amplitude=2000.0, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    image = rng.normal(loc=50.0, scale=3.0, size=shape).astype(np.float64)
    yy, xx = np.mgrid[0:ny, 0:nx]
    for x0, y0 in positions:
        image += amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return image


def test_recover_empirical_psf_sigma_matches_injected_sigma():
    true_sigma = 2.5
    image = _synthetic_star_field_image(sigma=true_sigma)
    recovered = recover_empirical_psf_sigma(image, n_stars=3)
    assert abs(recovered - true_sigma) < 0.3
