"""
wcsangle.py — Local WCS orientation angle utilities.

Computes the image-plane angle of celestial north (and east) at any pixel
position by projecting a small sky offset back through the WCS.  The
primary use-case is evaluating the WCS orientation at the image center,
which is the X term in the polarization-angle error budget:

    θ_sky  =  θ_pixel  +  θ_north(x_c, y_c)  +  (residual distortion)

Algorithm
---------
For a pixel (x, y):
1. Convert (x, y) → sky coordinate c0 via ``wcs.pixel_to_world``.
2. Displace c0 by ``step_arcsec`` toward the north pole (or east) using
   ``SkyCoord.directional_offset_by``, which moves along a great circle
   on the sphere.  This is correct at all declinations; the naive
   ``dec + Δdec`` shortcut breaks near the poles and ignores the metric.
3. Project the displaced sky coordinate back to pixel via
   ``wcs.world_to_pixel``.
4. Return ``arctan2(Δy, Δx)`` in degrees, counter-clockwise from +x.

Angle convention
----------------
- ``north_angle_deg``: CCW from the +x (column) pixel axis.
- ``east_angle_deg``:  same convention, for celestial east (PA = 90°).
- Both are expressed in degrees, range (−180°, 180°].

On uncertainty
--------------
``north_angle_uncertainty_deg`` is intentionally ``None`` in all returned
objects.  A single WCS solution from nova.astrometry.net carries no
internal covariance for the local orientation derivative.  To estimate
uncertainty, run a bootstrap over WCS re-solves on random 80% subsets
of the matched-source list, call :func:`center_wcs_angle_metrics` on
each, and take the standard deviation of the resulting ``north_angle_deg``
values using :func:`angle_diff_deg` to avoid wrap-point artifacts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.wcs import WCS


# ---------------------------------------------------------------------------
# Angle arithmetic helper
# ---------------------------------------------------------------------------

def angle_diff_deg(a, b):
    """Signed circular difference ``a − b`` mapped to (−180°, 180°].

    Works element-wise when *a* or *b* are arrays.  Safe for angles near
    the 0°/360° wrap point — use this instead of plain subtraction when
    computing scatter or residuals of angular quantities.

    Examples
    --------
    >>> angle_diff_deg(5.0, 355.0)   # not -350
    10.0
    >>> angle_diff_deg(-179.0, 179.0)
    2.0
    """
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Local north / east angles
# ---------------------------------------------------------------------------

def local_north_angle_deg(
    wcs: WCS,
    x: float,
    y: float,
    step_arcsec: float = 10.0,
) -> float:
    """Image-plane angle of celestial north at pixel ``(x, y)``.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Plate-solved WCS; SIP distortion is applied automatically via
        ``pixel_to_world`` / ``world_to_pixel``.
    x, y : float
        Pixel coordinates (0-based; x = column, y = row).
    step_arcsec : float
        Sky offset used to define "north".  10 arcsec is well-behaved for
        typical wide-field astrometry (pixel scale 1–5 arcsec/px).
        Increase for coarser WCS or reduce for very fine grids.

    Returns
    -------
    float
        Degrees CCW from the +x pixel axis.  Range: (−180°, 180°].
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c0 = wcs.pixel_to_world(x, y)
        c_north = c0.directional_offset_by(
            position_angle=0.0 * u.deg,
            separation=step_arcsec * u.arcsec,
        )
        x1, y1 = wcs.world_to_pixel(c_north)
    return float(np.degrees(np.arctan2(float(y1) - y, float(x1) - x)))


def local_east_angle_deg(
    wcs: WCS,
    x: float,
    y: float,
    step_arcsec: float = 10.0,
) -> float:
    """Image-plane angle of celestial east at pixel ``(x, y)``.

    Celestial east is at position angle 90° from north (standard
    astronomical convention).  Returned in the same image-plane convention
    as :func:`local_north_angle_deg`.

    Returns
    -------
    float
        Degrees CCW from the +x pixel axis.  Range: (−180°, 180°].
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c0 = wcs.pixel_to_world(x, y)
        c_east = c0.directional_offset_by(
            position_angle=90.0 * u.deg,
            separation=step_arcsec * u.arcsec,
        )
        x1, y1 = wcs.world_to_pixel(c_east)
    return float(np.degrees(np.arctan2(float(y1) - y, float(x1) - x)))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class WcsAngleMetrics:
    """WCS orientation metrics evaluated at a single image position.

    Attributes
    ----------
    x_center, y_center : float
        Pixel position (0-based) at which the metrics are evaluated.
    ra_deg, dec_deg : float
        Sky coordinates at the evaluation point.
    north_angle_deg : float
        Image-plane angle of celestial north (CCW from +x pixel axis).
        This is the WCS orientation term X in the polarization-angle
        error budget; it converts pixel-space trace angles to sky angles.
    east_angle_deg : float or None
        Image-plane angle of celestial east.  None if not requested.
    north_angle_uncertainty_deg : None
        Always None.  A single WCS solution carries no internal covariance
        for this local derivative.  Estimate uncertainty externally via
        a bootstrap over WCS re-solves; see module docstring.
    """
    x_center: float
    y_center: float
    ra_deg: float
    dec_deg: float
    north_angle_deg: float
    east_angle_deg: Optional[float] = None
    north_angle_uncertainty_deg: None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Convenience: image-center evaluation
# ---------------------------------------------------------------------------

def center_wcs_angle_metrics(
    wcs: WCS,
    image_shape: tuple[int, int],
    step_arcsec: float = 10.0,
    compute_east: bool = True,
) -> WcsAngleMetrics:
    """Evaluate WCS orientation metrics at the image center pixel.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Plate-solved WCS (SIP-aware).
    image_shape : (ny, nx)
        NumPy image shape (rows, columns).  The center pixel is
        ``x_c = (nx-1)/2``, ``y_c = (ny-1)/2``.
    step_arcsec : float
        Sky-offset step for the north/east direction probe.  Default 10".
    compute_east : bool
        If True (default), also compute ``east_angle_deg``.

    Returns
    -------
    WcsAngleMetrics
        ``north_angle_uncertainty_deg`` is always ``None``; see
        module docstring for the bootstrap procedure.

    Notes
    -----
    The returned ``north_angle_deg`` is a purely local WCS derivative at
    the image center.  The full distortion field enters through the WCS
    fit; this function reads off what that fit implies for the local
    orientation.  Spatial variation of the north angle (i.e., field-
    dependent distortion) is not captured here — that is the concern of
    ``wcs_geometry.pixel_angle_to_sky_angle``.
    """
    ny, nx = image_shape
    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sky_center = wcs.pixel_to_world(xc, yc)

    north = local_north_angle_deg(wcs, xc, yc, step_arcsec=step_arcsec)
    east = (
        local_east_angle_deg(wcs, xc, yc, step_arcsec=step_arcsec)
        if compute_east
        else None
    )

    return WcsAngleMetrics(
        x_center=xc,
        y_center=yc,
        ra_deg=float(sky_center.ra.deg),
        dec_deg=float(sky_center.dec.deg),
        north_angle_deg=north,
        east_angle_deg=east,
    )
