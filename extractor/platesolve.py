"""
platesolve.py — Phase 1 general plate-solving via nova.astrometry.net.

This module is the second step of the preliminary astrometry phase:
  stars.extract_stars()  →  platesolve()  →  working/general_platesolve/

It submits a source list, retrieves a broad field solution, and saves the
astrometry.net product bundle (WCS header, corr/axy/rdls/image_rd tables)
to a `working/` checkpoint directory.  The resulting WCS is suitable for
rough field identification and as a starting point for a later custom
distortion model; it is **not** the final precision WCS.

Astrometry.net product URLs
----------------------------
  corr      : https://nova.astrometry.net/corr_file/{job_id}     → corr.fits
  axy       : https://nova.astrometry.net/axy_file/{job_id}      → axy.fits
  rdls      : https://nova.astrometry.net/rdls_file/{job_id}     → rdls.fits
  image_rd  : https://nova.astrometry.net/image_rd_file/{job_id} → image-radec.fits
  new_image : https://nova.astrometry.net/new_fits_file/{job_id} → new-image.fits

Authentication
--------------
The corr_file endpoint requires a logged-in Django session cookie.  We use
requests.Session() so the cookie set during /api/login is preserved for all
subsequent downloads — including the corr file.

Usage
-----
    from extractor.platesolve import platesolve, has_sip, wcs_summary

    result = platesolve("image.fits", cache=Path("cache.pkl"))
    result.corr_table.colnames          # full correspondence table
    result.fetch_status                 # {"corr": "ok", "axy": "HTTP 404 …", …}
    result.job_id                       # astrometry.net job ID
"""

from __future__ import annotations

import io
import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table

from .stars import extract_stars, make_xylist

_API_URL          = "https://nova.astrometry.net/api"
_DEFAULT_TIMEOUT  = 300
_POLL_INTERVAL    = 5
_DEFAULT_KEY_FILE = Path(__file__).parent.parent / "astrometry_api.txt"
_SKIP_KEYS = frozenset({"SIMPLE", "BITPIX", "EXTEND", "END", "COMMENT", "HISTORY", ""})

_PRODUCT_URLS: dict[str, str] = {
    "corr":      "https://nova.astrometry.net/corr_file/{job_id}",
    "axy":       "https://nova.astrometry.net/axy_file/{job_id}",
    "rdls":      "https://nova.astrometry.net/rdls_file/{job_id}",
    "image_rd":  "https://nova.astrometry.net/image_rd_file/{job_id}",
    "new_image": "https://nova.astrometry.net/new_fits_file/{job_id}",
}
_TABLE_PRODUCTS = ("corr", "axy", "rdls", "image_rd")
_PRODUCT_FILENAMES: dict[str, str] = {
    "corr":      "corr.fits",
    "axy":       "axy.fits",
    "rdls":      "rdls.fits",
    "image_rd":  "image-radec.fits",
    "new_image": "new-image.fits",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PlatesolveResult:
    """Complete result of a plate-solve operation.

    Backward-compatible fields
    --------------------------
    header       : merged FITS header with WCS keywords from astrometry.net
    detected_x/y : pixel positions (0-based) of sources submitted for solving
    matched_x/y  : catalog-matched subset, derived from corr_table field_x/y

    New fields
    ----------
    submission_id      : nova.astrometry.net submission ID
    job_id             : nova.astrometry.net job ID
    status             : "success" | "failure" | "timeout" | "unknown"
    corr_table         : full correspondence table (corr.fits, all columns)
    axy_table          : sources detected by astrometry.net (axy.fits)
    rdls_table         : reference-catalog stars used in solve (rdls.fits)
    image_radec_table  : submitted sources projected to RA/Dec (image-radec.fits)
    new_image_hdul     : plate-solved FITS image (new-image.fits), if fetched
    wcs_header_raw     : raw WCS FITS header as returned before merging
    product_urls       : URLs attempted for each product
    fetch_status       : outcome for each product ("ok" or descriptive error)
    """
    header: fits.Header

    detected_x: np.ndarray = field(default_factory=lambda: np.array([]))
    detected_y: np.ndarray = field(default_factory=lambda: np.array([]))
    matched_x:  np.ndarray = field(default_factory=lambda: np.array([]))
    matched_y:  np.ndarray = field(default_factory=lambda: np.array([]))

    submission_id: Optional[int] = None
    job_id:        Optional[int] = None
    status:        str           = "unknown"

    corr_table:        Optional[Table] = None
    axy_table:         Optional[Table] = None
    rdls_table:        Optional[Table] = None
    image_radec_table: Optional[Table] = None
    new_image_hdul:    Optional[fits.HDUList] = None

    wcs_header_raw: Optional[fits.Header] = None

    product_urls:  dict = field(default_factory=dict)
    fetch_status:  dict = field(default_factory=dict)

    def __setstate__(self, state: dict) -> None:
        """Provide defaults for fields added after an older cache was written."""
        _defaults = {
            "submission_id":     None,
            "job_id":            None,
            "status":            "unknown",
            "axy_table":         None,
            "rdls_table":        None,
            "image_radec_table": None,
            "new_image_hdul":    None,
            "wcs_header_raw":    None,
            "product_urls":      {},
            "fetch_status":      {},
        }
        for k, v in _defaults.items():
            state.setdefault(k, v)
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# WCS diagnostic utilities
# ---------------------------------------------------------------------------

def has_sip(header: fits.Header) -> bool:
    """Return True if *header* contains SIP distortion coefficients."""
    return any(k.startswith(("A_", "B_", "AP_", "BP_")) for k in header.keys())


def wcs_summary(header: fits.Header) -> str:
    """Return a compact human-readable WCS summary for *header*."""
    lines = []
    for k in ("CTYPE1", "CTYPE2"):
        if k in header:
            lines.append(f"  {k:<8}: {header[k]}")
    for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"):
        if k in header:
            lines.append(f"  {k:<8}: {header[k]:.8g}")
    for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2",
              "CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2"):
        if k in header:
            lines.append(f"  {k:<8}: {header[k]:.6g}")
    if has_sip(header):
        order = max(header.get("A_ORDER", 0), header.get("B_ORDER", 0))
        lines.append(f"  SIP     : yes (order {order})")
    else:
        lines.append("  SIP     : no")
    return "\n".join(lines)


def _resolve_cache_path(filepath: Path, cache: Union[bool, Path]) -> Optional[Path]:
    if cache is False:
        return None
    if cache is True:
        return filepath.with_suffix("").with_suffix(".plsolve.pkl")
    return Path(cache)


# ---------------------------------------------------------------------------
# FITS product fetch helpers
# ---------------------------------------------------------------------------

def _download_fits_bytes(
    url: str,
    product: str,
    verbose: bool = False,
    http_sess: Optional[requests.Session] = None,
) -> tuple[Optional[bytes], str]:
    """Download a FITS binary product from *url*.

    Uses *http_sess* if provided (preserving login cookies), otherwise falls
    back to a plain requests.get().  Validates that the response body starts
    with ``b"SIMPLE"``; if the server returns an HTML page instead (e.g. a
    CAPTCHA or login redirect), the status message explains what happened.
    """
    _get = http_sess.get if http_sess is not None else requests.get
    try:
        r = _get(url, timeout=60)
        ct = r.headers.get("Content-Type", "?")
        status_line = f"HTTP {r.status_code}, Content-Type={ct}, url={r.url}"
        r.raise_for_status()
        raw = r.content

        if verbose:
            print(f"  {product}: {status_line}")
            print(f"  {product}: {len(raw)} bytes, first 16: {raw[:16]!r}")

        if not raw.startswith(b"SIMPLE"):
            idx = raw.find(b"SIMPLE")
            preview = raw[:300].decode("utf-8", errors="replace")
            return None, (
                f"response is not a FITS file (does not start with 'SIMPLE'); "
                f"SIMPLE found at byte {idx}; {status_line}; "
                f"first bytes: {raw[:80]!r}; "
                f"preview: {preview!r}"
            )

        return raw, "ok"

    except requests.HTTPError as exc:
        return None, f"HTTP {exc.response.status_code} at {url}"
    except requests.Timeout:
        return None, f"timeout after 60s at {url}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _fetch_fits_table_product(
    url: str,
    product: str,
    verbose: bool = False,
    http_sess: Optional[requests.Session] = None,
) -> tuple[Optional[Table], Optional[bytes], str]:
    """Fetch and parse a FITS binary table product.

    Returns ``(table, raw_bytes, status_str)``.  ``raw_bytes`` is returned
    even on parse failure so it can be saved to disk for manual inspection.
    The table HDU is found by type (``BinTableHDU`` or ``TableHDU``) rather
    than by index, so primary-HDU-empty files work correctly.
    """
    raw, fetch_msg = _download_fits_bytes(url, product, verbose=verbose,
                                           http_sess=http_sess)
    if raw is None:
        return None, None, fetch_msg

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(io.BytesIO(raw)) as hdul:
                if verbose:
                    hdu_desc = ", ".join(
                        f"HDU{i}={type(h).__name__}" for i, h in enumerate(hdul)
                    )
                    print(f"  {product}: {hdu_desc}")

                # Find first binary or ASCII table extension — do NOT use hdul[0]
                table_hdu = next(
                    (h for h in hdul if isinstance(h, (fits.BinTableHDU, fits.TableHDU))),
                    None,
                )
                if table_hdu is None:
                    hdu_types = [type(h).__name__ for h in hdul]
                    return None, raw, (
                        f"FITS opened successfully but no BinTableHDU or TableHDU found; "
                        f"HDUs: {hdu_types}"
                    )

                table = Table(table_hdu.data)

    except Exception as exc:
        return None, raw, f"parse error: {type(exc).__name__}: {exc}"

    if verbose:
        print(f"  {product}: {len(table)} rows | cols: {table.colnames}")

    return table, raw, "ok"


def _col_array(table: Table, *names: str) -> np.ndarray:
    """Return the first matching column as a float array.

    Column lookup is case-insensitive and strips trailing whitespace
    (FITS TTYPE values can have trailing spaces).
    """
    lower_map = {c.strip().lower(): c for c in table.colnames}
    for n in names:
        key = lower_map.get(n.strip().lower())
        if key is not None:
            return np.asarray(table[key], dtype=float)
    return np.array([])


def get_col(table: Table, name: str):
    """Return column *name* from *table*, case-insensitive.  Returns None if missing."""
    lower_map = {c.strip().lower(): c for c in table.colnames}
    key = lower_map.get(name.strip().lower())
    return table[key] if key is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def platesolve(
    filepath: Union[str, Path],
    write: bool = False,
    use_source_list: bool = True,
    max_sources: int = 300,
    mask_spectra: bool = True,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    search_radius_deg: float = 5.0,
    scale_arcsec_per_px: Optional[float] = None,
    scale_tolerance: float = 0.25,
    tweak_order: Optional[int] = None,
    crpix_center: bool = False,
    api_key_file: Optional[Union[str, Path]] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    verbose: bool = True,
    cache: Union[bool, Path] = False,
    fetch_products: bool = True,
    save_products_dir: Optional[Union[str, Path]] = None,
) -> Optional[PlatesolveResult]:
    """Plate-solve a FITS image via nova.astrometry.net.

    Parameters
    ----------
    filepath : path-like
        FITS image to solve.
    write : bool
        Write WCS back into the FITS file on disk. Default False.
    use_source_list : bool
        Submit a source xylist instead of the full image. Default True.
    max_sources : int
        Maximum number of sources in the xylist. Default 300.
    mask_spectra : bool
        Mask elongated diffraction traces before source detection. Default True.
    ra_deg, dec_deg : float, optional
        Approximate field center to narrow the search.
    search_radius_deg : float
        Search radius around (ra_deg, dec_deg). Default 5°.
    scale_arcsec_per_px : float, optional
        Approximate plate scale; ±scale_tolerance band used as a hint.
    scale_tolerance : float
        Fractional tolerance on the scale hint. Default 0.25 (±25%).
    tweak_order : int, optional
        SIP polynomial order for astrometry.net's --tweak-order option.
        If None (default), the server chooses (currently order 2).
        Pass 3, 4, or 5 to request higher-order distortion fits.
    crpix_center : bool
        If True, request that astrometry.net place CRPIX at the image
        center rather than at the reference quad centroid. Default False.
    api_key_file : path-like, optional
        Text file containing the nova.astrometry.net API key.
    timeout : int
        Seconds to wait for a solution. Default 300.
    verbose : bool
        Print progress messages. Default True.
    cache : bool or Path
        If False, no caching. If True, cache next to the FITS file.
        If a Path, cache at that path. Cached results skip re-submission.
    fetch_products : bool
        After solving, fetch corr/axy/rdls/image_rd product tables.
        Default True. Each fetch failure is recorded in result.fetch_status
        with the URL, HTTP status, and first response bytes if applicable.
    save_products_dir : path-like, optional
        If provided, save all fetched raw FITS files here, plus
        solve_metadata.json with job info and fetch outcomes.

    Returns
    -------
    PlatesolveResult or None
        None if the solve failed or timed out.
    """
    filepath = Path(filepath)
    cache_path = _resolve_cache_path(filepath, cache)

    if cache_path is not None and cache_path.exists():
        if verbose:
            print(f"Loading cached result from {cache_path.name}")
        with open(cache_path, "rb") as _fh:
            return pickle.load(_fh)

    api_key = _read_key(Path(api_key_file) if api_key_file else _DEFAULT_KEY_FILE)

    with fits.open(filepath) as hdul:
        original_header = hdul[0].header.copy()
        if use_source_list:
            image = hdul[0].data.astype(np.float32)

    hints: dict = {}
    if ra_deg is not None and dec_deg is not None:
        hints.update(center_ra=ra_deg, center_dec=dec_deg, radius=search_radius_deg)
    if scale_arcsec_per_px is not None:
        hints.update(scale_lower=scale_arcsec_per_px * (1 - scale_tolerance),
                     scale_upper=scale_arcsec_per_px * (1 + scale_tolerance),
                     scale_units="arcsecperpix")
    if tweak_order is not None:
        hints["tweak_order"] = int(tweak_order)
    if crpix_center:
        hints["crpix_center"] = True

    # Login creates a requests.Session so the Django session cookie is preserved
    # for all subsequent calls, including corr_file which requires authentication.
    http_sess, api_session = _login(api_key, verbose)

    detected_x = detected_y = np.array([])
    img_h = img_w = None
    if use_source_list:
        xs, ys, _ = extract_stars(image, max_sources=max_sources,
                                   mask_spectra=mask_spectra)
        if len(xs) == 0:
            if verbose:
                print("No stars detected — cannot submit source list.")
            return None
        detected_x, detected_y = xs, ys
        if verbose:
            print(f"Detected {len(xs)} sources (mask_spectra={mask_spectra}).")
        img_h, img_w = image.shape[-2], image.shape[-1]
        sub_id = _upload_xylist(http_sess, api_session, make_xylist(xs, ys),
                                img_w, img_h, hints, verbose)
    else:
        sub_id = _upload_file(http_sess, api_session, filepath, hints, verbose)

    job_id = _await_job(http_sess, sub_id, timeout, verbose)
    if job_id is None:
        return None
    if not _await_solve(http_sess, job_id, timeout, verbose):
        return None

    wcs_header = _fetch_wcs(http_sess, job_id, verbose)
    if wcs_header is None:
        return None

    if write:
        _write_back(filepath, wcs_header, verbose)

    # ---------- fetch optional products ----------
    product_urls: dict = {}
    fetch_status: dict = {}
    corr_table = axy_table = rdls_table = image_radec_table = None

    save_dir = Path(save_products_dir) if save_products_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if fetch_products:
        if verbose:
            print("Fetching astrometry.net products:")
        for name in _TABLE_PRODUCTS:
            url = _PRODUCT_URLS[name].format(job_id=job_id)
            product_urls[name] = url
            table, raw, status = _fetch_fits_table_product(
                url, name, verbose=verbose, http_sess=http_sess
            )
            fetch_status[name] = status
            if status != "ok" and verbose:
                print(f"  {name:10s}: FAILED — {status}")
            if raw is not None and save_dir is not None:
                (save_dir / _PRODUCT_FILENAMES[name]).write_bytes(raw)
            if name == "corr":
                corr_table = table
            elif name == "axy":
                axy_table = table
            elif name == "rdls":
                rdls_table = table
            elif name == "image_rd":
                image_radec_table = table

    # Populate backward-compatible matched_x/matched_y from corr table
    matched_x = matched_y = np.array([])
    if corr_table is not None:
        fx = _col_array(corr_table, "field_x")
        fy = _col_array(corr_table, "field_y")
        if len(fx) > 0:
            matched_x, matched_y = fx, fy

    if save_dir is not None:
        metadata: dict = {
            "submission_id": sub_id,
            "job_id":        job_id,
            "status":        "success",
            "image_path":    str(filepath),
            "image_shape":   [int(img_h) if img_h else None,
                              int(img_w) if img_w else None],
            "n_submitted":   int(len(detected_x)),
            "n_matched":     int(len(matched_x)),
            "use_source_list": use_source_list,
            "max_sources":   max_sources,
            "mask_spectra":  mask_spectra,
            "tweak_order":   tweak_order,
            "crpix_center":  crpix_center,
            "hints":         hints,
            "fetch_status":  fetch_status,
            "product_urls":  product_urls,
        }
        (save_dir / "solve_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if verbose:
            print(f"Products and metadata saved to {save_dir}/")

    result = PlatesolveResult(
        header=_merge_wcs(original_header, wcs_header),
        detected_x=detected_x,
        detected_y=detected_y,
        matched_x=matched_x,
        matched_y=matched_y,
        submission_id=sub_id,
        job_id=job_id,
        status="success",
        corr_table=corr_table,
        axy_table=axy_table,
        rdls_table=rdls_table,
        image_radec_table=image_radec_table,
        wcs_header_raw=wcs_header,
        product_urls=product_urls,
        fetch_status=fetch_status,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as _fh:
            pickle.dump(result, _fh)
        if verbose:
            print(f"Result cached to {cache_path.name}")

    return result


def platesolve_xylist(
    xs: np.ndarray,
    ys: np.ndarray,
    image_width: int,
    image_height: int,
    original_header: Optional[fits.Header] = None,
    hints: Optional[dict] = None,
    api_key_file: Optional[Union[str, Path]] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    verbose: bool = True,
    cache: Union[bool, Path] = False,
    fetch_products: bool = True,
    save_products_dir: Optional[Union[str, Path]] = None,
) -> Optional["PlatesolveResult"]:
    """Plate-solve a custom source list via nova.astrometry.net.

    Like ``platesolve()`` but accepts explicit pixel coordinates instead of
    extracting sources from a FITS image.  Useful for submitting spatial
    subsets or pre-filtered source lists while keeping the original detector
    coordinate frame.

    Parameters
    ----------
    xs, ys : array-like
        Source x/y positions in full-image pixel coordinates.
    image_width, image_height : int
        Full image dimensions.  Always pass the *original* frame size even
        when xs/ys cover only a sub-region — astrometry.net needs the full
        extent to interpret the pixel coordinates correctly.
    original_header : fits.Header, optional
        FITS header from the original image; WCS keywords from the solution
        are merged into a copy of this header.  If None, a minimal header
        is used.
    hints : dict, optional
        Astrometry.net submission hints.  Useful keys::
            center_ra, center_dec, radius      – sky position hint
            scale_lower, scale_upper, scale_units – plate scale hint
    cache : Path
        If a Path, load from / save to that pickle path.
    fetch_products, save_products_dir : same as ``platesolve()``.
    """
    cache_path: Optional[Path] = None
    if cache and cache is not True:
        cache_path = Path(cache)

    if cache_path is not None and cache_path.exists():
        if verbose:
            print(f"Loading cached result from {cache_path.name}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0:
        if verbose:
            print("No sources provided — cannot submit.")
        return None

    api_key = _read_key(Path(api_key_file) if api_key_file else _DEFAULT_KEY_FILE)
    http_sess, api_session = _login(api_key, verbose)

    if verbose:
        print(f"Submitting {len(xs)} sources  ({image_width}x{image_height} frame).")

    _hints: dict = dict(hints) if hints else {}
    sub_id = _upload_xylist(
        http_sess, api_session,
        make_xylist(xs, ys),
        image_width, image_height,
        _hints, verbose,
    )

    job_id = _await_job(http_sess, sub_id, timeout, verbose)
    if job_id is None:
        return None
    if not _await_solve(http_sess, job_id, timeout, verbose):
        return None

    wcs_header = _fetch_wcs(http_sess, job_id, verbose)
    if wcs_header is None:
        return None

    # Fetch optional products
    product_urls: dict = {}
    fetch_status: dict = {}
    corr_table = axy_table = rdls_table = image_radec_table = None

    save_dir = Path(save_products_dir) if save_products_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if fetch_products:
        if verbose:
            print("Fetching products:")
        for name in _TABLE_PRODUCTS:
            url = _PRODUCT_URLS[name].format(job_id=job_id)
            product_urls[name] = url
            table, raw, status = _fetch_fits_table_product(
                url, name, verbose=verbose, http_sess=http_sess
            )
            fetch_status[name] = status
            if status != "ok" and verbose:
                print(f"  {name:10s}: FAILED — {status}")
            if raw is not None and save_dir is not None:
                (save_dir / _PRODUCT_FILENAMES[name]).write_bytes(raw)
            if name == "corr":
                corr_table = table
            elif name == "axy":
                axy_table = table
            elif name == "rdls":
                rdls_table = table
            elif name == "image_rd":
                image_radec_table = table

    matched_x = matched_y = np.array([])
    if corr_table is not None:
        fx = _col_array(corr_table, "field_x")
        fy = _col_array(corr_table, "field_y")
        if len(fx) > 0:
            matched_x, matched_y = fx, fy

    if save_dir is not None:
        metadata: dict = {
            "submission_id": sub_id,
            "job_id": job_id,
            "status": "success",
            "image_shape": [int(image_height), int(image_width)],
            "n_submitted": int(len(xs)),
            "n_matched": int(len(matched_x)),
            "hints": _hints,
            "fetch_status": fetch_status,
            "product_urls": product_urls,
        }
        (save_dir / "solve_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if verbose:
            print(f"Products saved to {save_dir}/")

    base_header = original_header.copy() if original_header is not None else fits.Header()
    merged_header = _merge_wcs(base_header, wcs_header)

    result = PlatesolveResult(
        header=merged_header,
        detected_x=xs,
        detected_y=ys,
        matched_x=matched_x,
        matched_y=matched_y,
        submission_id=sub_id,
        job_id=job_id,
        status="success",
        corr_table=corr_table,
        axy_table=axy_table,
        rdls_table=rdls_table,
        image_radec_table=image_radec_table,
        wcs_header_raw=wcs_header,
        product_urls=product_urls,
        fetch_status=fetch_status,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(result, fh)
        if verbose:
            print(f"Result cached to {cache_path.name}")

    return result


# ---------------------------------------------------------------------------
# nova.astrometry.net HTTP helpers
# ---------------------------------------------------------------------------

def _login(api_key: str, verbose: bool) -> tuple[requests.Session, str]:
    """Login to nova.astrometry.net.

    Creates a requests.Session so the Django session cookie persists across
    all subsequent HTTP calls.  The corr_file endpoint requires this cookie.

    Returns (http_session, api_session_key).
    """
    http_sess = requests.Session()
    r = http_sess.post(
        f"{_API_URL}/login",
        data={"request-json": json.dumps({"apikey": api_key})},
        timeout=30,
    )
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise RuntimeError(f"Login failed: {resp}")
    if verbose:
        print("Logged in to nova.astrometry.net")
    return http_sess, resp["session"]


def _upload_xylist(
    http_sess: requests.Session,
    api_session: str,
    xylist: io.BytesIO,
    image_width: int,
    image_height: int,
    hints: dict,
    verbose: bool,
) -> int:
    payload = {"session": api_session,
               "image_width": int(image_width), "image_height": int(image_height),
               **hints}
    r = http_sess.post(
        f"{_API_URL}/upload",
        files={"file": ("xylist.fits", xylist, "application/octet-stream")},
        data={"request-json": json.dumps(payload)},
        timeout=60,
    )
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise RuntimeError(f"Xylist upload rejected: {resp}")
    sub_id = resp["subid"]
    if verbose:
        print(f"Source list uploaded (submission {sub_id})")
    return sub_id


def _upload_file(
    http_sess: requests.Session,
    api_session: str,
    filepath: Path,
    hints: dict,
    verbose: bool,
) -> int:
    payload = {"session": api_session, **hints}
    with open(filepath, "rb") as fh:
        r = http_sess.post(
            f"{_API_URL}/upload",
            files={"file": (filepath.name, fh, "application/octet-stream")},
            data={"request-json": json.dumps(payload)},
            timeout=120,
        )
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise RuntimeError(f"File upload rejected: {resp}")
    sub_id = resp["subid"]
    if verbose:
        print(f"Uploaded '{filepath.name}' (submission {sub_id})")
    return sub_id


def _await_job(
    http_sess: requests.Session, sub_id: int, timeout: int, verbose: bool
) -> Optional[int]:
    if verbose:
        print("Waiting for job assignment", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(_POLL_INTERVAL)
        r = http_sess.get(f"{_API_URL}/submissions/{sub_id}", timeout=15)
        r.raise_for_status()
        jobs = [j for j in r.json().get("jobs", []) if j is not None]
        if jobs:
            if verbose:
                print(f" job {jobs[0]}")
            return jobs[0]
        if verbose:
            print(".", end="", flush=True)
    if verbose:
        print(" timed out.")
    return None


def _await_solve(
    http_sess: requests.Session, job_id: int, timeout: int, verbose: bool
) -> bool:
    if verbose:
        print("Solving", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(_POLL_INTERVAL)
        r = http_sess.get(f"{_API_URL}/jobs/{job_id}/info", timeout=15)
        r.raise_for_status()
        status = r.json().get("status", "")
        if verbose:
            print(".", end="", flush=True)
        if status == "success":
            if verbose:
                print(" solved!")
            return True
        if status == "failure":
            if verbose:
                print(" FAILED.")
            return False
    if verbose:
        print(" timed out.")
    return False


def _fetch_wcs(
    http_sess: requests.Session, job_id: int, verbose: bool
) -> Optional[fits.Header]:
    r = http_sess.get(f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=30)
    r.raise_for_status()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            header = fits.Header.fromstring(r.text)
        except Exception as exc:
            if verbose:
                print(f"Could not parse WCS header: {exc}")
            return None
    if verbose:
        print("WCS header fetched.")
    return header


# ---------------------------------------------------------------------------
# Header merge / write-back
# ---------------------------------------------------------------------------

def _wcs_cards(wcs_header: fits.Header) -> list:
    return [card for card in wcs_header.cards
            if (card.keyword or "").upper() not in _SKIP_KEYS
            and not (card.keyword or "").upper().startswith("NAXIS")]


def _merge_wcs(original: fits.Header, wcs_header: fits.Header) -> fits.Header:
    """Return a copy of *original* with WCS keywords from *wcs_header* merged in."""
    merged   = original.copy()
    incoming = {(c.keyword or "").upper() for c in _wcs_cards(wcs_header)}
    for key in list(merged.keys()):
        if key.upper() in incoming:
            del merged[key]
    for card in _wcs_cards(wcs_header):
        merged[card.keyword] = (card.value, card.comment)
    return merged


def _write_back(filepath: Path, wcs_header: fits.Header, verbose: bool) -> None:
    cards    = _wcs_cards(wcs_header)
    incoming = {(c.keyword or "").upper() for c in cards}
    with fits.open(filepath, mode="update") as hdul:
        hdr = hdul[0].header
        for key in list(hdr.keys()):
            if key.upper() in incoming:
                del hdr[key]
        for card in cards:
            hdr[card.keyword] = (card.value, card.comment)
        hdul.flush()
    if verbose:
        print(f"WCS written to '{filepath}'.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_key(path: Path) -> str:
    try:
        key = path.read_text().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file not found: {path}\n"
            "Create this file containing your nova.astrometry.net key."
        )
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key
