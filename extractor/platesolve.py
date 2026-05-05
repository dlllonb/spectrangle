"""
platesolve.py — Submit a FITS image to nova.astrometry.net and return a
PlatesolveResult containing the WCS header and source positions.

The only extractor dependency is stars.py (used when use_source_list=True).

WCS keywords written into PlatesolveResult.header
--------------------------------------------------
All keywords from the astrometry.net solution are merged in. Typical set:

    WCSAXES  CTYPE1  CTYPE2  EQUINOX  LONPOLE  LATPOLE
    CRVAL1   CRVAL2  CRPIX1  CRPIX2   CUNIT1   CUNIT2
    CD1_1    CD1_2   CD2_1   CD2_2

SIP distortion coefficients (A_*, B_*, AP_*, BP_*) are included when
astrometry.net returns them. Pre-existing WCS keywords in the original header
are removed before merging so the result is clean and non-redundant.

Usage
-----
    from extractor.platesolve import platesolve

    result = platesolve("image.fits")          # source-list method, no file write
    result = platesolve("image.fits", write=True)  # also writes WCS to file
    result.header["CRVAL1"]                    # RA of reference pixel
    result.detected_x, result.detected_y      # submitted source positions
    result.matched_x,  result.matched_y       # astrometry-confirmed subset
"""

from __future__ import annotations

import io
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
from astropy.io import fits

from .stars import extract_stars, make_xylist

_API_URL         = "https://nova.astrometry.net/api"
_DEFAULT_TIMEOUT = 300   # seconds
_POLL_INTERVAL   = 5     # seconds between status polls
_DEFAULT_KEY_FILE = Path(__file__).parent.parent / "astrometry_api.txt"
_SKIP_KEYS = frozenset({"SIMPLE", "BITPIX", "EXTEND", "END", "COMMENT", "HISTORY", ""})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PlatesolveResult:
    """Result of a plate-solve operation.

    Attributes
    ----------
    header : fits.Header
        Original FITS header with WCS keywords merged in.
    detected_x, detected_y : ndarray
        Pixel positions (0-based) of all sources submitted to astrometry.net.
        Empty when use_source_list=False.
    matched_x, matched_y : ndarray
        Subset of submitted sources confirmed as catalog matches by
        astrometry.net (from the correspondence file). Empty if unavailable.
    """
    header: fits.Header
    detected_x: np.ndarray = field(default_factory=lambda: np.array([]))
    detected_y: np.ndarray = field(default_factory=lambda: np.array([]))
    matched_x:  np.ndarray = field(default_factory=lambda: np.array([]))
    matched_y:  np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def platesolve(
    filepath: Union[str, Path],
    write: bool = False,
    use_source_list: bool = True,
    max_sources: int = 300,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    search_radius_deg: float = 5.0,
    scale_arcsec_per_px: Optional[float] = None,
    scale_tolerance: float = 0.25,
    api_key_file: Optional[Union[str, Path]] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    verbose: bool = True,
) -> Optional[PlatesolveResult]:
    """Plate-solve a FITS image via nova.astrometry.net.

    Parameters
    ----------
    filepath : path-like
        FITS image to solve.
    write : bool, optional
        Write WCS keywords back into the file on disk. Default False.
    use_source_list : bool, optional
        Extract stars and submit an xylist instead of the full image.
        Default True — faster and more reliable.
    max_sources : int, optional
        Cap on sources submitted in the xylist. Default 300.
    ra_deg, dec_deg : float, optional
        Approximate field centre to narrow the search.
    search_radius_deg : float, optional
        Search radius around (ra_deg, dec_deg). Default 5.
    scale_arcsec_per_px : float, optional
        Approximate plate scale; a ±scale_tolerance band is used as a hint.
    scale_tolerance : float, optional
        Fractional tolerance on the scale hint. Default 0.25 (±25 %).
    api_key_file : path-like, optional
        Text file containing the nova.astrometry.net API key.
        Defaults to astrometry_api.txt in the project root.
    timeout : int, optional
        Seconds to wait for a solution. Default 300.
    verbose : bool, optional
        Print progress messages. Default True.

    Returns
    -------
    PlatesolveResult or None
        None if the solve failed or timed out.
    """
    filepath = Path(filepath)
    api_key  = _read_key(Path(api_key_file) if api_key_file else _DEFAULT_KEY_FILE)

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

    session = _login(api_key, verbose)

    detected_x = detected_y = np.array([])
    if use_source_list:
        xs, ys, _ = extract_stars(image, max_sources=max_sources)
        if len(xs) == 0:
            if verbose:
                print("No stars detected — cannot submit source list.")
            return None
        detected_x, detected_y = xs, ys
        if verbose:
            print(f"Detected {len(xs)} sources for submission.")
        h, w = image.shape[-2], image.shape[-1]
        sub_id = _upload_xylist(session, make_xylist(xs, ys), w, h, hints, verbose)
    else:
        sub_id = _upload_file(session, filepath, hints, verbose)

    job_id = _await_job(sub_id, timeout, verbose)
    if job_id is None:
        return None
    if not _await_solve(job_id, timeout, verbose):
        return None

    wcs_header = _fetch_wcs(job_id, verbose)
    if wcs_header is None:
        return None

    matched_x, matched_y = _fetch_corr(job_id)

    if write:
        _write_back(filepath, wcs_header, verbose)

    return PlatesolveResult(
        header=_merge_wcs(original_header, wcs_header),
        detected_x=detected_x,
        detected_y=detected_y,
        matched_x=matched_x,
        matched_y=matched_y,
    )


# ---------------------------------------------------------------------------
# nova.astrometry.net HTTP helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict, timeout: int = 30) -> dict:
    r = requests.post(f"{_API_URL}/{endpoint}",
                      data={"request-json": json.dumps(payload)}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _login(api_key: str, verbose: bool) -> str:
    resp = _post("login", {"apikey": api_key})
    if resp.get("status") != "success":
        raise RuntimeError(f"Login failed: {resp}")
    if verbose:
        print("Logged in to nova.astrometry.net")
    return resp["session"]


def _upload_xylist(session: str, xylist: io.BytesIO,
                   image_width: int, image_height: int,
                   hints: dict, verbose: bool) -> int:
    payload = {"session": session,
               "image_width": int(image_width), "image_height": int(image_height),
               **hints}
    r = requests.post(f"{_API_URL}/upload",
                      files={"file": ("xylist.fits", xylist, "application/octet-stream")},
                      data={"request-json": json.dumps(payload)}, timeout=60)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise RuntimeError(f"Xylist upload rejected: {resp}")
    sub_id = resp["subid"]
    if verbose:
        print(f"Source list uploaded (submission {sub_id})")
    return sub_id


def _upload_file(session: str, filepath: Path, hints: dict, verbose: bool) -> int:
    payload = {"session": session, **hints}
    with open(filepath, "rb") as fh:
        r = requests.post(f"{_API_URL}/upload",
                          files={"file": (filepath.name, fh, "application/octet-stream")},
                          data={"request-json": json.dumps(payload)}, timeout=120)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise RuntimeError(f"File upload rejected: {resp}")
    sub_id = resp["subid"]
    if verbose:
        print(f"Uploaded '{filepath.name}' (submission {sub_id})")
    return sub_id


def _await_job(sub_id: int, timeout: int, verbose: bool) -> Optional[int]:
    if verbose:
        print("Waiting for job assignment", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(_POLL_INTERVAL)
        r = requests.get(f"{_API_URL}/submissions/{sub_id}", timeout=15)
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


def _await_solve(job_id: int, timeout: int, verbose: bool) -> bool:
    if verbose:
        print("Solving", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(_POLL_INTERVAL)
        r = requests.get(f"{_API_URL}/jobs/{job_id}/info", timeout=15)
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


def _fetch_wcs(job_id: int, verbose: bool) -> Optional[fits.Header]:
    r = requests.get(f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=30)
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


def _fetch_corr(job_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Fetch pixel positions of astrometry-confirmed sources from the corr file.

    The correspondence file is a FITS binary table with field_x / field_y
    columns giving the pixel positions (in the submitted coordinate frame) of
    sources that were successfully matched to catalog stars.
    Returns empty arrays if the fetch fails or the columns are absent.
    """
    try:
        r = requests.get(f"https://nova.astrometry.net/corr_file/{job_id}", timeout=30)
        r.raise_for_status()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(io.BytesIO(r.content)) as hdul:
                tbl = hdul[1].data
                x = np.asarray(tbl["field_x"], dtype=float)
                y = np.asarray(tbl["field_y"], dtype=float)
        return x, y
    except Exception:
        return np.array([]), np.array([])


# ---------------------------------------------------------------------------
# Header merge / write-back
# ---------------------------------------------------------------------------

def _wcs_cards(wcs_header: fits.Header) -> list[fits.Card]:
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
