#!/usr/bin/env python3
"""scripts/gen_nb14.py — generate notebooks/14_angle_extraction_summary.ipynb"""
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / '14_angle_extraction_summary.ipynb'

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "id": f"m{len(cells):03d}",
                  "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}",
                  "metadata": {}, "outputs": [], "execution_count": None,
                  "source": src})


# =============================================================================
# CELL 1 — TITLE
# =============================================================================
md("""\
# SpectrAngle: Diffraction Grid Angle from Plate-Solved Images
## Progress Summary

This notebook presents the current proof-of-concept results for the \
SpectrAngle angle-extraction pipeline.

**High-level flow:**

```
Real image  →  star detection  →  local plate-solve (astrometry.net)
                                        ↓
                                    WCS + θ_north
                                        ↓
           diffraction trace detection  →  θ_spectra (pixel)
                                        ↓
                              θ_grid (sky frame)  =  pixel_angle_to_sky_angle(WCS, θ_spectra)
```

The **WCS** gives the local sky basis at the image center — which direction \
is celestial north in pixel coordinates.  The **diffraction angle** gives the \
grating orientation in pixel coordinates.  Together they yield the grating \
orientation in a sky/celestial reference frame.

**Test image:** `fuji6_asi178_100_15s.fit` — real observation, 15 s, \
~69° × 46° field, ASI178MC + wide-field lens + grating.  \
This is a non-optimized proof-of-concept image, not a final calibration setup.

**Status:** Phase 1 (star detection + plate-solve + WCS orientation) complete \
and validated.  Diffraction trace extraction is a functional prototype; \
a robust fitting pipeline is in development.
""")

# =============================================================================
# CELL 2 — ENVIRONMENT CHECK
# =============================================================================
md("## Cell 2 — Environment\n")

code("""\
import sys, shutil, warnings, subprocess, pickle, csv, json as _json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, median_filter
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u

_pkg_root = str(Path('..').resolve())
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from extractor import extract_stars
from extractor.platesolve import solve_plate, wcs_summary, LocalSolveFieldError, _col_array, get_col
from extractor.wcsangle import (
    angle_diff_deg, center_wcs_angle_metrics,
    local_north_angle_deg, pixel_angle_to_sky_angle,
)

# ── Paths ────────────────────────────────────────────────────────────────────
FITS_PATH   = sorted((Path('..') / 'data').glob('*.fit'))[0]
OUT_DIR     = Path('..') / 'outputs' / 'angle_extraction_summary'
FID_CACHE   = Path('..') / 'out' / 'local_bootstrap_10' / 'fiducial_result.pkl'
SRC_CACHE   = Path('..') / 'out' / 'local_bootstrap_10' / 'sources.pkl'
BOOT10_CSV  = Path('..') / 'out' / 'local_bootstrap_10' / 'bootstrap_results.csv'
BOOT11_CSV  = Path('..') / 'out' / 'local_bootstrap_11' / 'bootstrap_results.csv'
BOOT10_JSON = Path('..') / 'out' / 'local_bootstrap_10' / 'bootstrap_summary.json'
BACKEND_CFG = '/mnt/c/Users/bassd/.astrometry/backend.cfg'
INDEX_DIR   = Path('/mnt/c/Users/bassd/astrometry-data/4100/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _make_wcs(header):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        warnings.filterwarnings('ignore', message='.*datfix.*')
        return WCS(header)

# ── Checks ───────────────────────────────────────────────────────────────────
print(f'Python      : {sys.executable}  ({sys.version.split()[0]})')
sf = shutil.which('solve-field')
print(f'solve-field : {sf or "NOT FOUND — switch to spectrangle WSL kernel"}')
print(f'backend cfg : {BACKEND_CFG}  ✓' if Path(BACKEND_CFG).exists() else
      f'backend cfg : {BACKEND_CFG}  MISSING')
print(f'index dir   : {INDEX_DIR}  ✓' if INDEX_DIR.exists() else
      f'index dir   : {INDEX_DIR}  MISSING')
print(f'test image  : {FITS_PATH.name}  ✓' if FITS_PATH.exists() else
      f'test image  : MISSING')
print(f'fiducial WCS: {"✓ cached" if FID_CACHE.exists() else "not cached — will solve"}')
print(f'bootstrap 10: {"✓" if BOOT10_CSV.exists() else "missing"}  '
      f'  bootstrap 11: {"✓" if BOOT11_CSV.exists() else "missing"}')
try:
    _r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                        capture_output=True, text=True, cwd=str(ROOT), timeout=5)
    print(f'git commit  : {_r.stdout.strip()}')
except Exception:
    pass

import extractor
print(f'extractor   : {extractor.__version__}')

plt.rcParams.update({
    'figure.facecolor': '#0d0d0d', 'axes.facecolor': '#0d0d0d',
    'text.color': '#e0e0e0',       'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#aaa',         'ytick.color': '#aaa',
    'axes.edgecolor': '#444',      'axes.titlecolor': '#e0e0e0',
    'legend.facecolor': '#1a1a1a', 'legend.edgecolor': '#555',
    'grid.color': '#2a2a2a',       'grid.alpha': 0.6,
    'font.size': 11,               'axes.titlesize': 12,
    'savefig.facecolor': '#0d0d0d','savefig.dpi': 150,
    'savefig.bbox': 'tight',
})
print('\\nEnvironment OK.')
""")


# =============================================================================
# CELL 3 — LOAD IMAGE AND DISPLAY
# =============================================================================
md("""\
## Cell 3 — Test image

The test image is a real wide-field observation taken through a diffraction \
grating.  Stars appear as compact point sources; the grating produces an \
elongated spectrum on either side of each bright star along the grating axis.
""")

code("""\
with fits.open(FITS_PATH) as hdul:
    image       = hdul[0].data.astype(np.float32)
    orig_header = hdul[0].header.copy()

ny, nx = image.shape
xc, yc = (nx - 1) / 2., (ny - 1) / 2.
print(f'Image  : {FITS_PATH.name}  ({nx} × {ny} px)')
print(f'Median : {np.median(image):.1f}   99.5th pct : {np.percentile(image, 99.5):.1f}')

# ── Three stretch candidates; we'll pick the best ─────────────────────────────
_bg = gaussian_filter(image.astype(float), sigma=50)
_sub = np.clip(image - _bg, 0, None)   # background-subtracted

def _arcsinh_stretch(arr, lo_pct=0.5, hi_pct=99.7):
    lo, hi = np.percentile(arr[arr > 0], [lo_pct, hi_pct])
    clipped = np.clip(arr, lo, hi)
    return np.arcsinh(clipped), float(np.arcsinh(lo)), float(np.arcsinh(hi))

def _log_stretch(arr, lo_pct=1.0, hi_pct=99.5):
    lo, hi = np.percentile(arr[arr > 0], [lo_pct, hi_pct])
    clipped = np.clip(arr, max(lo, 1e-3), hi)
    return np.log10(clipped), float(np.log10(max(lo, 1e-3))), float(np.log10(hi))

def _pct_stretch(arr, lo_pct=1.0, hi_pct=99.5):
    lo, hi = np.percentile(arr[arr > 0], [lo_pct, hi_pct])
    return np.clip(arr, lo, hi), lo, hi

disp_arcsinh, _alo, _ahi = _arcsinh_stretch(_sub)
disp_log,     _llo, _lhi = _log_stretch(_sub)
disp_pct,     _plo, _phi = _pct_stretch(_sub, 0.5, 99.5)

# Comparison (run once; choose the best)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, d, lo, hi, title in [
    (axes[0], disp_pct,     _plo, _phi, 'Percentile (0.5–99.5)'),
    (axes[1], disp_arcsinh, _alo, _ahi, 'arcsinh (BG-sub, 0.5–99.7)'),
    (axes[2], disp_log,     _llo, _lhi, 'log10 (BG-sub, 1–99.5)'),
]:
    ax.imshow(d, origin='lower', cmap='gray_r', vmin=lo, vmax=hi,
              aspect='equal', interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
fig.suptitle('Stretch comparison — pick the cleanest', fontsize=11, y=1.01)
fig.tight_layout()
plt.show()

# ── Final display: arcsinh on background-subtracted (most balanced) ───────────
fig, ax = plt.subplots(figsize=(14, 9.5))
ax.imshow(disp_arcsinh, origin='lower', cmap='gray_r', vmin=_alo, vmax=_ahi,
          aspect='equal', interpolation='nearest')
ax.set_xlabel('x  (pixels)')
ax.set_ylabel('y  (pixels)')
ax.set_title(f'{FITS_PATH.name}   —   {nx}×{ny} px   |   15 s   |   wide-field + grating',
             fontsize=12)
_ann = ('Real test image\nASI178MC + wide-field lens + grating\n'
        f'~69° × 46° FOV  |  ~80 arcsec/px')
ax.text(0.01, 0.98, _ann, transform=ax.transAxes,
        fontsize=9, color='#aaa', va='top',
        bbox=dict(facecolor='#111', alpha=0.7, edgecolor='none',
                  boxstyle='round,pad=0.35'))
fig.tight_layout()
fig.savefig(OUT_DIR / '01_image.png')
plt.show()

# Store for later cells
_disp    = disp_arcsinh
_disp_lo = _alo
_disp_hi = _ahi
""")


# =============================================================================
# CELL 4 — SOURCE DETECTION
# =============================================================================
md("""\
## Cell 4 — Source detection

The extractor uses DAOStarFinder on a background-subtracted, \
spectra-masked image.  Without masking, elongated grating traces \
contaminate the detection.
""")

code("""\
# Load from nb10 cache if available, otherwise re-detect
if SRC_CACHE.exists():
    with open(SRC_CACHE, 'rb') as _f:
        all_xs, all_ys, all_fluxes = pickle.load(_f)
    print(f'Loaded {len(all_xs)} sources from cache.')
else:
    all_xs, all_ys, all_fluxes = extract_stars(image, max_sources=300, mask_spectra=True)
    print(f'Detected {len(all_xs)} sources.')

print(f'Flux range : {all_fluxes.min():.0f} – {all_fluxes.max():.0f}  '
      f'(median {np.median(all_fluxes):.0f})')

# Also run without masking to show the difference
from extractor.stars import _diffraction_mask
_diff_mask = _diffraction_mask(_sub)
print(f'\\nDiffraction mask: {_diff_mask.sum():,} pixels flagged as trace features  '
      f'({100*_diff_mask.mean():.1f}% of image)')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, _xs, _ys, _label, _col in [
    (axes[0], all_xs, all_ys, f'mask_spectra=True  →  {len(all_xs)} stars', 'lime'),
]:
    ax.imshow(_disp, origin='lower', cmap='gray_r', vmin=_disp_lo, vmax=_disp_hi,
              aspect='equal', interpolation='nearest')
    # Diffraction mask overlay (semi-transparent)
    _mask_rgba = np.zeros((*_diff_mask.shape, 4), dtype=float)
    _mask_rgba[_diff_mask, 0] = 1.0  # red channel
    _mask_rgba[_diff_mask, 3] = 0.35
    ax.imshow(_mask_rgba, origin='lower', aspect='equal', interpolation='nearest')
    ax.scatter(_xs, _ys, s=25, facecolors='none', edgecolors=_col,
               linewidths=1.0, alpha=0.85)
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_title(f'Detected stars (green circles)\nDiffraction traces (red mask)\n{_label}',
                 fontsize=10)
    ax.set_xlabel('x  (px)'); ax.set_ylabel('y  (px)')

# Flux histogram
ax2 = axes[1]
ax2.hist(np.log10(all_fluxes + 1), bins=25, color='#4fc3f7', edgecolor='#111', alpha=0.9)
ax2.set_xlabel('log₁₀ (flux + 1)')
ax2.set_ylabel('Count')
ax2.set_title(f'Source flux distribution  (N = {len(all_xs)})', fontsize=10)
ax2.grid(alpha=0.3)

fig.suptitle('Extractor output: stars separated from grating traces', fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / '02_source_detection.png')
plt.show()

print(f'\\nStar detection   : {len(all_xs)} sources (with grating-trace masking)')
print(f'Trace pixels     : {_diff_mask.sum():,}  ({100*_diff_mask.mean():.1f}% of image)')
""")


# =============================================================================
# CELL 5 — FIDUCIAL PLATE SOLVE
# =============================================================================
md("""\
## Cell 5 — Local plate-solve  *(may use cached result)*

Order-5 SIP solve via local `solve-field`.  Scale bounds 70–95 arcsec/px. \
The fiducial result is loaded from cache if available.
""")

code("""\
SCALE_LOW, SCALE_HIGH = 70.0, 95.0
TWEAK_ORDER   = 5
SOLVE_TIMEOUT = 120

# ── Try cache (nb10 fiducial) first ──────────────────────────────────────────
_fid_alt = Path('..') / 'out' / 'local_bootstrap_10' / 'fiducial'
if FID_CACHE.exists():
    print('Loading fiducial result from cache (notebook 10)…')
    with open(FID_CACHE, 'rb') as _f:
        fid_result = pickle.load(_f)
else:
    print(f'Running fiducial order-{TWEAK_ORDER} solve ({len(all_xs)} sources)…')
    import time as _t
    _t0 = _t.time()
    fid_result = solve_plate(
        xs=all_xs, ys=all_ys,
        image_width=nx, image_height=ny,
        original_header=orig_header,
        backend='local', backend_config=BACKEND_CFG,
        scale_units='arcsecperpix',
        scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
        tweak_order=TWEAK_ORDER,
        output_dir=Path('..') / 'out' / 'local_bootstrap_10' / 'fiducial',
        timeout=SOLVE_TIMEOUT, verbose=True,
    )
    if fid_result is None:
        raise RuntimeError('Fiducial solve failed — check solve-field output.')
    print(f'Solve finished in {_t.time()-_t0:.1f} s')

fid_wcs     = _make_wcs(fid_result.header)
fid_m       = center_wcs_angle_metrics(fid_wcs, image.shape, compute_east=True)
_scales_deg = proj_plane_pixel_scales(fid_wcs)
_ps         = float(np.mean(np.abs(_scales_deg)) * 3600)
_fovx       = float(np.abs(_scales_deg[0]) * nx)
_fovy       = float(np.abs(_scales_deg[1]) * ny)
_ne_dep     = abs(angle_diff_deg(fid_m.east_angle_deg, fid_m.north_angle_deg) - 90.0)

# Read index ID if available
def _match_meta(d):
    p = Path(d) / 'xylist.match'
    if not p.exists():
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from astropy.table import Table
            with fits.open(str(p)) as h:
                t = Table(h[1].data)
                return dict(indexid=int(t['INDEXID'][0]),
                            logodds=float(t['LOGODDS'][0]))
    except Exception:
        return {}

_mm = _match_meta(_fid_alt)

print()
print('━' * 55)
print('  Fiducial WCS  (order-5 SIP, local solve-field)')
print('━' * 55)
print(f'  Sources submitted  : {len(fid_result.detected_x)}')
print(f'  Sources matched    : {len(fid_result.matched_x)}')
if _mm:
    print(f'  Index INDEXID      : {_mm.get("indexid","?")}   '
          f'(logodds {_mm.get("logodds",0):.1f})')
print(f'  Plate scale        : {_ps:.2f} arcsec/px')
print(f'  FOV                : {_fovx:.1f}° × {_fovy:.1f}°')
print(f'  Center RA / Dec    : {fid_m.ra_deg:.5f}°  /  {fid_m.dec_deg:.5f}°')
print(f'  θ_north            : {fid_m.north_angle_deg:.4f}°  (CCW from +x pixel axis)')
print(f'  θ_east             : {fid_m.east_angle_deg:.4f}°')
print(f'  N/E non-orthogon.  : {_ne_dep:.4f}°  (ideally 0)')
print('━' * 55)
print()
print(wcs_summary(fid_result.header))
""")


# =============================================================================
# CELL 6 — WCS OVERLAY
# =============================================================================
md("""\
## Cell 6 — WCS overlay

The plate-solve maps every pixel to a sky position.  The WCS grid confirms the \
solution covers the full field.  Matched catalog stars are shown in orange.
""")

code("""\
from astropy.visualization.wcsaxes import WCSAxes

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', message='.*datfix.*')
    warnings.filterwarnings('ignore', message='.*cdfix.*')
    warnings.filterwarnings('ignore', message='.*No WCS.*')

    fig = plt.figure(figsize=(14, 9.5))
    ax  = fig.add_subplot(111, projection=fid_wcs)

    ax.imshow(_disp, origin='lower', cmap='gray_r',
              vmin=_disp_lo, vmax=_disp_hi,
              aspect='equal', interpolation='nearest')

    # WCS coordinate grid (coarser spacing for this wide field)
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='#1a6b8a', linestyle='--', linewidth=0.7, alpha=0.55)
    overlay['ra'].set_major_formatter('hh:mm')
    overlay['dec'].set_major_formatter('dd')
    overlay['ra'].set_ticks(spacing=5 * u.deg)
    overlay['dec'].set_ticks(spacing=5 * u.deg)
    overlay['ra'].set_ticklabel(color='#7ec8e3', size=9)
    overlay['dec'].set_ticklabel(color='#7ec8e3', size=9)
    ax.set_xlabel('Right Ascension', color='#7ec8e3')
    ax.set_ylabel('Declination', color='#7ec8e3')

    # Matched catalog sources
    if fid_result.corr_table is not None:
        _mfx = _col_array(fid_result.corr_table, 'field_x')
        _mfy = _col_array(fid_result.corr_table, 'field_y')
    else:
        _mfx, _mfy = fid_result.matched_x, fid_result.matched_y

    if len(_mfx) > 0:
        ax.scatter(_mfx, _mfy, transform=ax.get_transform('pixel'),
                   s=70, facecolors='none', edgecolors='#f5a623',
                   linewidths=1.4, alpha=0.9,
                   label=f'catalog-matched ({len(_mfx)})')

    ax.scatter(all_xs, all_ys, transform=ax.get_transform('pixel'),
               s=16, facecolors='none', edgecolors='lime',
               linewidths=0.7, alpha=0.5,
               label=f'detected ({len(all_xs)})')

    ax.legend(fontsize=9, loc='upper right',
              framealpha=0.7, edgecolor='#555')

    _ann = (
        f'Order-5 SIP  |  {_ps:.1f} "/px\n'
        f'FOV {_fovx:.1f}° × {_fovy:.1f}°  |  idx 4117\n'
        f'θ_north = {fid_m.north_angle_deg:.4f}°\n'
        f'Matched {len(_mfx)} / {len(all_xs)} sources'
    )
    ax.text(0.01, 0.01, _ann, transform=ax.transAxes,
            fontsize=9, color='white', va='bottom',
            bbox=dict(facecolor='#0d0d0d', alpha=0.8, edgecolor='#555',
                      boxstyle='round,pad=0.4'))
    ax.set_title('WCS overlay — local order-5 solve-field solution', fontsize=12)

fig.tight_layout()
fig.savefig(OUT_DIR / '03_wcs_overlay.png')
plt.show()
print('WCS overlay saved.')
""")


# =============================================================================
# CELL 7 — SPECTRA / DIFFRACTION ANGLE EXTRACTION
# =============================================================================
md("""\
## Cell 7 — Diffraction trace angle  *(prototype)*

**Approach:** The same star-suppression / connected-component pipeline that \
masks grating traces for star detection is here repurposed to *measure* their \
orientation.  For each sufficiently elongated blob the major-axis angle is \
extracted using `skimage.measure.regionprops`.  Axial circular statistics \
(the doubled-angle trick) combine individual measurements into a single robust \
pixel-space angle **θ_spectra**.

> ⚠️  This is a functional prototype — not a final fitting pipeline. \
A proper implementation using Radon/Hough transforms or ridge filters is in \
development.  The current estimate gives a reasonable working value for \
proof-of-concept validation.
""")

code("""\
from skimage.measure import label, regionprops

# ── Trace detection (reimplemented from _diffraction_mask internals) ──────────
def _extract_trace_angles(img, threshold_pct=70.0, min_pixels=30,
                           min_eccentricity=0.88, min_major_px=20.0):
    '''
    Return list of dicts with centroid, pixel angle, and shape metrics
    for each detected elongated trace blob.

    Pixel angle convention: CCW from +x pixel axis, range (-90, 90] deg.
    skimage regionprops.orientation = angle from row axis (0th axis) CCW.
    Conversion: theta_pix = ((90 - deg(orientation)) + 90) % 180 - 90
    '''
    factor = max(1, max(img.shape) // 1200)
    small  = img[::factor, ::factor].astype(np.float32)

    suppressed  = median_filter(small, size=7)
    feature_img = np.clip(suppressed - gaussian_filter(suppressed, sigma=30.0), 0, None)

    nz = feature_img[feature_img > 0]
    if nz.size == 0:
        return []

    thresh    = np.percentile(nz, threshold_pct)
    label_img = label(feature_img >= thresh)

    traces = []
    for reg in regionprops(label_img, intensity_image=feature_img):
        if reg.num_pixels < min_pixels:
            continue
        major = float(getattr(reg, 'axis_major_length', None) or reg.major_axis_length)
        minor = float(getattr(reg, 'axis_minor_length', None) or reg.minor_axis_length)
        if major < min_major_px or reg.eccentricity < min_eccentricity:
            continue

        # Convert skimage orientation (from row-axis, CCW) → CCW from +x pixel axis
        # ori=0 → vertical blob → theta_pix=±90°
        # ori=±pi/2 → horizontal blob → theta_pix=0°
        theta_pix = ((90.0 - np.degrees(reg.orientation)) + 90.0) % 180.0 - 90.0

        traces.append({
            'cx':        reg.centroid[1] * factor,
            'cy':        reg.centroid[0] * factor,
            'theta_pix': theta_pix,
            'major_px':  major * factor,
            'ecc':       reg.eccentricity,
            'weight':    major * reg.eccentricity,   # longer + more elongated = higher weight
        })
    return traces

traces = _extract_trace_angles(_sub)
print(f'Detected {len(traces)} elongated trace features.')

if len(traces) == 0:
    print('No traces detected — check threshold or eccentricity parameters.')
    _theta_spectra_pix = float('nan')
else:
    # ── Axial circular statistics (doubled-angle trick) ─────────────────────
    _theta_arr = np.array([t['theta_pix'] for t in traces])
    _weights   = np.array([t['weight']    for t in traces])
    _weights   /= _weights.sum()

    _phi = np.deg2rad(2.0 * _theta_arr)           # double the angles
    _S   = np.sum(_weights * np.sin(_phi))
    _C   = np.sum(_weights * np.cos(_phi))
    _theta_spectra_pix = float(np.degrees(np.arctan2(_S, _C)) / 2.0)  # halve back
    _R   = float(np.hypot(_S, _C))   # resultant length ∈ [0,1]; 1=perfect agreement

    # Robust scatter (MAD in doubled-angle space, converted back)
    _phi_mean = np.deg2rad(2.0 * _theta_spectra_pix)
    _phi_diffs = np.angle(np.exp(1j * (_phi - _phi_mean)))
    _theta_mad = float(np.degrees(np.median(np.abs(_phi_diffs)) / 2.0))

    print(f'\\n  θ_spectra (pixel) : {_theta_spectra_pix:.3f}°  (CCW from +x pixel axis)')
    print(f'  Resultant R       : {_R:.3f}  (1=all traces parallel)')
    print(f'  MAD spread        : ±{_theta_mad:.2f}°')
    print(f'  N traces used     : {len(traces)}')
    print(f'  Angle range       : [{_theta_arr.min():.1f}°, {_theta_arr.max():.1f}°]')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: image with trace overlays
    ax = axes[0]
    ax.imshow(_disp, origin='lower', cmap='gray_r', vmin=_disp_lo, vmax=_disp_hi,
              aspect='equal', interpolation='nearest')

    # Color-code traces by angle relative to mean
    _cmap = plt.cm.RdYlGn
    _norm = plt.Normalize(vmin=-15, vmax=15)
    for t in traces:
        _da   = angle_diff_deg(t['theta_pix'], _theta_spectra_pix)
        _c    = _cmap(_norm(_da))
        _half = t['major_px'] / 2.0
        _rad  = np.deg2rad(t['theta_pix'])
        _dx   = _half * np.cos(_rad)
        _dy   = _half * np.sin(_rad)
        ax.plot([t['cx'] - _dx, t['cx'] + _dx],
                [t['cy'] - _dy, t['cy'] + _dy],
                color=_c, lw=1.5, alpha=0.8)

    # Mean angle line through image center
    _L   = min(nx, ny) * 0.35
    _rad = np.deg2rad(_theta_spectra_pix)
    ax.annotate('', xy=(xc + _L*np.cos(_rad), yc + _L*np.sin(_rad)),
                xytext=(xc - _L*np.cos(_rad), yc - _L*np.sin(_rad)),
                arrowprops=dict(arrowstyle='->', color='#ffe082', lw=2.5))
    ax.scatter(all_xs, all_ys, s=10, facecolors='none', edgecolors='cyan',
               linewidths=0.6, alpha=0.4, label='stars')
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_title(f'Trace detections (N={len(traces)})\\n'
                 f'θ_spectra = {_theta_spectra_pix:.2f}°  |  R = {_R:.2f}',
                 fontsize=10)
    ax.set_xlabel('x  (px)'); ax.set_ylabel('y  (px)')
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Δθ from mean (°)', fraction=0.035, pad=0.02)

    # Right: angle distribution
    ax2 = axes[1]
    ax2.hist(_theta_arr, bins=min(20, len(traces)),
             color='#4fc3f7', edgecolor='#0d0d0d', alpha=0.9,
             weights=_weights * len(traces))
    ax2.axvline(_theta_spectra_pix, color='#ffe082', lw=2.5,
                label=f'mean  {_theta_spectra_pix:.2f}°')
    ax2.axvspan(_theta_spectra_pix - _theta_mad, _theta_spectra_pix + _theta_mad,
                alpha=0.2, color='#ffe082', label=f'±MAD {_theta_mad:.2f}°')
    ax2.set_xlabel('Pixel trace angle  θ_spectra  (°  CCW from +x)')
    ax2.set_ylabel('Weighted count')
    ax2.set_title('Trace angle distribution', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle('Diffraction trace angle extraction  (prototype)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '04_trace_angle.png')
    plt.show()
""")


# =============================================================================
# CELL 8 — SKY-FRAME ANGLE
# =============================================================================
md("""\
## Cell 8 — Sky-referenced grid angle

The WCS gives the local sky basis at the image center: $\\theta_\\mathrm{north}$ \
is the angle of celestial north in pixel coordinates.  Combining this with the \
measured pixel-space trace angle $\\theta_\\mathrm{spectra}$ gives the grating \
orientation in the sky tangent-plane.

Two methods are compared:

| Method | Formula | Notes |
|--------|---------|-------|
| Simple subtraction | $\\theta_\\mathrm{sky} \\approx \\theta_\\mathrm{spectra} - \\theta_\\mathrm{north}$ | Exact only when north/east are orthogonal in pixel space |
| Full Jacobian | `pixel_angle_to_sky_angle(wcs, xc, yc, θ_spectra)` | Handles shear/non-orthogonality; always use this |

The N/E non-orthogonality is **~0.04°** for this image, so the two methods \
agree at that level.
""")

code("""\
if np.isnan(_theta_spectra_pix):
    print('Trace angle not available — Cell 7 must succeed first.')
else:
    # Method 1: simple subtraction (approximate)
    _sky_simple = angle_diff_deg(_theta_spectra_pix - fid_m.north_angle_deg, 0.0)
    _sky_simple = (_sky_simple + 180) % 360 - 180  # wrap to (-180,180]

    # Method 2: full local WCS Jacobian (pixel_angle_to_sky_angle)
    _sky_full = pixel_angle_to_sky_angle(fid_wcs, xc, yc, _theta_spectra_pix)

    _diff_methods = angle_diff_deg(_sky_full, _sky_simple)

    print('━' * 60)
    print('  SKY-REFERENCED GRATING ORIENTATION')
    print('━' * 60)
    print(f'  θ_spectra (pixel)     : {_theta_spectra_pix:+.4f}°  (CCW from +x pixel axis)')
    print(f'  θ_north   (WCS)       : {fid_m.north_angle_deg:+.4f}°')
    print(f'  N/E non-orthogonality : {_ne_dep:.4f}°')
    print()
    print(f'  θ_grid, simple method : {_sky_simple:+.4f}°  east of north (approx)')
    print(f'  θ_grid, full Jacobian : {_sky_full:+.4f}°  east of north')
    print(f'  |Δ| between methods   : {abs(_diff_methods):.5f}°')
    print()
    print(f'  ▶  Best estimate  θ_grid = {_sky_full:+.4f}°  east of north')
    print()
    print('  Angle convention: positive = east of north (standard astronomical PA)')
    print('  θ_grid east-of-north is the grating axis orientation in the sky frame.')
    print('━' * 60)

    # Summary diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)

    def _arrow(ax, angle_deg, color, label, r=1.1, lw=2.5):
        _r   = np.deg2rad(angle_deg)
        ax.annotate('', xy=(r*np.cos(_r), r*np.sin(_r)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        ax.text(1.22*np.cos(_r), 1.22*np.sin(_r), label,
                ha='center', va='center', color=color, fontsize=10)

    # θ_north direction in pixel space (CCW from +x)
    _arrow(ax, fid_m.north_angle_deg, '#7ec8e3', 'N', r=1.0)
    _arrow(ax, fid_m.east_angle_deg,  '#5eb8a0', 'E', r=1.0)
    # θ_spectra direction
    _arrow(ax, _theta_spectra_pix, '#ffe082', 'grating\n(pixel)', r=0.85)
    # Reference +x
    ax.annotate('', xy=(1.0, 0), xytext=(-1.0, 0),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
    ax.text(1.1, 0, '+x', ha='left', va='center', color='#555', fontsize=9)

    ax.set_title(f'Sky basis at image center\n'
                 f'θ_north = {fid_m.north_angle_deg:.2f}°   '
                 f'θ_grid = {_sky_full:.2f}° E of N',
                 fontsize=10)
    ax.set_xlabel('x  (pixel axis)'); ax.set_ylabel('y  (pixel axis)')
    ax.grid(alpha=0.25)
    ax.axhline(0, color='#333', lw=0.8); ax.axvline(0, color='#333', lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '05_sky_angle_diagram.png')
    plt.show()
""")


# =============================================================================
# CELL 9 — WCS UNCERTAINTY SUMMARY
# =============================================================================
md("""\
## Cell 9 — WCS orientation uncertainty  (X in error budget)

The WCS orientation term **X = σ(θ_north)** sets the floor on how well we \
know the sky-frame grating angle.  Estimated by bootstrap resampling: \
re-solving with random sub-lists of sources gives a spread of θ_north values.

Results from prior notebooks (loaded from saved CSVs):
""")

code("""\
# ── Load bootstrap results ────────────────────────────────────────────────────
def _load_boot_csv(path):
    rows = []
    if not Path(path).exists():
        return np.array([])
    with open(path, newline='', encoding='utf-8') as _f:
        for _row in csv.DictReader(_f):
            if _row.get('status') == 'ok' and _row.get('north_angle_deg'):
                rows.append(float(_row['north_angle_deg']))
    return np.array(rows)

boot10 = _load_boot_csv(BOOT10_CSV)
boot11 = _load_boot_csv(BOOT11_CSV)

# Load summary JSON for quick stats
_summ = {}
if BOOT10_JSON.exists():
    with open(BOOT10_JSON) as _f:
        _summ = _json.load(_f)

_fid_theta = fid_m.north_angle_deg

def _boot_stats(arr, label, pool_desc, n_total):
    if len(arr) < 2:
        print(f'  {label}: no data')
        return None
    dth = angle_diff_deg(arr, _fid_theta)
    std = float(dth.std())
    mad = float(np.median(np.abs(dth - np.median(dth))))
    return {'label': label, 'pool': pool_desc, 'n_ok': len(arr), 'n_total': n_total,
            'std': std, 'mad': mad, 'mean_offset': float(dth.mean()), 'dth': dth}

s10 = _boot_stats(boot10, 'nb10 — full detected list',
                   '255 detected sources, 80% random', 50)
s11 = _boot_stats(boot11, 'nb11 — matched sources only',
                   '86 matched sources, 80% random', 250)

print('WCS Orientation Uncertainty  (σ of θ_north under bootstrap)')
print()
print(f'  Fiducial θ_north = {_fid_theta:.5f}°')
print()
print(f'  {"Method":<40}  {"N solves":>8}  {"σ (°)":>8}  {"MAD (°)":>8}  {"mean Δ (°)":>10}')
print('  ' + '─' * 80)
for s in [s10, s11]:
    if s:
        print(f'  {s["label"]:<40}  {s["n_ok"]:>8}  {s["std"]:>8.4f}  '
              f'{s["mad"]:>8.4f}  {s["mean_offset"]:>+10.4f}')

print()
print('  Interpretation:')
print('  • Bootstrap σ ≈ 0.09–0.13° — WCS orientation is stable at the ~0.1° level.')
print('  • MAD ≈ 0.05–0.07° — the distribution has heavy tails; a few solutions')
print('    deviate more than 1σ (typical for resampled high-order SIP fits).')
print('  • Source selection and spatial coverage likely dominate the scatter,')
print('    not fundamental astrometric noise (see nb12 spatially stratified bootstrap).')
print('  • This is a proof-of-concept result with a non-optimized image.')
""")


# =============================================================================
# CELL 10 — BOOTSTRAP HISTOGRAM
# =============================================================================
md("## Cell 10 — Bootstrap θ_north distribution\n")

code("""\
if s10 is None and s11 is None:
    print('No bootstrap data available.')
else:
    fig, axes = plt.subplots(1, 2 if (s10 and s11) else 1,
                              figsize=(14 if (s10 and s11) else 8, 5),
                              squeeze=False)

    def _plot_boot(ax, s, color, title):
        if s is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color='#aaa', fontsize=12)
            return
        dth = s['dth']
        ax.hist(dth, bins=min(15, len(dth)),
                color=color, edgecolor='#0d0d0d', alpha=0.9)
        ax.axvline(0, color='#ffe082', lw=2.2, ls='--', label='fiducial (Δθ=0)')
        ax.axvline(s['mean_offset'], color='#f5a623', lw=1.8,
                   label=f"mean  {s['mean_offset']:+.3f}°")
        ax.axvspan(-s['std'], s['std'], alpha=0.18, color='white',
                   label=f"±1σ = {s['std']:.4f}°")
        ax.set_xlabel('Δθ_north  (bootstrap − fiducial,  °)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    _plot_boot(axes[0, 0], s10, '#4fc3f7',
               f'nb10: random 80% of {s10["pool"] if s10 else "?"}\n'
               + (f'N={s10["n_ok"]},  σ={s10["std"]:.4f}°,  MAD={s10["mad"]:.4f}°' if s10 else ''))

    if s11 and axes.shape[1] > 1:
        _plot_boot(axes[0, 1], s11, '#ff6b6b',
                   f'nb11: random 80% of {s11["pool"] if s11 else "?"}\n'
                   + (f'N={s11["n_ok"]},  σ={s11["std"]:.4f}°,  MAD={s11["mad"]:.4f}°' if s11 else ''))

    fig.suptitle(
        'WCS θ_north bootstrap distributions  —  '
        'uncertainty in sky orientation term X',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / '06_bootstrap_histogram.png')
    plt.show()

    # Print the current best-estimate X
    _best_sigma = min(
        (s['std'] for s in [s10, s11] if s), default=float('nan')
    )
    print(f'\\nCurrent best estimate of X (WCS orientation uncertainty):')
    print(f'  X ≈ {_best_sigma:.4f}°  (1σ, nb10 bootstrap)')
    print(f'  (nb11 matched-source bootstrap gives {s11["std"]:.4f}° — higher because')
    print(f'   only 86 matched sources are re-sampled; harder to constrain high-order SIP)')
""")


# =============================================================================
# CELL 11 — RESIDUAL DIAGNOSTIC
# =============================================================================
md("""\
## Cell 11 — WCS residuals  *(optional diagnostic)*

Residuals between matched catalog star positions and their WCS-projected \
positions reveal whether the distortion model is working.  Random residuals \
indicate the SIP model is capturing the field geometry well.  A systematic \
trend with field radius would indicate under-fitting.
""")

code("""\
if fid_result.corr_table is None:
    print('corr_table not available — cannot compute residuals.')
else:
    _corr  = fid_result.corr_table
    print('Corr table columns:', _corr.colnames)

    _fx = np.asarray(_corr['field_x'], dtype=float)
    _fy = np.asarray(_corr['field_y'], dtype=float)
    _ir  = get_col(_corr, 'index_ra')
    _id  = get_col(_corr, 'index_dec')

    if _ir is None or _id is None:
        print('index_ra/dec not in corr table — skipping pixel residuals.')
        print('Available:', _corr.colnames)
    else:
        _ir = np.asarray(_ir, float)
        _id = np.asarray(_id, float)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FITSFixedWarning)
            _rx, _ry = fid_wcs.world_to_pixel_values(_ir, _id)
        _res_x = _fx - _rx
        _res_y = _fy - _ry
        _res   = np.hypot(_res_x, _res_y)
        _rad   = np.hypot(_fx - xc, _fy - yc)
        _res_arcsec = _res * _ps

        print(f'Matched sources : {len(_fx)}')
        print(f'RMS residual    : {_res.mean():.3f} px  ({_res_arcsec.mean():.2f}")')
        print(f'90th pct        : {np.percentile(_res, 90):.3f} px  '
              f'({np.percentile(_res_arcsec, 90):.2f}")')
        print(f'Mean residual   : ({_res_x.mean():+.3f}, {_res_y.mean():+.3f}) px  '
              f'{"← systematic offset present" if np.hypot(_res_x.mean(),_res_y.mean())>0.3 else "← random (good)"}')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.imshow(_disp, origin='lower', cmap='gray_r', vmin=_disp_lo, vmax=_disp_hi,
                  aspect='equal', interpolation='nearest', alpha=0.55)
        _sc = max(1.0, 3.0 / max(_res.max(), 0.01))
        ax.quiver(_fx, _fy, _res_x, _res_y, color='red',
                  angles='xy', scale_units='xy', scale=1.0/_sc,
                  width=0.002, alpha=0.9)
        ax.scatter(_fx, _fy, s=22, c='#f5a623', alpha=0.7, zorder=4)
        ax.set_xlim(0, nx); ax.set_ylim(0, ny)
        ax.set_title(f'Residual vectors (×{_sc:.0f})  —  rms {_res_arcsec.mean():.2f}"',
                     fontsize=10)

        ax2 = axes[1]
        sc2 = ax2.scatter(_rad, _res_arcsec, c=_res_arcsec, cmap='YlOrRd',
                          s=40, alpha=0.85, vmin=0)
        plt.colorbar(sc2, ax=ax2, label='Residual (arcsec)')
        _p = np.polyfit(_rad, _res_arcsec, 1)
        _rv = np.linspace(0, _rad.max(), 100)
        ax2.plot(_rv, np.polyval(_p, _rv), '--', color='cyan', lw=1.5,
                 label=f'trend {_p[0]*1000:+.3f}"/kpx')
        ax2.axhline(_res_arcsec.mean(), color='yellow', ls='--', lw=1.2,
                    label=f'mean {_res_arcsec.mean():.2f}"')
        ax2.set_xlabel('Radius from image center (px)'); ax2.set_ylabel('Residual (arcsec)')
        ax2.set_title('Residual vs field radius', fontsize=10)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        fig.suptitle('WCS fit residuals  (field − catalog projection)', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / '07_residuals.png')
        plt.show()
""")


# =============================================================================
# CELL 12 — FINAL SUMMARY
# =============================================================================
md("""\
## Cell 12 — Summary

### What works today

| Step | Status | Key number |
|------|--------|-----------|
| Star detection (with grating masking) | ✅ Working | 255 sources detected, 86 matched |
| Local plate-solve (order-5 SIP) | ✅ Working | ~3 s solve time, index 4117 |
| WCS center orientation θ_north | ✅ Working | −107.0° ± ~0.09° |
| WCS orientation bootstrap (nb10) | ✅ Done | σ(θ_north) = 0.092° |
| Diffraction trace detection | ⚠️ Prototype | Connected-component regionprops |
| Pixel-to-sky angle conversion | ✅ Implemented | `pixel_angle_to_sky_angle()` |
| Sky-frame grating angle θ_grid | ⚠️ Prototype-level | Depends on trace extraction quality |

### Current best estimates

- **X** (WCS orientation uncertainty): **≈ 0.09°** (nb10 bootstrap σ)
- **Y** (sky-frame grating angle): **see Cell 8** — θ_grid from prototype trace extraction
- Both estimates are for this specific non-optimized test image

### Key findings

1. **Local solve-field works reliably** — the switch from nova.astrometry.net to a \
local backend solved the API queue problem.  A typical solve takes ~3–30 s depending \
on source count.

2. **The WCS orientation is well-constrained** — bootstrap scatter is ~0.09° with the \
full detected source list.  The distribution has heavy tails (some large deviations) \
that are consistent with SIP instability under source sub-sampling, not fundamental \
astrometric noise.

3. **Source selection matters** — using only matched sources (nb11) gave a *larger* \
bootstrap scatter (0.13°) because fewer sources reduce spatial leverage for the \
high-order SIP fit.  Spatially balanced sub-sampling (notebook 12) is expected to \
reduce this.

4. **Trace detection is a functional prototype** — the connected-component extractor \
finds the diffraction features and extracts a consistent angle.  A proper Radon/Hough \
or ridge-filter pipeline would be more robust and give better angle uncertainties.

5. **The pixel→sky conversion is implemented** — `pixel_angle_to_sky_angle()` handles \
the ~0.04° N/E non-orthogonality correctly; simple subtraction (θ_sky ≈ θ_pixel − θ_north) \
is an adequate approximation at the current level of precision.

### What is needed next

- **Better trace extraction**: Radon transform or Sato/Meijering ridge filter for \
cleaner trace detection, RANSAC line fitting, multi-trace robust combination via \
axial circular statistics.
- **Optimized test image**: image with better SNR grating traces, cleaner PSF, \
known grating angle for ground-truth validation.
- **Spatially balanced bootstrap**: stratified sampling across the FOV to reduce \
source-coverage bias in the WCS uncertainty estimate.
- **End-to-end validation**: synthetic image (tele-img-sim) with known grating angle \
→ verify recovered θ_grid matches ground truth within ~0.01°.

### Conclusion

> The SpectrAngle proof-of-concept pipeline is working end-to-end: a real grating \
image can be plate-solved locally, the WCS orientation is measured at the ~0.1° \
level, diffraction features are detected and yield a pixel-space angle, and the \
full pixel→sky angle conversion is implemented.  The current precision is \
dominated by WCS model systematics and a prototype-level trace extractor, \
not fundamental limitations.  A clear path to improvement exists.
""")


# =============================================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (spectrangle WSL)",
            "language": "python",
            "name": "spectrangle-wsl"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"Notebook written to: {NB_PATH}")
print(f"Cells: {len(cells)}  ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
