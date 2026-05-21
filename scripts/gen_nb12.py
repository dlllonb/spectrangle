#!/usr/bin/env python3
"""
scripts/gen_nb12.py
Generate notebooks/12_wcs_orientation_diagnostics.ipynb
"""
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / '12_wcs_orientation_diagnostics.ipynb'

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "id": f"m{len(cells):03d}",
                  "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}",
                  "metadata": {}, "outputs": [], "execution_count": None,
                  "source": src})


# =============================================================================
# TITLE
# =============================================================================
md("""\
# 12 — WCS Orientation Diagnostics

**Purpose:** Diagnose why the WCS center north-angle bootstrap scatter is \
~0.09–0.13° and determine how to improve it.

Tests: source selection, spatial coverage, SIP stability, residual patterns, \
stratified bootstrap, affine cross-check, pixel→sky angle conversion.

- Notebook 10 (nb10): σ ≈ 0.092° — 80% random bootstrap on 255 detected sources
- Notebook 11 (nb11): σ ≈ 0.128° — 80% random bootstrap on 86 matched sources

Theoretical expectation: with this wide a field (~65°×46°) and many well-distributed \
matched stars the orientation should be much better constrained than ~0.1°.
If the WCS model is stable this scatter is systematic, not statistical.

**Run from the `Python (spectrangle WSL)` kernel. Uses local solve-field only.**
""")


# =============================================================================
# CELL 1 — ENVIRONMENT CHECK
# =============================================================================
md("## Cell 1 — Environment check\n")

code("""\
import sys, shutil, warnings, time, csv, json as _json, platform, pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table
from astropy import units as u

_pkg_root = str(Path('..').resolve())
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from extractor import extract_stars, make_xylist
from extractor.platesolve import (
    solve_plate, wcs_summary, LocalSolveFieldError, _col_array, get_col,
)
from extractor.wcsangle import (
    angle_diff_deg, center_wcs_angle_metrics, WcsAngleMetrics,
    local_north_angle_deg, local_east_angle_deg,
)

try:
    from extractor.wcsangle import pixel_angle_to_sky_angle, local_wcs_jacobian
    _has_pasa = True
    print('pixel_angle_to_sky_angle : available in extractor.wcsangle')
except ImportError:
    _has_pasa = False
    print('pixel_angle_to_sky_angle : NOT yet in wcsangle — will define locally in Cell 10')

# ── Shared configuration ─────────────────────────────────────────────────────
BACKEND_CFG   = '/mnt/c/Users/bassd/.astrometry/backend.cfg'
INDEX_DIR     = Path('/mnt/c/Users/bassd/astrometry-data/4100/')
FITS_PATH     = sorted((Path('..') / 'data').glob('*.fit'))[0]
OUT_DIR       = Path('..') / 'outputs' / 'wcs_orientation_diagnostics'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_LOW     = 70.0
SCALE_HIGH    = 95.0
TWEAK_ORDER   = 5
SOLVE_TIMEOUT = 120
INDEX_FILE    = str(INDEX_DIR / 'index-4117.fits')
BACKEND_CFG_STR = str(BACKEND_CFG)

# Suppress only harmless FITSFixedWarning/date warnings
def _make_wcs(header):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        warnings.filterwarnings('ignore', message='.*datfix.*')
        return WCS(header)

# ── Environment info ──────────────────────────────────────────────────────────
print(f'Python      : {sys.executable}')
print(f'Version     : {sys.version.split()[0]}')
print(f'Platform    : {platform.system()} {platform.release()}')
if platform.system() == 'Windows':
    print('*** WARNING: Windows kernel — local solve-field will fail.')
    print('    Switch to Python (spectrangle WSL) kernel.')

sf = shutil.which('solve-field')
print(f'solve-field : {sf or "NOT FOUND"}')
print(f'backend cfg : {BACKEND_CFG}')
print(f'  exists    : {Path(BACKEND_CFG).exists()}')
print(f'index dir   : {INDEX_DIR}  exists={INDEX_DIR.exists()}')
if INDEX_DIR.exists():
    for p in sorted(INDEX_DIR.glob('index-41*.fits')):
        print(f'  {p.name}')
import extractor
print(f'extractor   : {extractor.__version__}  ({extractor.__file__})')
print(f'output dir  : {OUT_DIR}')

plt.rcParams.update({
    'figure.facecolor': '#111', 'axes.facecolor': '#111',
    'text.color': 'white',      'axes.labelcolor': 'white',
    'xtick.color': '#ccc',      'ytick.color': '#ccc',
    'axes.edgecolor': '#555',   'axes.titlecolor': 'white',
    'legend.facecolor': '#1e1e1e', 'legend.edgecolor': '#555',
    'grid.color': '#333', 'grid.alpha': 0.4,
    'font.size': 10, 'axes.titlesize': 11,
    'savefig.facecolor': '#111', 'savefig.dpi': 150,
})
""")


# =============================================================================
# CELL 2 — LOAD IMAGE AND SOURCE DETECTION
# =============================================================================
md("## Cell 2 — Load image and run source detection\n")

code("""\
# Load image
with fits.open(FITS_PATH) as hdul:
    image       = hdul[0].data.astype(np.float32)
    orig_header = hdul[0].header.copy()

ny, nx = image.shape
xc_img = (nx - 1) / 2.
yc_img = (ny - 1) / 2.
print(f'Image  : {FITS_PATH.name}  ({nx} x {ny} px)')

# Background-subtracted display image (shared across cells)
_bg   = gaussian_filter(image, sigma=50)
_proc = np.clip(image - _bg, 0, None)
_lo, _hi = np.percentile(_proc[np.isfinite(_proc)], [0.5, 99.5])
disp = np.arcsinh(np.clip(_proc, _lo, _hi))
_dlo, _dhi = float(np.arcsinh(_lo)), float(np.arcsinh(_hi))

# Source extraction (share nb10 cache if available)
_src_nb10  = Path('..') / 'out' / 'local_bootstrap_10' / 'sources.pkl'
_src_cache = OUT_DIR / 'sources.pkl'
if _src_nb10.exists():
    with open(_src_nb10, 'rb') as fh:
        all_xs, all_ys, all_fluxes = pickle.load(fh)
    print(f'Sources : {len(all_xs)} (loaded from nb10 cache)')
elif _src_cache.exists():
    with open(_src_cache, 'rb') as fh:
        all_xs, all_ys, all_fluxes = pickle.load(fh)
    print(f'Sources : {len(all_xs)} (loaded from local cache)')
else:
    all_xs, all_ys, all_fluxes = extract_stars(image, max_sources=300, mask_spectra=True)
    with open(_src_cache, 'wb') as fh:
        pickle.dump((all_xs, all_ys, all_fluxes), fh)
    print(f'Sources : {len(all_xs)} (detected and cached)')

sel_xs, sel_ys, sel_fluxes = all_xs, all_ys, all_fluxes
print(f'Selected for solve: {len(sel_xs)}')

# Compute radius of each source from image center
src_r = np.hypot(sel_xs - xc_img, sel_ys - yc_img)
R_img = min(nx, ny) / 2.0   # conservative estimate of illuminated-region radius
print(f'Image half-size (R_img): {R_img:.0f} px')
print(f'Sources within R_img   : {(src_r < R_img).sum()}')
print(f'Sources within 0.9 R   : {(src_r < 0.9*R_img).sum()}')
print(f'Sources within 0.75 R  : {(src_r < 0.75*R_img).sum()}')

# Plot sources on image
fig, ax = plt.subplots(figsize=(13, 8.5))
ax.imshow(disp, origin='lower', cmap='gray_r', vmin=_dlo, vmax=_dhi,
          aspect='equal', interpolation='nearest')
ax.scatter(sel_xs, sel_ys, s=18, facecolors='none', edgecolors='lime',
           linewidths=0.8, alpha=0.7, label=f'detected ({len(sel_xs)})')
# Draw R_img circle
_theta = np.linspace(0, 2*np.pi, 500)
ax.plot(xc_img + R_img*np.cos(_theta), yc_img + R_img*np.sin(_theta),
        '--', color='cyan', lw=1.2, alpha=0.6, label=f'R_img={R_img:.0f} px')
ax.plot(xc_img + 0.9*R_img*np.cos(_theta), yc_img + 0.9*R_img*np.sin(_theta),
        ':', color='yellow', lw=1.0, alpha=0.5, label='0.9 R')
ax.plot(xc_img + 0.75*R_img*np.cos(_theta), yc_img + 0.75*R_img*np.sin(_theta),
        ':', color='orange', lw=1.0, alpha=0.5, label='0.75 R')
ax.scatter([xc_img], [yc_img], marker='+', s=150, c='red', zorder=5)
ax.legend(fontsize=9, loc='upper right')
ax.set_title(f'Detected sources on {FITS_PATH.name}')
ax.set_xlim(0, nx); ax.set_ylim(0, ny)
fig.tight_layout()
fig.savefig(OUT_DIR / 'source_overview.png', bbox_inches='tight')
plt.show()
""")


# =============================================================================
# CELL 3 — FIDUCIAL LOCAL ORDER-5 SOLVE
# =============================================================================
md("## Cell 3 — Fiducial local order-5 solve\n")

code("""\
FID_DIR   = OUT_DIR / 'fiducial'
FID_CACHE = OUT_DIR / 'fiducial_result.pkl'
_fid_nb10 = Path('..') / 'out' / 'local_bootstrap_10' / 'fiducial_result.pkl'

def _read_match_meta(output_dir):
    p = Path(output_dir) / 'xylist.match'
    if not p.exists():
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with fits.open(str(p)) as h:
                t = Table(h[1].data)
                return dict(indexid=int(t['INDEXID'][0]),
                            nmatch=int(t['NMATCH'][0]),
                            logodds=float(t['LOGODDS'][0]))
    except Exception:
        return {}

if _fid_nb10.exists():
    print('Loading fiducial from nb10 cache...')
    with open(_fid_nb10, 'rb') as fh:
        fid_result = pickle.load(fh)
elif FID_CACHE.exists():
    print('Loading fiducial from local cache...')
    with open(FID_CACHE, 'rb') as fh:
        fid_result = pickle.load(fh)
else:
    print(f'Running fiducial order-{TWEAK_ORDER} solve ({len(sel_xs)} sources)...')
    _t0 = time.time()
    fid_result = solve_plate(
        xs=sel_xs, ys=sel_ys,
        image_width=nx, image_height=ny,
        original_header=orig_header,
        backend='local', backend_config=BACKEND_CFG_STR,
        scale_units='arcsecperpix',
        scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
        tweak_order=TWEAK_ORDER,
        output_dir=FID_DIR, timeout=SOLVE_TIMEOUT, verbose=True,
    )
    if fid_result is None:
        raise RuntimeError('Fiducial solve failed — check solve-field output above.')
    with open(FID_CACHE, 'wb') as fh:
        pickle.dump(fid_result, fh)
    print(f'Solve finished in {time.time()-_t0:.1f}s')

fid_wcs     = _make_wcs(fid_result.header)
fid_metrics = center_wcs_angle_metrics(fid_wcs, image.shape, compute_east=True)
_scales_deg = proj_plane_pixel_scales(fid_wcs)
_ps_arcsec  = float(np.mean(np.abs(_scales_deg)) * 3600)
_fov_x_deg  = float(np.abs(_scales_deg[0]) * nx)
_fov_y_deg  = float(np.abs(_scales_deg[1]) * ny)
_mmeta      = _read_match_meta(FID_DIR if FID_DIR.exists() else
                                Path('..') / 'out' / 'local_bootstrap_10' / 'fiducial')

_orth_dep = None
if fid_metrics.east_angle_deg is not None:
    _orth_dep = abs(angle_diff_deg(fid_metrics.east_angle_deg,
                                    fid_metrics.north_angle_deg) - 90.0)

print()
print('=== Fiducial WCS ===')
print(f'  Selected sources  : {len(fid_result.detected_x)}')
print(f'  Matched sources   : {len(fid_result.matched_x)}')
print(f'  Index INDEXID     : {_mmeta.get("indexid", "n/a")}')
print(f'  Match logodds     : {_mmeta.get("logodds", "n/a")}')
print(f'  Plate scale       : {_ps_arcsec:.2f} arcsec/px')
print(f'  FOV               : {_fov_x_deg:.2f}° × {_fov_y_deg:.2f}°')
print(f'  Center RA/Dec     : {fid_metrics.ra_deg:.6f}° / {fid_metrics.dec_deg:.6f}°')
print(f'  θ_north (fiducial): {fid_metrics.north_angle_deg:.5f}°  (CCW from +x)')
print(f'  θ_east            : {fid_metrics.east_angle_deg:.5f}°')
if _orth_dep is not None:
    print(f'  N/E departure     : {_orth_dep:.5f}°  (from 90°)')
print()
print(wcs_summary(fid_result.header))
""")


# =============================================================================
# CELL 4 — MATCHED-SOURCE RESIDUAL ANALYSIS
# =============================================================================
md("""\
## Cell 4 — Matched-source residual analysis

Parse the corr table, compute pixel-space residuals between detected source
positions and WCS-reprojected catalog positions, and look for coherent patterns.
""")

code("""\
if fid_result.corr_table is None:
    print('ERROR: corr_table is None — re-run the fiducial solve.')
else:
    corr = fid_result.corr_table
    print('corr_table columns:', corr.colnames)
    print(f'Rows: {len(corr)}')

    field_x = np.asarray(corr['field_x'], dtype=float)
    field_y = np.asarray(corr['field_y'], dtype=float)

    # Try to get catalog RA/Dec for residual computation
    idx_ra_col  = get_col(corr, 'index_ra')
    idx_dec_col = get_col(corr, 'index_dec')

    if idx_ra_col is not None and idx_dec_col is not None:
        idx_ra  = np.asarray(idx_ra_col,  dtype=float)
        idx_dec = np.asarray(idx_dec_col, dtype=float)

        # Project catalog positions back to pixels using fiducial WCS
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FITSFixedWarning)
            ref_x, ref_y = fid_wcs.world_to_pixel_values(idx_ra, idx_dec)

        res_x   = field_x - ref_x
        res_y   = field_y - ref_y
        res_mag = np.hypot(res_x, res_y)
        res_arcsec = res_mag * _ps_arcsec

        radius = np.hypot(field_x - xc_img, field_y - yc_img)

        print(f'\\nResidual statistics:')
        print(f'  N matched         : {len(field_x)}')
        print(f'  RMS residual      : {res_mag.mean():.3f} px  ({res_arcsec.mean():.2f}")')
        print(f'  Median residual   : {np.median(res_mag):.3f} px')
        print(f'  90th percentile   : {np.percentile(res_mag, 90):.3f} px  '
              f'({np.percentile(res_arcsec, 90):.2f}")')
        print(f'  Max residual      : {res_mag.max():.3f} px  ({res_arcsec.max():.2f}")')

        # Check for coherent structure via mean residual direction
        _mean_res_x = res_x.mean()
        _mean_res_y = res_y.mean()
        print(f'  Mean residual vec : ({_mean_res_x:+.3f}, {_mean_res_y:+.3f}) px')
        if np.hypot(_mean_res_x, _mean_res_y) > 0.3 * res_mag.mean():
            print('  → Coherent systematic offset detected (mean ≫ 0) — '
                  'possible CRPIX/zeropoint bias.')
        else:
            print('  → Residuals appear random (mean ≈ 0).')

        # Radial trend: bin residuals by radius
        _rbins = np.linspace(0, radius.max(), 6)
        print('\\n  Residual by radius bin (arcsec):')
        for i in range(len(_rbins)-1):
            _mask = (radius >= _rbins[i]) & (radius < _rbins[i+1])
            if _mask.sum() > 0:
                print(f'    r=[{_rbins[i]:.0f},{_rbins[i+1]:.0f}]px  '
                      f'n={_mask.sum():2d}  '
                      f'rms={res_arcsec[_mask].mean():.2f}"')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Residual vector map
        ax = axes[0]
        ax.imshow(disp, origin='lower', cmap='gray_r', vmin=_dlo, vmax=_dhi,
                  aspect='equal', interpolation='nearest', alpha=0.6)
        _scale = max(1.0, 5.0 / max(res_mag.max(), 0.01))
        ax.quiver(field_x, field_y, res_x, res_y,
                  color='red', angles='xy', scale_units='xy',
                  scale=1.0/_scale, width=0.0025, alpha=0.9,
                  label=f'residual (×{_scale:.0f})')
        ax.scatter(field_x, field_y, s=18, c='orange', alpha=0.6, zorder=4)
        ax.scatter([xc_img], [yc_img], marker='+', s=200, c='cyan', zorder=5)
        ax.set_xlim(0, nx); ax.set_ylim(0, ny)
        ax.set_title('WCS residual vectors (field − catalog projection)')
        ax.legend(fontsize=8)

        # Residual magnitude vs radius
        ax2 = axes[1]
        sc = ax2.scatter(radius, res_arcsec,
                         c=np.degrees(np.arctan2(res_y, res_x)),
                         cmap='twilight', s=35, alpha=0.85)
        plt.colorbar(sc, ax=ax2, label='Residual angle (deg)')
        ax2.axhline(res_arcsec.mean(), color='yellow', ls='--',
                    lw=1.5, label=f'mean {res_arcsec.mean():.2f}"')
        # Fit a simple linear trend
        if len(radius) > 3:
            _p = np.polyfit(radius, res_arcsec, 1)
            _rv = np.linspace(0, radius.max(), 100)
            ax2.plot(_rv, np.polyval(_p, _rv), 'cyan', lw=1.5, ls='--',
                     label=f'trend {_p[0]*100:+.4f}"/100px')
        ax2.set_xlabel('Radius from image center (px)')
        ax2.set_ylabel('Residual magnitude (arcsec)')
        ax2.set_title('Residual vs field radius (colour = residual angle)')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUT_DIR / 'residual_analysis.png', bbox_inches='tight')
        plt.show()
        print(f'Saved {OUT_DIR}/residual_analysis.png')

        # Store for use in Cell 5
        _matched_field_x = field_x
        _matched_field_y = field_y
        _matched_res_mag = res_mag
    else:
        print('index_ra / index_dec columns not found in corr_table.')
        print('Available columns:', corr.colnames)
        print('Residual analysis skipped — corr file may be from an older solve.')
        _matched_field_x = field_x
        _matched_field_y = field_y
        _matched_res_mag = None
""")


# =============================================================================
# CELL 5 — SOURCE MASKS / FIELD CUTS
# =============================================================================
md("""\
## Cell 5 — Source masks and field cuts

Test whether the WCS orientation depends on which part of the image is used \
for the solve.  Each selection runs one order-5 local solve.
""")

code("""\
# Build candidate source selections
_selections = {}
_selections['all_detected']   = (sel_xs, sel_ys)
_selections['r<0.9R']         = (sel_xs[src_r < 0.9*R_img],  sel_ys[src_r < 0.9*R_img])
_selections['r<0.75R']        = (sel_xs[src_r < 0.75*R_img], sel_ys[src_r < 0.75*R_img])
_selections['r<0.5R']         = (sel_xs[src_r < 0.5*R_img],  sel_ys[src_r < 0.5*R_img])

# Brightness-ranked subsets (flux sorted brightest-first by extract_stars)
for _n in (50, 100, 150):
    _n = min(_n, len(sel_xs))
    _selections[f'top{_n}'] = (sel_xs[:_n], sel_ys[:_n])

# Low-residual matched subset (if Cell 4 succeeded)
if '_matched_res_mag' in dir() and _matched_res_mag is not None:
    _lr_mask = _matched_res_mag < np.percentile(_matched_res_mag, 50)
    _selections['matched_low_res50pct'] = (
        _matched_field_x[_lr_mask], _matched_field_y[_lr_mask]
    )

# Shared extra_args with field hint
_extra_fid = [
    '--index', INDEX_FILE,
    '--ra',    str(fid_metrics.ra_deg),
    '--dec',   str(fid_metrics.dec_deg),
    '--radius','15',
]

print(f'Running {len(_selections)} source-selection solves...')
print()

_cut_results = []
for _name, (_xs, _ys) in _selections.items():
    _n = len(_xs)
    if _n < 20:
        print(f'  {_name:<25}: skip ({_n} sources, too few)')
        _cut_results.append({'name': _name, 'n_sel': _n, 'status': 'too_few'})
        continue

    _odir  = OUT_DIR / f'cut_{_name}'
    _cache = OUT_DIR / f'cut_{_name}.pkl'

    if _cache.exists():
        with open(_cache, 'rb') as _fh:
            _r = pickle.load(_fh)
    else:
        _t0 = time.time()
        try:
            _r = solve_plate(
                xs=_xs, ys=_ys,
                image_width=nx, image_height=ny,
                original_header=orig_header,
                backend='local', backend_config=BACKEND_CFG_STR,
                scale_units='arcsecperpix',
                scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
                tweak_order=TWEAK_ORDER,
                output_dir=_odir, timeout=SOLVE_TIMEOUT, verbose=False,
                extra_args=_extra_fid,
            )
        except LocalSolveFieldError as _e:
            _r = None
            print(f'  {_name:<25}: solver error: {str(_e)[:60]}')
        if _r is not None:
            with open(_cache, 'wb') as _fh:
                pickle.dump(_r, _fh)
        _elapsed = time.time() - _t0

    if _r is None:
        _cut_results.append({'name': _name, 'n_sel': _n, 'status': 'no_solution'})
        print(f'  {_name:<25}: n={_n:<4}  NO SOLUTION')
        continue

    _wcs = _make_wcs(_r.header)
    _m   = center_wcs_angle_metrics(_wcs, image.shape, compute_east=False)
    _dth = angle_diff_deg(_m.north_angle_deg, fid_metrics.north_angle_deg)
    _ps  = float(np.mean(np.abs(proj_plane_pixel_scales(_wcs))) * 3600)
    _row = {
        'name': _name, 'n_sel': _n, 'n_matched': len(_r.matched_x),
        'north_angle_deg': _m.north_angle_deg, 'delta_theta': _dth,
        'ps_arcsec': _ps, 'status': 'ok',
    }
    _cut_results.append(_row)
    print(f'  {_name:<25}: n={_n:<4}  matched={len(_r.matched_x):<3}  '
          f'θ={_m.north_angle_deg:.4f}°  Δθ={_dth:+.4f}°  '
          f'scale={_ps:.1f}"/px')

# Summary table
print()
print(f'{"Name":<26} {"N_sel":>5} {"N_match":>7} {"θ_north":>10} {"Δθ":>8}')
print('-' * 60)
for _r in _cut_results:
    if _r.get('status') == 'ok':
        print(f'{_r["name"]:<26} {_r["n_sel"]:>5} {_r["n_matched"]:>7} '
              f'{_r["north_angle_deg"]:>10.4f} {_r["delta_theta"]:>+8.4f}')
    else:
        print(f'{_r["name"]:<26} {_r["n_sel"]:>5} {"—":>7} {"—":>10} '
              f'{_r["status"]:>8}')

# Plot θ_north vs selection
_ok_rows = [r for r in _cut_results if r.get('status') == 'ok']
if _ok_rows:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _names   = [r['name'] for r in _ok_rows]
    _thetas  = [r['north_angle_deg'] for r in _ok_rows]
    _deltas  = [r['delta_theta'] for r in _ok_rows]
    _n_match = [r['n_matched'] for r in _ok_rows]

    ax = axes[0]
    ax.barh(_names, _deltas, color='#ff6b6b', edgecolor='#111', alpha=0.85)
    ax.axvline(0, color='yellow', lw=1.5, ls='--', label='fiducial (Δθ=0)')
    ax.set_xlabel('Δθ_north = θ − θ_fiducial (°)')
    ax.set_title('North angle offset by source selection')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(_n_match, _deltas, c='#4fc3f7', s=60, alpha=0.9, zorder=3)
    for _nm, _dt, _na in zip(_n_match, _deltas, _names):
        ax2.annotate(_na, (_nm, _dt), fontsize=7, xytext=(4, 0),
                     textcoords='offset points', color='#ccc')
    ax2.axhline(0, color='yellow', lw=1.2, ls='--')
    ax2.set_xlabel('Matched source count')
    ax2.set_ylabel('Δθ_north (°)')
    ax2.set_title('Δθ vs matched count')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'source_cuts.png', bbox_inches='tight')
    plt.show()
    print(f'Saved {OUT_DIR}/source_cuts.png')

    # Which selection is most consistent with fiducial?
    _best = min(_ok_rows, key=lambda r: abs(r['delta_theta']))
    print(f'\\nSmallest deviation: {_best["name"]} (Δθ = {_best["delta_theta"]:+.5f}°)')
    _best_selection_name = _best['name']
else:
    print('No successful solves — cannot plot.')
    _best_selection_name = 'all_detected'
""")


# =============================================================================
# CELL 6 — SOURCE-COUNT DEPENDENCE TEST
# =============================================================================
md("""\
## Cell 6 — Source-count dependence test

Using the cleanest selection from Cell 5, vary the number of submitted sources
to test whether the WCS orientation converges with more sources.
""")

code("""\
# Use the cleanest selection identified in Cell 5, or fall back to all_detected
_base_name = _best_selection_name if '_best_selection_name' in dir() else 'all_detected'
_base_xs, _base_ys = _selections.get(_base_name, (sel_xs, sel_ys))
print(f'Base selection : {_base_name}  ({len(_base_xs)} sources, flux-sorted)')

# N values to test
_Ns = [30, 50, 75, 100, 150, 200, len(_base_xs)]
_Ns = sorted(set(min(n, len(_base_xs)) for n in _Ns))
print(f'N values       : {_Ns}')
print()

_count_results = []
for _n in _Ns:
    _xs_n = _base_xs[:_n]
    _ys_n = _base_ys[:_n]
    _odir  = OUT_DIR / f'count_n{_n:04d}'
    _cache = OUT_DIR / f'count_n{_n:04d}.pkl'

    if _cache.exists():
        with open(_cache, 'rb') as _fh:
            _r = pickle.load(_fh)
    else:
        _t0 = time.time()
        try:
            _r = solve_plate(
                xs=_xs_n, ys=_ys_n,
                image_width=nx, image_height=ny,
                original_header=orig_header,
                backend='local', backend_config=BACKEND_CFG_STR,
                scale_units='arcsecperpix',
                scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
                tweak_order=TWEAK_ORDER,
                output_dir=_odir, timeout=SOLVE_TIMEOUT, verbose=False,
                extra_args=_extra_fid,
            )
        except LocalSolveFieldError as _e:
            _r = None
        if _r is not None:
            with open(_cache, 'wb') as _fh:
                pickle.dump(_r, _fh)

    if _r is None:
        _count_results.append({'n': _n, 'status': 'no_solution'})
        print(f'  N={_n:<4}: NO SOLUTION')
        continue

    _wcs = _make_wcs(_r.header)
    _m   = center_wcs_angle_metrics(_wcs, image.shape, compute_east=False)
    _dth = angle_diff_deg(_m.north_angle_deg, fid_metrics.north_angle_deg)
    _count_results.append({
        'n': _n, 'n_matched': len(_r.matched_x),
        'north_angle_deg': _m.north_angle_deg, 'delta_theta': _dth,
        'status': 'ok',
    })
    print(f'  N={_n:<4}: matched={len(_r.matched_x):<3}  '
          f'θ={_m.north_angle_deg:.4f}°  Δθ={_dth:+.4f}°')

_ok_c = [r for r in _count_results if r.get('status') == 'ok']
if len(_ok_c) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _ns     = [r['n'] for r in _ok_c]
    _thetas = [r['north_angle_deg'] for r in _ok_c]
    _deltas = [r['delta_theta'] for r in _ok_c]
    _nm     = [r['n_matched'] for r in _ok_c]

    ax = axes[0]
    ax.plot(_ns, _deltas, 'o-', color='#ff6b6b', lw=1.5, ms=7)
    ax.axhline(0, color='yellow', ls='--', lw=1.2)
    ax.fill_between(_ns, -0.05, 0.05, alpha=0.15, color='cyan', label='±0.05°')
    ax.set_xlabel('N sources submitted')
    ax.set_ylabel('Δθ_north (°)')
    ax.set_title('North angle offset vs submitted source count')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(_ns, _nm, 's--', color='#4fc3f7', lw=1.5, ms=7)
    ax2.set_xlabel('N sources submitted')
    ax2.set_ylabel('Matched source count')
    ax2.set_title('Matched count vs submitted count')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'source_count_dependence.png', bbox_inches='tight')
    plt.show()
    print(f'Saved {OUT_DIR}/source_count_dependence.png')
""")


# =============================================================================
# CELL 7 — SPATIALLY STRATIFIED BOOTSTRAP
# =============================================================================
md("""\
## Cell 7 — Spatially stratified bootstrap

Instead of pure random 80% sampling (which can accidentally exclude whole
regions of the field), sample from a 3×3 spatial grid so every bootstrap
sample has astrometric leverage from across the full field.

Saves results to CSV after each solve; re-running this cell resumes from
where it left off.
""")

code("""\
# ── Parameters ────────────────────────────────────────────────────────────────
N_GRID_X       = 3
N_GRID_Y       = 3
BOOT_FRAC_STRAT = 0.80
N_BOOT_STRAT   = 50
STRAT_CSV      = OUT_DIR / 'strat_bootstrap.csv'
STRAT_BOOT_DIR = OUT_DIR / 'strat_solves'
STRAT_BOOT_DIR.mkdir(parents=True, exist_ok=True)

# Source pool: use all detected sources or cleanest selection
_pool_xs = sel_xs
_pool_ys = sel_ys
print(f'Source pool       : {len(_pool_xs)} sources')
print(f'Grid              : {N_GRID_X} x {N_GRID_Y}')
print(f'Bootstrap fraction: {BOOT_FRAC_STRAT:.0%}')
print(f'N_BOOT            : {N_BOOT_STRAT}')

# Assign each source to a grid cell
_cx = np.floor(_pool_xs / nx * N_GRID_X).astype(int).clip(0, N_GRID_X - 1)
_cy = np.floor(_pool_ys / ny * N_GRID_Y).astype(int).clip(0, N_GRID_Y - 1)
_cell_id = _cx + _cy * N_GRID_X
_n_cells = N_GRID_X * N_GRID_Y

print('\\nGrid cell populations:')
for _cid in range(_n_cells):
    _m = (_cell_id == _cid).sum()
    print(f'  cell {_cid}: {_m} sources')

def _stratified_sample(rng, frac=BOOT_FRAC_STRAT):
    idx = []
    for _cid in range(_n_cells):
        _members = np.where(_cell_id == _cid)[0]
        if len(_members) == 0:
            continue
        _n_take = max(1, int(len(_members) * frac))
        idx.append(rng.choice(_members, size=_n_take, replace=False))
    return np.concatenate(idx) if idx else np.array([], dtype=int)

# CSV header
_CSV_HDR = ['boot_idx', 'n_src', 'north_angle_deg', 'ra_deg', 'dec_deg', 'elapsed_s', 'status']
if not STRAT_CSV.exists():
    with open(STRAT_CSV, 'w', newline='', encoding='utf-8') as _f:
        csv.writer(_f).writerow(_CSV_HDR)

_done = {}
with open(STRAT_CSV, newline='', encoding='utf-8') as _f:
    for _row in csv.DictReader(_f):
        if _row['status'] == 'ok':
            _done[int(_row['boot_idx'])] = _row
print(f'\\nResuming: {len(_done)} successful results already saved.')

rng_strat = np.random.default_rng(999)
_t_start  = time.time()

for _i in range(N_BOOT_STRAT):
    if _i in _done:
        continue

    _idx   = _stratified_sample(rng_strat)
    _sub_x = _pool_xs[_idx]
    _sub_y = _pool_ys[_idx]
    _odir  = STRAT_BOOT_DIR / f'iter_{_i:03d}'
    _t0    = time.time()
    _status = 'ok'
    _m     = None

    # Re-read existing WCS without re-running
    _wcs_f = _odir / 'xylist.wcs'
    if _wcs_f.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with fits.open(str(_wcs_f)) as _h:
                    _hdr = _h[0].header.copy()
            _m = center_wcs_angle_metrics(_make_wcs(_hdr), image.shape, compute_east=False)
        except Exception as _e:
            _status = f'wcs_read_error: {str(_e)[:60]}'
    else:
        try:
            _r = solve_plate(
                xs=_sub_x, ys=_sub_y,
                image_width=nx, image_height=ny,
                original_header=orig_header,
                backend='local', backend_config=BACKEND_CFG_STR,
                scale_units='arcsecperpix',
                scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
                tweak_order=TWEAK_ORDER,
                output_dir=_odir, timeout=SOLVE_TIMEOUT, verbose=False,
                extra_args=_extra_fid,
            )
            if _r is None:
                _status = 'no_solution'
            else:
                _m = center_wcs_angle_metrics(_make_wcs(_r.header), image.shape,
                                              compute_east=False)
        except LocalSolveFieldError as _e:
            _status = f'solver_error: {str(_e)[:60]}'
        except Exception as _e:
            _status = f'error: {type(_e).__name__}: {str(_e)[:60]}'

    _elapsed = time.time() - _t0
    if _m is not None:
        _dth = angle_diff_deg(_m.north_angle_deg, fid_metrics.north_angle_deg)
        print(f'{_i:03d}/{N_BOOT_STRAT}  n={len(_sub_x)}  '
              f'θ={_m.north_angle_deg:.4f}°  Δθ={_dth:+.4f}°  [{_elapsed:.0f}s]')
    else:
        print(f'{_i:03d}/{N_BOOT_STRAT}  FAILED  [{_status[:50]}  {_elapsed:.0f}s]')

    _row = [_i, len(_sub_x),
            _m.north_angle_deg if _m else '',
            _m.ra_deg          if _m else '',
            _m.dec_deg         if _m else '',
            f'{_elapsed:.2f}', _status]
    with open(STRAT_CSV, 'a', newline='', encoding='utf-8') as _f:
        csv.writer(_f).writerow(_row)

print(f'\\nStratified bootstrap loop finished in {(time.time()-_t_start)/60:.1f} min.')

# ── Statistics ────────────────────────────────────────────────────────────────
_strat_rows = []
with open(STRAT_CSV, newline='', encoding='utf-8') as _f:
    for _row in csv.DictReader(_f):
        if _row['status'] == 'ok' and _row['north_angle_deg']:
            _strat_rows.append(float(_row['north_angle_deg']))

if len(_strat_rows) >= 2:
    _strat_arr   = np.array(_strat_rows)
    _strat_dth   = angle_diff_deg(_strat_arr, fid_metrics.north_angle_deg)
    _strat_std   = float(_strat_dth.std())
    _strat_mad   = float(np.median(np.abs(_strat_dth - np.median(_strat_dth))))
    _strat_mean  = float(_strat_dth.mean())

    # Load nb10/nb11 for comparison
    _nb10_std = _nb11_std = None
    _nb10_csv = Path('..') / 'out' / 'local_bootstrap_10' / 'bootstrap_results.csv'
    _nb11_csv = Path('..') / 'out' / 'local_bootstrap_11' / 'bootstrap_results.csv'
    for _csv_path, _attr_name in [(_nb10_csv, '_nb10_std'), (_nb11_csv, '_nb11_std')]:
        if _csv_path.exists():
            _rows = []
            with open(_csv_path, newline='', encoding='utf-8') as _f:
                for _row in csv.DictReader(_f):
                    if _row['status'] == 'ok' and _row['north_angle_deg']:
                        _rows.append(float(_row['north_angle_deg']))
            if _rows:
                _dth = angle_diff_deg(np.array(_rows), fid_metrics.north_angle_deg)
                if _attr_name == '_nb10_std':
                    _nb10_std = float(_dth.std())
                else:
                    _nb11_std = float(_dth.std())

    print('=' * 60)
    print('  BOOTSTRAP COMPARISON')
    print('=' * 60)
    print(f'  nb10 (random, full list)  : σ = {_nb10_std:.5f}°' if _nb10_std else
          '  nb10 : CSV not found')
    print(f'  nb11 (random, matched)    : σ = {_nb11_std:.5f}°' if _nb11_std else
          '  nb11 : CSV not found')
    print(f'  nb12 (stratified 3x3)     : σ = {_strat_std:.5f}°   ← NEW')
    print(f'  nb12 MAD                  : {_strat_mad:.5f}°')
    print(f'  nb12 mean offset          : {_strat_mean:+.5f}°')
    print('=' * 60)

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(_strat_dth, bins=min(15, len(_strat_rows)),
            color='#4fc3f7', edgecolor='#111', alpha=0.85, label='stratified 3×3 (nb12)')
    ax.axvline(0, color='yellow', lw=2, ls='--', label='fiducial (0)')
    ax.axvline(_strat_mean, color='lime', lw=1.5, label=f'mean {_strat_mean:+.4f}°')
    ax.axvspan(-_strat_std, _strat_std, alpha=0.18, color='cyan',
               label=f'±1σ = {_strat_std:.4f}°')
    ax.set_xlabel('Δθ_north  (bootstrap − fiducial, °)')
    ax.set_ylabel('Count')
    ax.set_title(f'Stratified 3×3 bootstrap  (N={len(_strat_rows)}, '
                 f'frac={BOOT_FRAC_STRAT:.0%})  σ = {_strat_std:.4f}°')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'strat_bootstrap_hist.png', bbox_inches='tight')
    plt.show()
    print(f'Saved {OUT_DIR}/strat_bootstrap_hist.png')

    _strat_sigma = _strat_std   # store for Cell 11
""")


# =============================================================================
# CELL 8 — BOOTSTRAP FRACTION SENSITIVITY
# =============================================================================
md("""\
## Cell 8 — Bootstrap fraction sensitivity

Using the stratified 3×3 method, test fractions 0.80, 0.90, and 0.95 with \
20 samples each to see how strongly the scatter depends on the removed fraction.
""")

code("""\
_FRACS      = [0.80, 0.90, 0.95]
N_BOOT_FRAC = 20
rng_frac    = np.random.default_rng(777)

_frac_summary = {}

for _frac in _FRACS:
    _frac_csv = OUT_DIR / f'frac_boot_{int(_frac*100):02d}.csv'
    _frac_dir = OUT_DIR / f'frac_solves_{int(_frac*100):02d}'
    _frac_dir.mkdir(parents=True, exist_ok=True)

    if not _frac_csv.exists():
        with open(_frac_csv, 'w', newline='', encoding='utf-8') as _f:
            csv.writer(_f).writerow(['i', 'n_src', 'north_angle_deg', 'status'])

    _done_frac = set()
    with open(_frac_csv, newline='', encoding='utf-8') as _f:
        for _row in csv.DictReader(_f):
            if _row['status'] == 'ok':
                _done_frac.add(int(_row['i']))

    for _i in range(N_BOOT_FRAC):
        if _i in _done_frac:
            continue
        _idx   = _stratified_sample(rng_frac, frac=_frac)
        _sub_x = _pool_xs[_idx]
        _sub_y = _pool_ys[_idx]
        _odir  = _frac_dir / f'iter_{_i:03d}'
        try:
            _r = solve_plate(
                xs=_sub_x, ys=_sub_y,
                image_width=nx, image_height=ny,
                original_header=orig_header,
                backend='local', backend_config=BACKEND_CFG_STR,
                scale_units='arcsecperpix',
                scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
                tweak_order=TWEAK_ORDER,
                output_dir=_odir, timeout=SOLVE_TIMEOUT, verbose=False,
                extra_args=_extra_fid,
            )
        except LocalSolveFieldError:
            _r = None
        if _r is not None:
            _m = center_wcs_angle_metrics(_make_wcs(_r.header), image.shape,
                                          compute_east=False)
            with open(_frac_csv, 'a', newline='', encoding='utf-8') as _f:
                csv.writer(_f).writerow([_i, len(_sub_x), _m.north_angle_deg, 'ok'])
        else:
            with open(_frac_csv, 'a', newline='', encoding='utf-8') as _f:
                csv.writer(_f).writerow([_i, len(_sub_x), '', 'no_solution'])

    # Collect results
    _rows_frac = []
    with open(_frac_csv, newline='', encoding='utf-8') as _f:
        for _row in csv.DictReader(_f):
            if _row['status'] == 'ok' and _row['north_angle_deg']:
                _rows_frac.append(float(_row['north_angle_deg']))

    if len(_rows_frac) >= 2:
        _dth = angle_diff_deg(np.array(_rows_frac), fid_metrics.north_angle_deg)
        _std = float(_dth.std())
        _mad = float(np.median(np.abs(_dth - np.median(_dth))))
        _frac_summary[_frac] = {'n_ok': len(_rows_frac), 'std': _std, 'mad': _mad}
        print(f'  frac={_frac:.2f}  n_ok={len(_rows_frac):2d}  '
              f'σ={_std:.5f}°  MAD={_mad:.5f}°')

print()
print('Fraction sensitivity summary:')
print(f'  {"Frac":>5}  {"N_ok":>5}  {"σ (deg)":>10}  {"MAD (deg)":>10}')
print('-' * 40)
for _frac, _s in sorted(_frac_summary.items()):
    print(f'  {_frac:>5.2f}  {_s["n_ok"]:>5}  {_s["std"]:>10.5f}  {_s["mad"]:>10.5f}')

if _frac_summary:
    fig, ax = plt.subplots(figsize=(7, 4))
    _fs  = sorted(_frac_summary.keys())
    _std_vals = [_frac_summary[f]['std'] for f in _fs]
    _mad_vals = [_frac_summary[f]['mad'] for f in _fs]
    ax.plot(_fs, _std_vals, 'o-', color='#ff6b6b', lw=2, ms=8, label='σ (std)')
    ax.plot(_fs, _mad_vals, 's--', color='#4fc3f7', lw=2, ms=8, label='MAD')
    ax.set_xlabel('Bootstrap fraction')
    ax.set_ylabel('Scatter (°)')
    ax.set_title('Orientation scatter vs bootstrap fraction (stratified 3×3)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'frac_sensitivity.png', bbox_inches='tight')
    plt.show()
    print(f'Saved {OUT_DIR}/frac_sensitivity.png')
""")


# =============================================================================
# CELL 9 — LOCAL AFFINE ORIENTATION CROSS-CHECK
# =============================================================================
md("""\
## Cell 9 — Local affine orientation estimator cross-check

Fit a local affine transform from pixel offsets (from image center) to \
sky tangent-plane offsets, using only the matched sources.  Extract the \
rotation angle from the affine matrix and compare it to the WCS-derivative \
north angle from Cell 3.

This cross-check is independent of the SIP model; it uses only the \
matched star positions and their catalog coordinates.
""")

code("""\
from astropy.coordinates import SkyCoord

if fid_result.corr_table is None:
    print('corr_table not available — skip Cell 9.')
else:
    _corr = fid_result.corr_table
    _fx   = np.asarray(_corr['field_x'], dtype=float)
    _fy   = np.asarray(_corr['field_y'], dtype=float)

    _idx_ra  = get_col(_corr, 'index_ra')
    _idx_dec = get_col(_corr, 'index_dec')

    if _idx_ra is None or _idx_dec is None:
        print('index_ra / index_dec not in corr_table — skip affine test.')
    else:
        _idx_ra  = np.asarray(_idx_ra,  dtype=float)
        _idx_dec = np.asarray(_idx_dec, dtype=float)
        n_match  = len(_fx)

        # Sky coordinate of image center
        _c_ctr = fid_wcs.pixel_to_world(xc_img, yc_img)

        # Project catalog coords into a tangent plane centred at image center
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _cat_coords = SkyCoord(ra=_idx_ra * u.deg, dec=_idx_dec * u.deg)
            _frame_ctr  = _c_ctr.skyoffset_frame()
            _offsets    = _cat_coords.transform_to(_frame_ctr)

        # Tangent-plane coords in arcsec (east = +lon, north = +lat)
        _eta = np.array([float(o.lat.arcsec) for o in _offsets])  # north
        _xi  = np.array([float(o.lon.wrap_at(180*u.deg).arcsec) for o in _offsets])  # east

        # Pixel offsets from image center
        _dx = _fx - xc_img
        _dy = _fy - yc_img

        # Fit affine: [xi, eta] = A @ [dx, dy] + b
        # Build design matrix [dx, dy, 1] per source
        _D = np.column_stack([_dx, _dy, np.ones(n_match)])
        _Y = np.column_stack([_xi, _eta])

        # Least-squares fit
        _Amat, _res, _rank, _sv = np.linalg.lstsq(_D, _Y, rcond=None)
        # _Amat[0] = [dxi/dx, dxi/dy, xi_offset]
        # _Amat[1] = [deta/dx, deta/dy, eta_offset]

        # Extract rotation: affine maps pixel to sky tangent plane
        # North direction in pixel space: solve A @ [dx, dy]^T = [0, 1] (north)
        # The matrix [[dxi/dx, dxi/dy], [deta/dx, deta/dy]] maps pixel vectors to sky vectors
        _J = _Amat[:2, :2].T    # 2×2 pixel→sky Jacobian (row = pixel coord)

        # North vector in sky tangent plane is [xi=0, eta=1]
        # Solve J @ v_north = [0, 1] for v_north pixel direction
        # Equivalently: north pixel direction = J^{-1} @ [0, 1]
        try:
            _J_inv  = np.linalg.inv(_J)
            _v_north = _J_inv @ np.array([0.0, 1.0])  # north in pixel space
            _affine_north_deg = float(np.degrees(np.arctan2(_v_north[1], _v_north[0])))
        except np.linalg.LinAlgError:
            _affine_north_deg = None
            print('Affine matrix singular — cannot extract north angle.')

        # Also extract scale from affine
        _plate_scale_arcsec_px = float(np.sqrt(np.abs(np.linalg.det(_J))))

        # Fit residuals
        _Y_pred  = _D @ _Amat
        _res_aff = np.hypot(_Y[:,0] - _Y_pred[:,0], _Y[:,1] - _Y_pred[:,1])

        print('=== Affine cross-check ===')
        print(f'  N matched sources : {n_match}')
        print(f'  Affine residual   : {_res_aff.mean():.3f}" (mean)  '
              f'{_res_aff.max():.3f}" (max)')
        print(f'  Affine plate scale: {_plate_scale_arcsec_px:.3f} arcsec/px')
        if _affine_north_deg is not None:
            _diff = angle_diff_deg(_affine_north_deg, fid_metrics.north_angle_deg)
            print(f'  Affine θ_north    : {_affine_north_deg:.5f}°')
            print(f'  WCS-deriv θ_north : {fid_metrics.north_angle_deg:.5f}°')
            print(f'  Difference        : {_diff:+.5f}°')
            if abs(_diff) < 0.01:
                print('  → Excellent agreement — WCS derivative is reliable.')
            elif abs(_diff) < 0.05:
                print('  → Good agreement — small difference, both methods consistent.')
            else:
                print('  → Notable difference — check WCS or affine fit quality.')

        # Repeat for different radius cuts
        print()
        print(f'  {"Radius cut":>12}  {"N":>4}  {"θ_affine":>10}  {"Δθ (vs WCS)":>12}')
        print('  ' + '-' * 45)
        for _rfrac in [0.5, 0.75, 1.0]:
            _rm = np.hypot(_fx - xc_img, _fy - yc_img) < _rfrac * R_img
            if _rm.sum() < 4:
                continue
            _Dm  = np.column_stack([_dx[_rm], _dy[_rm], np.ones(_rm.sum())])
            _Ym  = np.column_stack([_xi[_rm], _eta[_rm]])
            _Am, _, _, _ = np.linalg.lstsq(_Dm, _Ym, rcond=None)
            _Jm  = _Am[:2, :2].T
            try:
                _vn   = np.linalg.inv(_Jm) @ np.array([0.0, 1.0])
                _thn  = float(np.degrees(np.arctan2(_vn[1], _vn[0])))
                _ddth = angle_diff_deg(_thn, fid_metrics.north_angle_deg)
                print(f'  r < {_rfrac:.2f} R_img  {_rm.sum():>4}  '
                      f'{_thn:>10.4f}°  {_ddth:>+12.5f}°')
            except np.linalg.LinAlgError:
                print(f'  r < {_rfrac:.2f} R_img  {_rm.sum():>4}  SINGULAR')
""")


# =============================================================================
# CELL 10 — PIXEL ANGLE TO SKY ANGLE DIAGNOSTIC
# =============================================================================
md("""\
## Cell 10 — `pixel_angle_to_sky_angle` diagnostic

Test the utility that maps an arbitrary pixel-space direction through the \
local WCS Jacobian into a sky tangent-plane angle.  Compare it to the simple \
subtraction approximation (subtract θ_north) and quantify the difference.

The ~0.04° north/east non-orthogonality means that the simple subtraction \
may differ from the full Jacobian result by up to that amount for directions \
near east.
""")

code("""\
# ── Ensure pixel_angle_to_sky_angle is available ──────────────────────────────
if not _has_pasa:
    # Define locally if not yet in package
    def pixel_angle_to_sky_angle(wcs, x, y, theta_pix_deg, step_px=10.0):
        \"\"\"Map pixel-space direction to sky east-of-north PA.\"\"\"
        from astropy.coordinates import SkyCoord
        theta_rad = np.radians(theta_pix_deg)
        x2 = float(x) + step_px * np.cos(theta_rad)
        y2 = float(y) + step_px * np.sin(theta_rad)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            c0 = fid_wcs.pixel_to_world(float(x), float(y))
            c1 = fid_wcs.pixel_to_world(x2, y2)
        frame  = c0.skyoffset_frame()
        c1_off = c1.transform_to(frame)
        lon_deg = float(c1_off.lon.wrap_at(180 * u.deg).deg)
        lat_deg = float(c1_off.lat.deg)
        return float(np.degrees(np.arctan2(lon_deg, lat_deg)))
    print('Using locally-defined pixel_angle_to_sky_angle.')
else:
    print('Using pixel_angle_to_sky_angle from extractor.wcsangle.')

# ── Test at image center ──────────────────────────────────────────────────────
print()
print(f'Image center: ({xc_img:.1f}, {yc_img:.1f})')
print(f'θ_north       = {fid_metrics.north_angle_deg:.5f}°  (CCW from +x)')
print(f'θ_east        = {fid_metrics.east_angle_deg:.5f}°')
print(f'N/E departure = {_orth_dep:.5f}°  from 90°' if _orth_dep is not None
      else 'N/E departure : not computed')

# Test pixel angles: 0, 45, 90, 135, 180, ... in pixel space
# Simple approximation: sky_angle ≈ theta_pix - theta_north  (for orientation)
# Full result: pixel_angle_to_sky_angle(...)
print()
print(f'  {"θ_pix":>8}  {"sky (full)":>12}  {"sky (simple)":>14}  {"Δ (full-simple)":>16}')
print('  ' + '-' * 58)

_theta_pix_tests = np.arange(0, 360, 15)
_full_vals  = []
_simple_vals = []
_diffs       = []

for _tp in _theta_pix_tests:
    _sky_full   = pixel_angle_to_sky_angle(fid_wcs, xc_img, yc_img, _tp)
    _sky_simple = angle_diff_deg(_tp - fid_metrics.north_angle_deg, 0.0)
    # Wrap to (-180, 180]
    _sky_simple = (_sky_simple + 180) % 360 - 180
    _diff       = angle_diff_deg(_sky_full, _sky_simple)
    _full_vals.append(_sky_full)
    _simple_vals.append(_sky_simple)
    _diffs.append(_diff)
    print(f'  {_tp:>8.1f}°  {_sky_full:>12.4f}°  {_sky_simple:>14.4f}°  {_diff:>+16.5f}°')

_max_diff = float(np.max(np.abs(_diffs)))
print()
print(f'Max |full − simple| over all tested directions: {_max_diff:.5f}°')
if _max_diff < 0.01:
    print('→ Simple subtraction is adequate at image center (<0.01° error).')
elif _max_diff < _orth_dep if _orth_dep else 0.1:
    print('→ Small but non-negligible difference; equals the N/E departure.')
else:
    print(f'→ Significant difference ({_max_diff:.4f}°) — use full Jacobian method.')

# Plot sky angle vs pixel angle
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(_theta_pix_tests, _full_vals, 'o-', color='#ff6b6b', lw=1.5, ms=5,
        label='full Jacobian')
ax.plot(_theta_pix_tests, _simple_vals, 's--', color='#4fc3f7', lw=1.5, ms=5,
        label='simple subtraction')
ax.set_xlabel('Pixel direction θ_pix (°)')
ax.set_ylabel('Sky direction east of north (°)')
ax.set_title('Sky direction vs pixel direction at image center')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(_theta_pix_tests, _diffs, 'o-', color='#ffe082', lw=1.5, ms=5)
ax2.axhline(0, color='#555', lw=1.0)
ax2.axhline(_max_diff, color='red', ls=':', lw=1.2,
            label=f'max |Δ| = {_max_diff:.5f}°')
ax2.axhline(-_max_diff, color='red', ls=':', lw=1.2)
ax2.set_xlabel('Pixel direction θ_pix (°)')
ax2.set_ylabel('Δ = full − simple (°)')
ax2.set_title('Difference: full Jacobian vs simple subtraction')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / 'pix_to_sky_angle.png', bbox_inches='tight')
plt.show()

# Also test at a few off-center positions
print()
print('Off-center comparison (at several field positions):')
print(f'  {"Position":>20}  {"full":>10}  {"simple":>10}  {"Δ":>8}')
print('  ' + '-' * 55)
_test_px = 0.0   # use 0 deg pixel angle for comparison
for _px, _py in [(xc_img, yc_img),
                 (nx*0.1, ny*0.1), (nx*0.9, ny*0.1),
                 (nx*0.1, ny*0.9), (nx*0.9, ny*0.9),
                 (nx*0.5, ny*0.1), (nx*0.5, ny*0.9)]:
    _sf = pixel_angle_to_sky_angle(fid_wcs, _px, _py, _test_px)
    _tn = local_north_angle_deg(fid_wcs, _px, _py) if _has_pasa else \
          local_north_angle_deg(fid_wcs, _px, _py)
    _ss = angle_diff_deg(_test_px - _tn, 0.0)
    _ss = (_ss + 180) % 360 - 180
    _dd = angle_diff_deg(_sf, _ss)
    print(f'  ({_px:5.0f}, {_py:5.0f})          {_sf:>10.4f}°  {_ss:>10.4f}°  {_dd:>+8.5f}°')
""")


# =============================================================================
# CELL 11 — BEST WCS OVERLAY AND DIAGNOSTIC SUMMARY
# =============================================================================
md("""\
## Cell 11 — Best current WCS overlay and diagnostic summary

Overlay the fiducial WCS grid on the image, then print a collected summary \
of all diagnostic results and the recommended X value for the paper.
""")

code("""\
from astropy.visualization.wcsaxes import WCSAxes  # ensure import

# ── WCS overlay ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9.5))
ax  = fig.add_subplot(111, projection=fid_wcs)
ax.imshow(disp, origin='lower', cmap='gray_r', vmin=_dlo, vmax=_dhi,
          aspect='equal', interpolation='nearest')
ax.coords.grid(True, color='cyan', alpha=0.35, linestyle='--', linewidth=0.8)
ax.coords['ra'].set_major_formatter('hh:mm')
ax.coords['dec'].set_major_formatter('dd:mm')
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')

# Detected sources
ax.scatter(fid_result.detected_x, fid_result.detected_y,
           transform=ax.get_transform('pixel'),
           s=16, facecolors='none', edgecolors='lime', linewidths=0.8,
           alpha=0.6, label=f'detected ({len(fid_result.detected_x)})')

# Matched sources
if fid_result.corr_table is not None:
    _mfx = np.asarray(fid_result.corr_table['field_x'], dtype=float)
    _mfy = np.asarray(fid_result.corr_table['field_y'], dtype=float)
    ax.scatter(_mfx, _mfy, transform=ax.get_transform('pixel'),
               s=55, facecolors='none', edgecolors='orange', linewidths=1.3,
               alpha=0.85, label=f'matched ({len(_mfx)})')

# Residual vectors if available
if '_matched_res_mag' in dir() and _matched_res_mag is not None:
    _scale_v = max(1.0, 5.0 / max(res_mag.max(), 0.01))
    ax.quiver(_matched_field_x, _matched_field_y, res_x, res_y,
              transform=ax.get_transform('pixel'),
              color='red', angles='xy', scale_units='xy',
              scale=1.0/_scale_v, width=0.0015, alpha=0.7)

ax.legend(fontsize=9, loc='upper right')

_boot_str = f'σ(θ) = {_strat_sigma:.4f}° (stratified 3×3)' if '_strat_sigma' in dir() else ''
_ann = (
    f'Order {TWEAK_ORDER} SIP  |  {_ps_arcsec:.1f} arcsec/px\\n'
    f'FOV {_fov_x_deg:.1f}° × {_fov_y_deg:.1f}°\\n'
    f'Center RA {fid_metrics.ra_deg:.4f}°  Dec {fid_metrics.dec_deg:.4f}°\\n'
    f'θ_north = {fid_metrics.north_angle_deg:.4f}°\\n'
    + _boot_str
)
ax.text(0.01, 0.01, _ann, transform=ax.transAxes,
        fontsize=8.5, color='white', verticalalignment='bottom',
        bbox=dict(facecolor='#111', alpha=0.75, edgecolor='#555',
                  boxstyle='round,pad=0.4'))
ax.set_title(
    f'Fiducial WCS (order {TWEAK_ORDER})  —  {FITS_PATH.name}  '
    f'  detected {len(fid_result.detected_x)}  matched {len(fid_result.matched_x)}',
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fiducial_wcs_overlay.png', bbox_inches='tight')
plt.show()
print(f'Saved {OUT_DIR}/fiducial_wcs_overlay.png')

# ── Printed diagnostic summary ─────────────────────────────────────────────────
print()
print('=' * 70)
print('  DIAGNOSTIC SUMMARY')
print('=' * 70)
print(f'  Image              : {FITS_PATH.name}  ({nx} x {ny} px)')
print(f'  Detected sources   : {len(fid_result.detected_x)}')
print(f'  Matched sources    : {len(fid_result.matched_x)}')
print(f'  Plate scale        : {_ps_arcsec:.2f} arcsec/px')
print(f'  FOV                : {_fov_x_deg:.2f}° × {_fov_y_deg:.2f}°')
print(f'  Center RA/Dec      : {fid_metrics.ra_deg:.6f}° / {fid_metrics.dec_deg:.6f}°')
print(f'  θ_north (fiducial) : {fid_metrics.north_angle_deg:.5f}°')
print(f'  θ_east  (fiducial) : {fid_metrics.east_angle_deg:.5f}°')
print(f'  N/E departure      : {_orth_dep:.5f}° from 90°' if _orth_dep is not None
      else '  N/E departure      : not computed')
print()
print('  Bootstrap results:')
if '_nb10_std' in dir() and _nb10_std:
    print(f'    nb10 (random 80%, 255 srcs)        : σ = {_nb10_std:.5f}°')
if '_nb11_std' in dir() and _nb11_std:
    print(f'    nb11 (random 80%, 86 matched)       : σ = {_nb11_std:.5f}°')
if '_strat_sigma' in dir():
    print(f'    nb12 (stratified 3×3, 80%)          : σ = {_strat_sigma:.5f}°  ← NEW')
if '_frac_summary' in dir() and _frac_summary:
    for _fr, _fs in sorted(_frac_summary.items()):
        print(f'    nb12 (stratified, frac={_fr:.2f})       : σ = {_fs["std"]:.5f}°')
print()
if '_matched_res_mag' in dir() and _matched_res_mag is not None:
    print(f'  WCS residual RMS   : {res_mag.mean():.3f} px  ({res_arcsec.mean():.2f}")')
if '_max_diff' in dir():
    print(f'  Max full-vs-simple Δθ : {_max_diff:.5f}°  (pixel→sky at center)')
print()
# Recommended X value
_sigma_vals = []
if '_strat_sigma' in dir():
    _sigma_vals.append(('stratified bootstrap (nb12)', _strat_sigma))
if '_nb10_std' in dir() and _nb10_std:
    _sigma_vals.append(('random bootstrap nb10', _nb10_std))
if _sigma_vals:
    _best_sigma_name, _best_sigma = min(_sigma_vals, key=lambda t: t[1])
    print(f'  RECOMMENDED X      : {_best_sigma:.4f}° ({_best_sigma_name})')
print('=' * 70)

# Save summary JSON
_summary_data = {
    'image': str(FITS_PATH.name),
    'n_detected': int(len(fid_result.detected_x)),
    'n_matched': int(len(fid_result.matched_x)),
    'plate_scale_arcsec_px': _ps_arcsec,
    'fov_x_deg': _fov_x_deg, 'fov_y_deg': _fov_y_deg,
    'theta_north_fiducial_deg': fid_metrics.north_angle_deg,
    'theta_east_fiducial_deg': fid_metrics.east_angle_deg,
    'ne_orthogonality_departure_deg': _orth_dep,
    'strat_bootstrap_sigma_deg': _strat_sigma if '_strat_sigma' in dir() else None,
    'nb10_bootstrap_sigma_deg': _nb10_std if '_nb10_std' in dir() else None,
    'nb11_bootstrap_sigma_deg': _nb11_std if '_nb11_std' in dir() else None,
    'max_pix_to_sky_diff_deg': _max_diff if '_max_diff' in dir() else None,
}
(OUT_DIR / 'diagnostic_summary.json').write_text(
    _json.dumps(_summary_data, indent=2), encoding='utf-8'
)
print(f'Summary JSON saved to {OUT_DIR}/diagnostic_summary.json')
""")


# =============================================================================
# CELL 12 — CONCLUSION (markdown)
# =============================================================================
md("""\
## Cell 12 — Conclusion

*(Fill in after running all cells.  Template below.)*

### Source selection
- Did edge masking / circular cuts change the orientation estimate?
- Did limiting to high-SNR or low-residual matched sources improve stability?
- Which selection produced the most consistent WCS orientation?

### Spatial coverage
- Did the stratified 3×3 bootstrap produce smaller scatter than the
  previous random bootstrap (nb10/nb11)?
- If yes: the prior scatter was dominated by coverage gaps, not fundamental
  WCS uncertainty.  The stratified σ is the better estimate of X.
- If no: the scatter is intrinsic to the order-5 SIP fit.

### WCS model stability
- Were the residuals (Cell 4) coherent (systematic) or random?
  - Coherent residuals suggest the SIP model is biased.
  - Random residuals are consistent with photon noise / centroiding error.
- Was the affine cross-check (Cell 9) consistent with the WCS derivative?
  - Large discrepancy would indicate the high-order SIP is absorbing noise
    in a way that destabilises the local orientation.

### Pixel→sky conversion
- Is the simple-subtraction approximation adequate for the spectra angle pipeline?
  - If max |full − simple| < 0.01°: simple subtraction is fine.
  - Otherwise: use `pixel_angle_to_sky_angle()` per trace.

### Recommended value of X for the paper

| Method | σ (°) |
|--------|-------|
| nb10 random bootstrap (80%, 255 srcs) | 0.0923 |
| nb11 random bootstrap (80%, 86 matched) | 0.1276 |
| nb12 stratified 3×3 bootstrap | *fill in* |

**Current best estimate: X ≈ `____` °** *(fill in after running Cell 7)*

### Further improvements
1. Increase `N_BOOT_STRAT` to 200+ for a stable σ estimate.
2. If the affine cross-check reveals large discrepancies, consider order-3
   or order-4 SIP to reduce over-fitting.
3. Verify that the circular field mask (R_img) is accurately capturing the
   illuminated region; tight masking may improve SIP stability.
4. If source count dependence (Cell 6) shows a plateau, use that minimum N
   as the submission limit to reduce wasted computation.
""")


# =============================================================================
# ASSEMBLE AND WRITE
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
print(f"Cells: {len(cells)}")
