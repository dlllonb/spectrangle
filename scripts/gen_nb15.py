#!/usr/bin/env python3
"""scripts/gen_nb15.py — generate notebooks/15_sim_local_platesolve_bootstrap.ipynb"""
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / '15_sim_local_platesolve_bootstrap.ipynb'

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
# 15 — Simulated image: local solve-field + WCS orientation bootstrap

**Purpose:** Test whether local `solve-field` can solve the `tele-img-sim`
simulated 50 mm image (`110lmm50mm.fits`), and estimate the WCS center
north-angle precision via bootstrap resampling.

**Why this matters:**
- The real 6 mm test image (notebooks 10/11) gives σ(θ_north) ≈ 0.09–0.13°.
- This simulated image has known ground-truth geometry — plate scale, field
  centre, and grating angle are all encoded in the FITS header (`MASKANG`,
  `RA0DEG`, `DEC0DEG`, `PSARC`).
- Local solver previously failed on this image, likely because no suitable
  index files were installed. New 4100-series index files appropriate for the
  ~9.9 arcsec/px / 50 mm scale have now been downloaded.
- If the simulated image solves cleanly and bootstraps to tight precision,
  it may be more useful for the concept paper than the non-optimised real
  6 mm test image — it demonstrates the method under known conditions.

**Kernel:** Run from `Python (spectrangle WSL)`.
**Backend:** Local `solve-field` only; no remote astrometry.net calls.

| Header parameter | Value |
|:---|:---|
| Focal length | 50 mm |
| Pixel size | 2.4 µm (ASI178MC-like) |
| Expected scale | ~9.9 arcsec/px |
| Expected FOV | ~8.5° × 5.7° (3096 × 2080 px) |
| Grating angle | `MASKANG` = 23° ← ground truth |
""")


# =============================================================================
# CELL 2 — ENVIRONMENT CHECK
# =============================================================================
md("## Cell 2 — Environment check\n")

code("""\
import sys, shutil, warnings, time, csv, json, platform, pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table

_pkg_root = str(Path('..').resolve())
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from extractor import extract_stars, make_xylist
from extractor.platesolve import (
    solve_plate, wcs_summary, LocalSolveFieldError, _col_array
)
from extractor.wcsangle import (
    angle_diff_deg, center_wcs_angle_metrics, WcsAngleMetrics
)

BACKEND_CFG = "/mnt/c/Users/bassd/.astrometry/backend.cfg"
INDEX_DIR   = Path("/mnt/c/Users/bassd/astrometry-data/4100/")

print(f"Python        : {sys.executable}")
print(f"Version       : {sys.version.split()[0]}")
print(f"CWD           : {Path('.').resolve()}")
print(f"Platform      : {platform.system()} {platform.release()}")
if platform.system() == "Windows":
    print("\\n*** WARNING: Windows kernel -- local solve-field will fail.")
    print("    Switch to the 'Python (spectrangle WSL)' Jupyter kernel.")

sf = shutil.which("solve-field")
print(f"\\nsolve-field   : {sf or 'NOT FOUND -- install astrometry.net in WSL'}")
print(f"backend cfg   : {BACKEND_CFG}")
print(f"  exists      : {Path(BACKEND_CFG).exists()}")
print(f"index dir     : {INDEX_DIR}")
print(f"  exists      : {INDEX_DIR.exists()}")
if INDEX_DIR.exists():
    idx_files = sorted(INDEX_DIR.glob("index-41*.fits"))
    print(f"  {len(idx_files)} index file(s) found:")
    for p in idx_files:
        print(f"    {p.name}")
    # Highlight files most relevant for ~10 arcsec/px (4108-4115 range)
    rel = [p for p in idx_files if any(
        p.name.startswith(f"index-41{d:02d}") for d in range(8, 16)
    )]
    if rel:
        print("  Likely useful for ~10 arcsec/px (index-4108 to 4115):")
        for p in rel:
            print(f"    {p.name}")
    else:
        print("  NOTE: no index-4108 through 4115 found.")
        print("  These are likely needed for the 50 mm / ~10 arcsec/px scale.")
        print("  Try: index-4115 through index-4119 if the FOV is large enough.")

import extractor
print(f"\\nextractor     : {extractor.__version__}  ({extractor.__file__})")

plt.rcParams.update({
    'figure.facecolor': '#111', 'axes.facecolor': '#111',
    'text.color': 'white',      'axes.labelcolor': 'white',
    'xtick.color': '#ccc',      'ytick.color': '#ccc',
    'axes.edgecolor': '#555',   'axes.titlecolor': 'white',
    'legend.facecolor': '#1e1e1e', 'legend.edgecolor': '#555',
    'grid.color': '#333',       'grid.alpha': 0.4,
    'font.size': 10,            'axes.titlesize': 11,
    'savefig.facecolor': '#111', 'savefig.dpi': 150,
})
""")


# =============================================================================
# CELL 3 — PARAMETERS
# =============================================================================
md("## Cell 3 — Parameters\n\nAll tunable parameters in one place.\n")

code("""\
# ── Image path ────────────────────────────────────────────────────────────
# Search the usual locations; fail with a clear message if not found.
def _find_fits(name):
    candidates = [
        Path('..') / 'data'      / name,
        Path('..') / name,
        Path('..') / 'notebooks' / name,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None

FITS_PATH = _find_fits('110lmm50mm.fits')
if FITS_PATH is None:
    raise FileNotFoundError(
        "110lmm50mm.fits not found.\\n"
        "Expected location: spectrangle/data/110lmm50mm.fits\\n"
        "Please copy the file there and re-run this cell."
    )

# ── Solver parameters ─────────────────────────────────────────────────────
BACKEND_CFG_STR = "/mnt/c/Users/bassd/.astrometry/backend.cfg"
OUT_DIR         = Path('..') / 'out' / 'local_bootstrap_15'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_UNITS   = "arcsecperpix"
SCALE_LOW     = 8.0     # arcsec/px  (50 mm lens, 2.4 um px -> ~9.9"/px)
SCALE_HIGH    = 12.0
TWEAK_ORDER   = 5       # SIP polynomial order; change to 3 or 4 to compare
SOLVE_TIMEOUT = 180     # seconds

# ── Bootstrap parameters ──────────────────────────────────────────────────
N_BOOT    = 50
BOOT_FRAC = 0.80
RNG_SEED  = 42

RESULTS_CSV = OUT_DIR / 'bootstrap_results.csv'

print("=== Notebook parameters ===")
print(f"  Image        : {FITS_PATH}")
print(f"  Output dir   : {OUT_DIR.resolve()}")
print(f"  Backend cfg  : {BACKEND_CFG_STR}")
print(f"  Scale        : {SCALE_LOW}--{SCALE_HIGH} {SCALE_UNITS}")
print(f"  Tweak order  : {TWEAK_ORDER}")
print(f"  Timeout      : {SOLVE_TIMEOUT} s")
print(f"  N_BOOT       : {N_BOOT}")
print(f"  Boot frac    : {BOOT_FRAC:.0%}")
print(f"  RNG seed     : {RNG_SEED}")
""")


# =============================================================================
# CELL 4 — LOAD AND DISPLAY
# =============================================================================
md("## Cell 4 — Load image and display\n")

code("""\
with fits.open(FITS_PATH) as hdul:
    image       = hdul[0].data.astype(np.float32)
    orig_header = hdul[0].header.copy()

ny, nx = image.shape

GT_RA_DEG    = float(orig_header.get('RA0DEG',  float('nan')))
GT_DEC_DEG   = float(orig_header.get('DEC0DEG', float('nan')))
GT_SCALE_ARC = float(orig_header.get('PSARC',   float('nan')))
GT_MASK_ANG  = float(orig_header.get('MASKANG', float('nan')))
GT_ROT_DEG   = float(orig_header.get('ROTDEG',  float('nan')))
GT_FOV_X     = float(orig_header.get('FOVX',    float('nan')))
GT_FOV_Y     = float(orig_header.get('FOVY',    float('nan')))

print(f"Image       : {FITS_PATH.name}  ({nx} x {ny} px)")
print(f"Origin      : {orig_header.get('ORIGIN', '?')}")
print(f"Centre      : RA {GT_RA_DEG:.5f} deg  Dec {GT_DEC_DEG:.5f} deg")
print(f"Plate scale : {GT_SCALE_ARC:.3f} arcsec/px")
print(f"FOV         : {GT_FOV_X:.2f} x {GT_FOV_Y:.2f} deg")
print(f"Rotation    : {GT_ROT_DEG:.2f} deg")
print(f"Grating     : MASKANG = {GT_MASK_ANG:.2f} deg  <- ground-truth angle")
print()
print("Selected header cards:")
_shown = 0
for card in orig_header.cards:
    kw = (card.keyword or '').upper()
    if kw and kw not in ('SIMPLE', 'EXTEND', 'END', ''):
        print(f"  {kw:<10} = {str(card.value):<20}  {card.comment[:45]}")
        _shown += 1
        if _shown >= 30:
            break

# ── arcsinh stretch (background subtracted) ───────────────────────────────
_bg   = gaussian_filter(image, sigma=50)
_proc = np.clip(image - _bg, 0, None)
_lo, _hi = np.percentile(_proc[np.isfinite(_proc)], [0.5, 99.5])
disp  = np.arcsinh(np.clip(_proc, _lo, _hi))
_dlo  = float(np.arcsinh(_lo))
_dhi  = float(np.arcsinh(_hi))

fig, ax = plt.subplots(figsize=(14, 9.5))
ax.imshow(disp, origin='lower', cmap='gray_r', vmin=_dlo, vmax=_dhi,
          aspect='equal', interpolation='nearest')
ax.set_title(
    f"{FITS_PATH.name}  ({nx}x{ny})  arcsinh stretch\\n"
    f"RA={GT_RA_DEG:.3f}  Dec={GT_DEC_DEG:.3f}  "
    f"scale={GT_SCALE_ARC:.2f} arcsec/px  MASKANG={GT_MASK_ANG:.1f} deg",
    fontsize=10)
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
fig.tight_layout()
_p = OUT_DIR / '01_image.png'
fig.savefig(_p, bbox_inches='tight')
plt.show()
print(f"Saved {_p}")
""")


# =============================================================================
# CELL 5 — SOURCE DETECTION
# =============================================================================
md("## Cell 5 — Source detection\n")

code("""\
# ── Cached extraction ─────────────────────────────────────────────────────
_src_cache = OUT_DIR / 'sources.pkl'
if _src_cache.exists():
    with open(_src_cache, 'rb') as fh:
        all_xs, all_ys, all_fluxes = pickle.load(fh)
    print(f"Loaded {len(all_xs)} sources from cache.")
else:
    print("Running source extraction...")
    all_xs, all_ys, all_fluxes = extract_stars(
        image, max_sources=300, mask_spectra=True
    )
    with open(_src_cache, 'wb') as fh:
        pickle.dump((all_xs, all_ys, all_fluxes), fh)
    print(f"Detected and cached {len(all_xs)} sources.")

selected_xs, selected_ys = all_xs, all_ys
print(f"Selected sources for plate-solve: {len(selected_xs)}")

# ── Flux distribution quick-look ─────────────────────────────────────────
print(f"Flux range  : {all_fluxes.min():.1f} -- {all_fluxes.max():.1f}")
print(f"Flux median : {np.median(all_fluxes):.1f}")

# ── Overlay plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9.5))
ax.imshow(disp, origin='lower', cmap='gray_r', vmin=_dlo, vmax=_dhi,
          aspect='equal', interpolation='nearest')
ax.scatter(selected_xs, selected_ys,
           s=20, facecolors='none', edgecolors='#00e5ff', linewidths=0.9,
           alpha=0.8, label=f'detected ({len(selected_xs)})')
ax.set_title(f"Source detection -- {FITS_PATH.name}  (N={len(selected_xs)})",
             fontsize=11)
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
ax.legend(fontsize=9)
fig.tight_layout()
_p = OUT_DIR / '02_sources.png'
fig.savefig(_p, bbox_inches='tight')
plt.show()
print(f"Saved {_p}")
""")


# =============================================================================
# CELL 6 — FIDUCIAL LOCAL ORDER-5 SOLVE
# =============================================================================
md("""\
## Cell 6 — Fiducial local order-5 solve

Run once with the full source list.  Result is cached so re-running this cell
is instant after the first solve.
""")

code("""\
FID_DIR   = OUT_DIR / 'fiducial'
FID_CACHE = OUT_DIR / 'fiducial_result.pkl'

def _make_wcs(header):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        return WCS(header)

def _read_match_meta(output_dir):
    p = Path(output_dir) / 'xylist.match'
    if not p.exists():
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with fits.open(str(p)) as h:
                t = Table(h[1].data)
                return dict(
                    indexid=int(t['INDEXID'][0]),
                    nmatch=int(t['NMATCH'][0]),
                    logodds=float(t['LOGODDS'][0]),
                )
    except Exception:
        return {}

if FID_CACHE.exists():
    print("Loading fiducial solve from cache...")
    with open(FID_CACHE, 'rb') as fh:
        fid_result = pickle.load(fh)
    print("OK.")
else:
    print(f"Running fiducial solve  ({len(selected_xs)} sources, order {TWEAK_ORDER}) ...")
    _t0 = time.time()
    try:
        fid_result = solve_plate(
            xs=selected_xs, ys=selected_ys,
            image_width=nx, image_height=ny,
            original_header=orig_header,
            backend='local',
            backend_config=BACKEND_CFG_STR,
            scale_units=SCALE_UNITS,
            scale_low=SCALE_LOW,
            scale_high=SCALE_HIGH,
            tweak_order=TWEAK_ORDER,
            output_dir=FID_DIR,
            timeout=SOLVE_TIMEOUT,
            verbose=True,
        )
        print(f"Fiducial solve finished in {time.time()-_t0:.1f} s")
    except LocalSolveFieldError as _e:
        print(f"\\n*** SOLVE FAILED ***\\n{_e}")
        fid_result = None

    if fid_result is None:
        print("\\nNo solution found.  Possible reasons:")
        print(f"  - No suitable index files in {INDEX_DIR}")
        print(f"    Need index-4108 to 4119 for ~{SCALE_LOW}-{SCALE_HIGH} arcsec/px")
        print("  - Scale bounds too tight (try 7--14 arcsec/px)")
        print("  - Too few sources detected")
    else:
        with open(FID_CACHE, 'wb') as fh:
            pickle.dump(fid_result, fh)

if fid_result is None:
    raise RuntimeError("Fiducial solve failed -- cannot continue.  "
                       "See output above for diagnostics.")

# ── WCS metrics ───────────────────────────────────────────────────────────
fid_wcs     = _make_wcs(fid_result.header)
fid_metrics = center_wcs_angle_metrics(fid_wcs, image.shape, compute_east=True)

_sc = proj_plane_pixel_scales(fid_wcs)
_ps_arcsec = float(np.mean(np.abs(_sc)) * 3600)
_fov_x_deg = float(np.abs(_sc[0]) * nx)
_fov_y_deg = float(np.abs(_sc[1]) * ny)
_mmeta     = _read_match_meta(FID_DIR)

_dne = angle_diff_deg(fid_metrics.east_angle_deg,
                      fid_metrics.north_angle_deg + 90.0)

print()
print("=== Fiducial WCS ===")
print(f"  Sources submitted : {len(fid_result.detected_x)}")
print(f"  Sources matched   : {len(fid_result.matched_x)}")
print(f"  Index INDEXID     : {_mmeta.get('indexid', 'n/a')}")
print(f"  Match logodds     : {_mmeta.get('logodds', 0.0):.2f}")
print(f"  Plate scale       : {_ps_arcsec:.3f} arcsec/px  (GT: {GT_SCALE_ARC:.3f})")
print(f"  dScale            : {_ps_arcsec - GT_SCALE_ARC:+.3f} arcsec/px")
print(f"  FOV               : {_fov_x_deg:.3f} x {_fov_y_deg:.3f} deg")
print(f"                       (GT: {GT_FOV_X:.3f} x {GT_FOV_Y:.3f} deg)")
print(f"  WCS file          : {FID_DIR / 'xylist.wcs'}")
print()
print(f"  Center pixel      : ({fid_metrics.x_center:.1f}, {fid_metrics.y_center:.1f})")
print(f"  Center RA         : {fid_metrics.ra_deg:.6f} deg  (GT: {GT_RA_DEG:.6f})")
print(f"  Center Dec        : {fid_metrics.dec_deg:.6f} deg  (GT: {GT_DEC_DEG:.6f})")
_dra  = angle_diff_deg(fid_metrics.ra_deg, GT_RA_DEG) * 3600
_ddec = (fid_metrics.dec_deg - GT_DEC_DEG) * 3600
print(f"  dRA  (WCS-GT)     : {_dra:+.2f} arcsec")
print(f"  dDec (WCS-GT)     : {_ddec:+.2f} arcsec")
print()
print(f"  theta_north       : {fid_metrics.north_angle_deg:.5f} deg  (CCW from +x)")
print(f"  theta_east        : {fid_metrics.east_angle_deg:.5f} deg")
print(f"  N/E departure     : {_dne:.4f} deg  (ideal = 0.000)")
print()
print(wcs_summary(fid_result.header))
""")


# =============================================================================
# CELL 7 — WCS OVERLAY
# =============================================================================
md("""\
## Cell 7 — WCS overlay and solve quality

Check that the WCS grid follows the image structure across the full FOV.
If only one side is well solved, the grid lines will diverge from reality
near the edges — this is visible at high tweak orders if the model overfits.
""")

code("""\
from astropy.visualization.wcsaxes import WCSAxes  # noqa

fig = plt.figure(figsize=(14, 9.5))
ax  = fig.add_subplot(111, projection=fid_wcs)

ax.imshow(disp, origin='lower', cmap='gray_r',
          vmin=_dlo, vmax=_dhi, aspect='equal', interpolation='nearest')

ax.coords.grid(True, color='cyan', alpha=0.30, linestyle='--', linewidth=0.8)
ax.coords['ra'].set_major_formatter('hh:mm')
ax.coords['dec'].set_major_formatter('dd:mm')
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')

# Detected (submitted) sources
ax.scatter(fid_result.detected_x, fid_result.detected_y,
           transform=ax.get_transform('pixel'),
           s=18, facecolors='none', edgecolors='lime', linewidths=0.8,
           alpha=0.55, label=f'detected ({len(fid_result.detected_x)})')

# Matched (catalog) sources
if fid_result.corr_table is not None:
    _fx = _col_array(fid_result.corr_table, 'field_x')
    _fy = _col_array(fid_result.corr_table, 'field_y')
    if len(_fx):
        ax.scatter(_fx, _fy,
                   transform=ax.get_transform('pixel'),
                   s=60, facecolors='none', edgecolors='orange',
                   linewidths=1.3, alpha=0.85,
                   label=f'matched ({len(_fx)})')

ax.legend(fontsize=9, loc='upper right')

_ann = (f"Order {TWEAK_ORDER} SIP  |  {_ps_arcsec:.2f} arcsec/px  "
        f"(GT {GT_SCALE_ARC:.2f} arcsec/px)\\n"
        f"FOV {_fov_x_deg:.2f} x {_fov_y_deg:.2f} deg  |  "
        f"INDEXID {_mmeta.get('indexid','?')}\\n"
        f"Center RA {fid_metrics.ra_deg:.4f}  Dec {fid_metrics.dec_deg:.4f}\\n"
        f"theta_north = {fid_metrics.north_angle_deg:.4f} deg\\n"
        f"N/E departure = {_dne:.4f} deg")
ax.text(0.01, 0.01, _ann, transform=ax.transAxes,
        fontsize=8.5, color='white', verticalalignment='bottom',
        bbox=dict(facecolor='#111', alpha=0.75, edgecolor='#555',
                  boxstyle='round,pad=0.4'))

ax.set_title(
    f'Fiducial WCS overlay -- {FITS_PATH.name}  '
    f'(det {len(fid_result.detected_x)}, match {len(fid_result.matched_x)})',
    fontsize=11)
fig.tight_layout()
_p = OUT_DIR / '03_wcs_overlay.png'
fig.savefig(_p, bbox_inches='tight')
plt.show()
print(f"Saved {_p}")
""")


# =============================================================================
# CELL 8 — PARAMETER TEST (TWEAK ORDERS)
# =============================================================================
md("""\
## Cell 8 — Parameter test: tweak order comparison

Try orders 3, 4, and 5 to confirm that order 5 is appropriate for this image.
Results are cached under `out/local_bootstrap_15/param_test/`.
""")

code("""\
_test_orders = [3, 4, 5]
# To also test scale bounds, uncomment and extend _test_scales:
_test_scales = [(SCALE_LOW, SCALE_HIGH)]
# _test_scales = [(7.0, 13.0), (8.0, 12.0), (9.0, 11.0)]

_param_rows = []
_param_test_dir = OUT_DIR / 'param_test'
_param_test_dir.mkdir(parents=True, exist_ok=True)

for _order in _test_orders:
    for _slo, _shi in _test_scales:
        _tag    = f"order{_order}_s{int(_slo)}-{int(_shi)}"
        _pdir   = _param_test_dir / _tag
        _pcache = _param_test_dir / f"{_tag}.pkl"

        if _pcache.exists():
            with open(_pcache, 'rb') as fh:
                _r = pickle.load(fh)
            _elapsed = None
        else:
            _t0 = time.time()
            try:
                _r = solve_plate(
                    xs=selected_xs, ys=selected_ys,
                    image_width=nx, image_height=ny,
                    original_header=orig_header,
                    backend='local',
                    backend_config=BACKEND_CFG_STR,
                    scale_units=SCALE_UNITS,
                    scale_low=float(_slo),
                    scale_high=float(_shi),
                    tweak_order=_order,
                    output_dir=_pdir,
                    timeout=SOLVE_TIMEOUT,
                    verbose=False,
                )
            except Exception:
                _r = None
            _elapsed = time.time() - _t0
            if _r is not None:
                with open(_pcache, 'wb') as fh:
                    pickle.dump(_r, fh)

        if _r is not None:
            _w  = _make_wcs(_r.header)
            _m  = center_wcs_angle_metrics(_w, image.shape, compute_east=False)
            _mm = _read_match_meta(_pdir)
            _sc2 = proj_plane_pixel_scales(_w)
            _ps2 = float(np.mean(np.abs(_sc2)) * 3600)
            _param_rows.append({
                'order': _order, 'scale': f"{_slo}-{_shi}",
                'status': 'ok',
                'n_matched': len(_r.matched_x),
                'indexid': _mm.get('indexid', '?'),
                'plate_scale': round(_ps2, 2),
                'theta_north': round(_m.north_angle_deg, 4),
                'elapsed_s': round(_elapsed, 1) if _elapsed is not None else 'cached',
            })
            print(f"order={_order} scale={_slo}-{_shi}  "
                  f"match={len(_r.matched_x)}  ps={_ps2:.2f} arcsec/px  "
                  f"theta={_m.north_angle_deg:.4f} deg"
                  + (f"  [{_elapsed:.0f} s]" if _elapsed is not None else " (cached)"))
        else:
            _param_rows.append({
                'order': _order, 'scale': f"{_slo}-{_shi}",
                'status': 'FAILED', 'n_matched': 0,
                'indexid': '', 'plate_scale': float('nan'),
                'theta_north': float('nan'),
                'elapsed_s': round(_elapsed, 1) if _elapsed is not None else 'n/a',
            })
            print(f"order={_order} scale={_slo}-{_shi}  FAILED"
                  + (f"  [{_elapsed:.0f} s]" if _elapsed is not None else ""))

print()
print(f"{'order':<7} {'scale':<10} {'status':<8} {'n_match':<9} "
      f"{'indexid':<9} {'ps (arcsec/px)':<16} {'theta_north (deg)':<20} elapsed")
print("-" * 90)
for r in _param_rows:
    print(f"{r['order']:<7} {r['scale']:<10} {r['status']:<8} {r['n_matched']:<9} "
          f"{str(r['indexid']):<9} {str(r['plate_scale']):<16} "
          f"{str(r['theta_north']):<20} {r['elapsed_s']}")
""")


# =============================================================================
# CELL 9 — BOOTSTRAP LOOP
# =============================================================================
md("""\
## Cell 9 — Bootstrap loop

Re-solve on `N_BOOT` random `BOOT_FRAC`-subsets of the detected sources.
Results are written to CSV after each iteration so progress survives a
kernel restart.  Re-running this cell skips already-completed iterations.
""")

code("""\
BOOT_OUT = OUT_DIR / 'boot_solves'
BOOT_OUT.mkdir(parents=True, exist_ok=True)

# Optionally restrict to the index used by the fiducial solve.
# This avoids re-testing all indices each bootstrap iteration.
_fid_indexid = _mmeta.get('indexid', None)
if _fid_indexid is not None:
    _idx_path = INDEX_DIR / f"index-{_fid_indexid}.fits"
    if _idx_path.exists():
        INDEX_RESTRICT = str(_idx_path)
        print(f"Index restriction : {INDEX_RESTRICT}  (fiducial INDEXID={_fid_indexid})")
    else:
        INDEX_RESTRICT = None
        print(f"Index file {_idx_path.name} not found in INDEX_DIR; "
              "using scale bounds only.")
else:
    INDEX_RESTRICT = None
    print("No fiducial INDEXID found; using scale bounds only.")

n_src_total = len(selected_xs)
n_boot_src  = int(n_src_total * BOOT_FRAC)
rng         = np.random.default_rng(RNG_SEED)
boot_indices = [
    rng.choice(n_src_total, size=n_boot_src, replace=False)
    for _ in range(N_BOOT)
]
_extra = ['--index', INDEX_RESTRICT] if INDEX_RESTRICT else []

print(f"\\nBootstrap config")
print(f"  N_BOOT        = {N_BOOT}")
print(f"  BOOT_FRAC     = {BOOT_FRAC:.0%}  ({n_boot_src} sources per sample)")
print(f"  TWEAK_ORDER   = {TWEAK_ORDER}")
print(f"  Scale         = {SCALE_LOW}--{SCALE_HIGH} {SCALE_UNITS}")
print(f"  SOLVE_TIMEOUT = {SOLVE_TIMEOUT} s")
print(f"  Results CSV   = {RESULTS_CSV}")
print(f"  Solve dirs    = {BOOT_OUT}/iter_NNN/")

_CSV_HDR = ['boot_idx', 'n_src', 'north_angle_deg', 'ra_deg', 'dec_deg',
            'elapsed_s', 'status']

_done = {}
if RESULTS_CSV.exists():
    with open(RESULTS_CSV, newline='') as _f:
        for _row in csv.DictReader(_f):
            if _row['status'] == 'ok':
                _done[int(_row['boot_idx'])] = _row
    print(f"\\nResuming: {len(_done)} successful results already saved.")
else:
    with open(RESULTS_CSV, 'w', newline='') as _f:
        csv.writer(_f).writerow(_CSV_HDR)
    print("\\nNew CSV created.")

t_run_start = time.time()

for i, idx in enumerate(boot_indices):
    if i in _done:
        continue

    sub_xs   = selected_xs[idx]
    sub_ys   = selected_ys[idx]
    iter_dir = BOOT_OUT / f'iter_{i:03d}'

    _wcs_exists = (iter_dir / 'xylist.wcs').exists()
    t0     = time.time()
    status = 'ok'
    m      = None

    if _wcs_exists:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with fits.open(str(iter_dir / 'xylist.wcs')) as _h:
                    _hdr = _h[0].header.copy()
            _wcs = _make_wcs(_hdr)
            m    = center_wcs_angle_metrics(_wcs, image.shape, compute_east=False)
        except Exception as _e:
            status = f'wcs_read_error: {str(_e)[:80]}'
    else:
        try:
            _r = solve_plate(
                xs=sub_xs, ys=sub_ys,
                image_width=nx, image_height=ny,
                original_header=orig_header,
                backend='local',
                backend_config=BACKEND_CFG_STR,
                scale_units=SCALE_UNITS,
                scale_low=SCALE_LOW,
                scale_high=SCALE_HIGH,
                tweak_order=TWEAK_ORDER,
                output_dir=iter_dir,
                timeout=SOLVE_TIMEOUT,
                verbose=False,
                extra_args=_extra if _extra else None,
            )
            if _r is None:
                status = 'no_solution'
            else:
                _wcs = _make_wcs(_r.header)
                m    = center_wcs_angle_metrics(_wcs, image.shape,
                                                compute_east=False)
        except LocalSolveFieldError as _e:
            status = f'solver_error: {str(_e)[:80]}'
        except Exception as _e:
            status = f'error: {type(_e).__name__}: {str(_e)[:80]}'

    elapsed = time.time() - t0

    if m is not None:
        _dth = angle_diff_deg(m.north_angle_deg, fid_metrics.north_angle_deg)
        _tag = (f"theta={m.north_angle_deg:.4f}  "
                f"delta={_dth:+.4f}  [{elapsed:.0f} s]")
    else:
        _tag = f"FAILED  [{status[:55]}  {elapsed:.0f} s]"
    print(f"{i:03d}/{N_BOOT}  n={len(sub_xs)}  {_tag}")

    _row = [
        i, len(sub_xs),
        m.north_angle_deg if m else '',
        m.ra_deg          if m else '',
        m.dec_deg         if m else '',
        f'{elapsed:.2f}',
        status,
    ]
    with open(RESULTS_CSV, 'a', newline='') as _f:
        csv.writer(_f).writerow(_row)

print(f"\\nBootstrap loop finished in {(time.time()-t_run_start)/60:.1f} min.")
""")


# =============================================================================
# CELL 10 — BOOTSTRAP SUMMARY
# =============================================================================
md("## Cell 10 — Bootstrap statistics\n")

code("""\
_ok_rows   = []
_fail_rows = []
with open(RESULTS_CSV, newline='') as _f:
    for _row in csv.DictReader(_f):
        if _row['status'] == 'ok' and _row['north_angle_deg']:
            _ok_rows.append(_row)
        else:
            _fail_rows.append(_row)

n_ok     = len(_ok_rows)
n_failed = len(_fail_rows)

if n_ok < 2:
    raise RuntimeError(
        f"Only {n_ok} successful bootstrap solves -- need at least 2.\\n"
        "Check that solve-field is working and re-run Cell 9."
    )

boot_north = np.array([float(r['north_angle_deg']) for r in _ok_rows])
boot_ra    = np.array([float(r['ra_deg'])          for r in _ok_rows])
boot_dec   = np.array([float(r['dec_deg'])         for r in _ok_rows])
boot_times = np.array([float(r['elapsed_s'])       for r in _ok_rows])

fid_theta  = fid_metrics.north_angle_deg
fid_ra     = fid_metrics.ra_deg
fid_dec    = fid_metrics.dec_deg

boot_dtheta  = angle_diff_deg(boot_north, fid_theta)
boot_dra_as  = (angle_diff_deg(boot_ra, fid_ra)
                * 3600 * np.cos(np.radians(fid_dec)))
boot_ddec_as = (boot_dec - fid_dec) * 3600
boot_sep_as  = np.hypot(boot_dra_as, boot_ddec_as)

boot_theta_std = float(boot_dtheta.std())
boot_theta_mad = float(np.median(np.abs(boot_dtheta - np.median(boot_dtheta))))
boot_sep_std   = float(boot_sep_as.std())
boot_ra_std    = float(boot_dra_as.std())
boot_dec_std   = float(boot_ddec_as.std())

# Reference value from real 6 mm image (notebook 10, N=50, 80%)
_nb10_std = 0.09228   # deg

_sep = '-' * 65
print(_sep)
print('  BOOTSTRAP STATISTICS  --  110lmm50mm.fits  (50 mm simulated)')
print(_sep)
print(f'  Attempted          : {N_BOOT}')
print(f'  Solved (ok)        : {n_ok}')
print(f'  Failed             : {n_failed}')
print(f'  Avg solve time     : {boot_times.mean():.1f} s  '
      f'(min {boot_times.min():.0f} s, max {boot_times.max():.0f} s)')
print()
print(f'  Fiducial theta_north : {fid_theta:.5f} deg')
print(f'  Bootstrap mean delta : {boot_dtheta.mean():+.5f} deg')
print(f'  Bootstrap sigma(theta): {boot_theta_std:.5f} deg   <- X (this image)')
print(f'  Bootstrap MAD(theta) : {boot_theta_mad:.5f} deg')
print(f'  theta range          : [{boot_north.min():.4f}, {boot_north.max():.4f}] deg')
print()
print(f'  sigma(sky sep)   : {boot_sep_std:.3f} arcsec')
print(f'  sigma(RA*cosDec) : {boot_ra_std:.3f} arcsec')
print(f'  sigma(Dec)       : {boot_dec_std:.3f} arcsec')
print()
print(f'  Comparison to real 6 mm image (nb10): sigma = {_nb10_std:.5f} deg')
print(f'  Ratio sim / real   : {boot_theta_std/_nb10_std:.2f}x')
if boot_theta_std < _nb10_std:
    print('  -> Simulated image gives TIGHTER orientation precision than real image.')
else:
    print('  -> Simulated image gives BROADER scatter than real image.')
print(_sep)

_summary = dict(
    image=str(FITS_PATH.name),
    n_boot=N_BOOT, n_ok=n_ok, n_failed=n_failed,
    tweak_order=TWEAK_ORDER, boot_frac=BOOT_FRAC,
    scale_low=SCALE_LOW, scale_high=SCALE_HIGH,
    fid_north_angle_deg=fid_theta, fid_ra_deg=fid_ra, fid_dec_deg=fid_dec,
    boot_theta_std_deg=boot_theta_std,
    boot_theta_mad_deg=boot_theta_mad,
    boot_theta_mean_offset_deg=float(boot_dtheta.mean()),
    boot_sep_std_arcsec=boot_sep_std,
    nb10_real_image_std_deg=_nb10_std,
)
_sp = OUT_DIR / 'bootstrap_summary.json'
_sp.write_text(json.dumps(_summary, indent=2), encoding='utf-8')
print(f"\\nSummary saved to {_sp}")
""")


# =============================================================================
# CELL 11 — BOOTSTRAP HISTOGRAM
# =============================================================================
md("## Cell 11 — Bootstrap histogram\n")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: angle-difference histogram ──────────────────────────────────────
ax = axes[0]
ax.hist(boot_dtheta, bins=min(15, n_ok),
        color='#4fc3f7', edgecolor='#222', alpha=0.85)
ax.axvline(0, color='yellow', lw=2, ls='--', label='fiducial (delta=0)')
ax.axvline(boot_dtheta.mean(), color='cyan', lw=1.5,
           label=f'boot mean {boot_dtheta.mean():+.4f} deg')
ax.axvspan(-boot_theta_std, boot_theta_std,
           alpha=0.15, color='cyan',
           label=f'+-1sigma = {boot_theta_std:.4f} deg')
ax.set_xlabel('delta theta_north  (bootstrap - fiducial,  deg)')
ax.set_ylabel('Count')
ax.set_title(f'Bootstrap scatter  (N={n_ok}, order {TWEAK_ORDER})\\n'
             f'sigma = {boot_theta_std:.4f} deg  |  '
             f'real 6mm (nb10): 0.0923 deg', fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ── Right: sky-position scatter ────────────────────────────────────────────
ax2 = axes[1]
sc  = ax2.scatter(boot_dra_as, boot_ddec_as,
                  c=boot_dtheta, cmap='plasma', s=35, alpha=0.85, zorder=3)
plt.colorbar(sc, ax=ax2, label='delta theta_north (deg)', shrink=0.85)
_t = np.linspace(0, 2*np.pi, 300)
for _rr, _lw in [(boot_sep_std, 1.2), (2*boot_sep_std, 0.6)]:
    ax2.plot(_rr*np.cos(_t), _rr*np.sin(_t), '--', color='cyan', lw=_lw,
             alpha=0.5)
ax2.axhline(0, color='#555', lw=0.7)
ax2.axvline(0, color='#555', lw=0.7)
ax2.scatter([0], [0], marker='+', color='yellow', s=150, zorder=5,
            label='fiducial')
ax2.set_xlabel('delta RA x cos(Dec)  (arcsec)')
ax2.set_ylabel('delta Dec  (arcsec)')
ax2.set_title('Sky scatter  (colour = delta theta_north)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle(
    f'WCS bootstrap -- 110lmm50mm.fits  '
    f'(N={n_ok}/{N_BOOT}, {int(BOOT_FRAC*100)}% subsets, order {TWEAK_ORDER})\\n'
    f'X = sigma(theta_north) = {boot_theta_std:.4f} deg',
    fontsize=11)
fig.tight_layout()
_p = OUT_DIR / 'bootstrap_histogram.png'
fig.savefig(_p, bbox_inches='tight')
plt.show()
print(f"Saved {_p}")
""")


# =============================================================================
# CELL 12 — FIT-BASED PRECISION ESTIMATE
# =============================================================================
md("""\
## Cell 12 — Fit-based orientation precision estimate

If the `.corr` table is available, compute astrometric residuals and estimate
the orientation precision from the formal covariance of a rigid-rotation fit:

$$\\sigma_\\theta \\approx \\frac{\\sigma_\\mathrm{ast}}{\\sqrt{\\sum_i r_i^2}}$$

where σ_ast is the per-star astrometric residual RMS (pixels) and r_i is each
matched star's distance from the image centre (pixels).  Compare this
analytical estimate to the empirical bootstrap scatter.
""")

code("""\
if fid_result.corr_table is None:
    print("No corr table available -- skipping fit-based estimate.")
    print("(corr table requires a successful local solve.)")
else:
    _ct = fid_result.corr_table
    print(f"Corr table columns: {_ct.colnames}")

    _fx  = _col_array(_ct, 'field_x')
    _fy  = _col_array(_ct, 'field_y')
    _ira = _col_array(_ct, 'index_ra')
    _ide = _col_array(_ct, 'index_dec')

    if len(_fx) < 3 or len(_ira) == 0:
        print(f"Insufficient corr data (field_x: {len(_fx)}, index_ra: {len(_ira)}).")
        print("TODO: inspect column names above and adjust _col_array calls.")
    else:
        # Predicted pixel positions from the catalog catalog RA/Dec via WCS
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _pred_x, _pred_y = fid_wcs.world_to_pixel_values(_ira, _ide)

        _res_x  = _fx - _pred_x
        _res_y  = _fy - _pred_y
        _res_px = np.hypot(_res_x, _res_y)
        _rms_px = float(np.sqrt(np.mean(_res_px**2)))
        _rms_as = _rms_px * _ps_arcsec

        xc = (nx - 1) / 2.0
        yc = (ny - 1) / 2.0
        _r_from_cen = np.hypot(_fx - xc, _fy - yc)
        _sum_r2     = float(np.sum(_r_from_cen**2))
        _r_rms_px   = float(np.sqrt(np.mean(_r_from_cen**2)))

        # Formal orientation precision (Cramer-Rao for rigid rotation)
        _sigma_theta_rad = _rms_px / np.sqrt(_sum_r2)
        _sigma_theta_fit = float(np.degrees(_sigma_theta_rad))
        _N_match = len(_fx)

        print("=== Fit-based orientation precision ===")
        print(f"  Matched sources   : {_N_match}")
        print(f"  RMS residual      : {_rms_px:.4f} px = {_rms_as:.4f} arcsec")
        print(f"  Residual range    : {_res_px.min():.3f} -- {_res_px.max():.3f} px")
        print(f"  RMS star radius   : {_r_rms_px:.1f} px from image center")
        print(f"  sigma_theta (fit) : {_sigma_theta_fit:.5f} deg")
        print(f"  sigma_theta (boot): {boot_theta_std:.5f} deg")
        ratio = boot_theta_std / _sigma_theta_fit if _sigma_theta_fit > 0 else float('nan')
        print(f"  Ratio (boot/fit)  : {ratio:.2f}x")
        print()
        if ratio > 3:
            print("  NOTE: Bootstrap >> formal estimate.")
            print("  Bootstrap scatter is driven by source-selection / model")
            print("  instability, not astrometric noise alone.")
        elif ratio < 0.5:
            print("  NOTE: Bootstrap << formal estimate.")
            print("  Bootstrap samples may be correlated, or the WCS model")
            print("  is over-constrained.")
        else:
            print("  Bootstrap and formal estimates are broadly consistent.")

        # ── Residual diagnostic plots ──────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        sc = ax.scatter(_fx - xc, _fy - yc, c=_res_px, cmap='hot_r',
                        s=20, vmin=0, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='residual (px)')
        ax.set_xlabel('x - xc  (px)'); ax.set_ylabel('y - yc  (px)')
        ax.set_title(f'WCS residuals  (RMS = {_rms_px:.3f} px = {_rms_as:.3f}")')
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.hist(_res_px, bins=min(20, _N_match),
                 color='#ff7043', edgecolor='#111', alpha=0.85)
        ax2.axvline(_rms_px, color='yellow', lw=1.5,
                    label=f'RMS = {_rms_px:.3f} px')
        ax2.set_xlabel('residual (px)'); ax2.set_ylabel('count')
        ax2.set_title(f'Residual histogram  (N={_N_match})')
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

        fig.suptitle(
            f'Astrometric residuals -- order {TWEAK_ORDER} SIP\\n'
            f'sigma_theta (fit) = {_sigma_theta_fit:.4f} deg  |  '
            f'sigma_theta (boot) = {boot_theta_std:.4f} deg',
            fontsize=11)
        fig.tight_layout()
        _p = OUT_DIR / '05_residuals.png'
        fig.savefig(_p, bbox_inches='tight')
        plt.show()
        print(f"Saved {_p}")
""")


# =============================================================================
# CELL 13 — FINAL SUMMARY
# =============================================================================
md("""\
## Cell 13 — Summary and conclusions

*(Fill in after running the notebook.)*

---

**1. Did local solve-field solve `110lmm50mm.fits`?**

> *(yes / no — record which index file was used and the logodds)*

---

**2. Which index file was used?  Were the new index files necessary?**

> *(e.g., index-4115 — yes, the old 4117–4119 files were too coarse for
> the 8–12 arcsec/px scale of the 50 mm image)*

---

**3. Did fifth-order SIP improve the solve quality vs orders 3 and 4?**

> *(compare matched count and theta_north agreement from Cell 8 results)*

---

**4. WCS overlay quality — is the full image well solved?**

> If only one side of the image is well solved, the WCS grid lines will
> visibly diverge from reality near the edges.  Check the overlay from Cell 7
> and the residual map from Cell 12.

---

**5. Bootstrap precision σ(θ_north)?**

> **X_sim = σ(θ_north) = ??? deg** (from Cell 10)
> Compare: real 6 mm image (nb10) X_real = 0.0923 deg

---

**6. Is this simulated image suitable for the concept paper?**

| Outcome | Interpretation |
|:---|:---|
| X_sim < X_real | Tighter precision — use as primary validation figure |
| X_sim ≈ X_real | Comparable — use both; simulated shows expected clean-case performance |
| X_sim > X_real | Investigate: index choice, source count, WCS order |

---

**Checklist:**
- [ ] Plate scale matches `PSARC` header value (~9.9 arcsec/px)
- [ ] Center RA/Dec within ~0.1° of `RA0DEG`/`DEC0DEG`
- [ ] All 50 bootstrap iterations solved (no catastrophic failures)
- [ ] Order-5 SIP gives clearly better residuals than orders 3 and 4
- [ ] WCS grid lines look correct across the full FOV in Cell 7
- [ ] Formal fit-based estimate (Cell 12) is broadly consistent with bootstrap
""")


# =============================================================================
# WRITE NOTEBOOK
# =============================================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (spectrangle WSL)",
            "language": "python",
            "name": "spectrangle-wsl",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Notebook written to: {NB_PATH}")
print(f"Cells: {len(cells)}  ({n_code} code, {n_md} markdown)")
