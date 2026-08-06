__version__ = "0.3.0"
__description__ = "Diffraction/spectral trace orientation angle extraction from plate-solved astronomical images."

# Pipeline overview
# -----------------
# Phase 1 (extractor/stars.py, platesolve.py):
#   stars.extract_stars()    → detect image point sources
#   platesolve.platesolve()  → broad astrometry.net field solution, save products to working/
#
# Phase 2 (planned): custom distortion model built from Phase 1 products
# Phase 3 (extractor/detection.py, fitting.py, combine.py, pipeline.py):
#   detection.detect_traces()      → candidate elongated trace segments
#   fitting.measure_trace() /
#     fitting.remove_contamination() → per-trace PCA fit + star-contamination removal
#   combine.combine_traces()       → robust ensemble pixel-space angle + uncertainty
#   pipeline.measure_grating_angle() → end-to-end entry point, with optional
#                                        sky-frame conversion via wcsangle.py
#
# sigma_WCS (the other half of the sky-frame error budget) is not yet
# reusable code in this package -- see pipeline.measure_grating_angle's
# docstring.

from .stars import extract_stars, make_xylist
from .platesolve import (
    platesolve, platesolve_xylist, PlatesolveResult, has_sip, wcs_summary,
    create_session,
    solve_plate, solve_plate_local, solve_plate_remote,
    PlateSolveError, LocalSolveFieldError,
)
from .wcsangle import (
    angle_diff_deg,
    local_north_angle_deg,
    local_east_angle_deg,
    center_wcs_angle_metrics,
    WcsAngleMetrics,
    pixel_angle_to_sky_angle,
    local_wcs_jacobian,
)
from .detection import detect_traces, sweep_min_length, TraceCandidate
from .fitting import (
    measure_trace, remove_contamination, trace_correlation_length_px,
    analytical_sigma_theta_deg, pca_fit, norm_axial, axial_diff, TraceMeasurement,
)
from .combine import combine_traces, CombinedAngleResult
from .pipeline import measure_grating_angle, AngleExtractionResult
from .wcs_uncertainty import (
    bootstrap_wcs_orientation, stratified_bootstrap_indices, WcsBootstrapResult,
    run_one_bootstrap_draw, robust_bootstrap_summary,
)
