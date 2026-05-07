__version__ = "0.2.0"
__description__ = "Diffraction/spectral trace orientation angle extraction from plate-solved astronomical images."

# Pipeline overview
# -----------------
# Phase 1 (this package — extractor/):
#   stars.extract_stars()    → detect image point sources
#   platesolve.platesolve()  → broad astrometry.net field solution, save products to working/
#
# Phase 2 (planned): custom distortion model built from Phase 1 products
# Phase 3 (planned): spectral trace angle extraction with per-trace WCS correction

from .stars import extract_stars, make_xylist
from .platesolve import platesolve, platesolve_xylist, PlatesolveResult, has_sip, wcs_summary
