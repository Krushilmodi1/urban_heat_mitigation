"""Landsat 8/9 Collection 2 Level-2 constants and reference formulas.

Single source of truth for scale factors, valid ranges and QA bits. The Earth
Engine code in extract_gee_data.py uses these constants; the NumPy versions
below are used by the offline tests to check the arithmetic.

Source: USGS Landsat 8-9 Collection 2 Level-2 Science Product Guide (LSDS-1619).
"""
import numpy as np

# Surface reflectance: rho = DN * 2.75e-5 - 0.2 ; valid DN 7273-43636 (rho ~ 0.0 to 1.0)
SR_MULT, SR_ADD = 0.0000275, -0.2
SR_VALID_DN = (7273, 43636)
# Surface temperature (ST_B10): T[K] = DN * 0.00341802 + 149.0 ; DN 0 = fill
ST_MULT, ST_ADD = 0.00341802, 149.0
ST_FILL_DN = 0
KELVIN_OFFSET = 273.15

# QA_PIXEL bits that must all be 0 for a clear pixel:
# 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow
QA_CLEAR_MASK = 0b11111
# QA_RADSAT: any set bit = saturated band (bit 11 = terrain occlusion) -> masked

# Band roles (identical for Landsat 8 OLI and Landsat 9 OLI-2)
RED, NIR, SWIR1, THERMAL = "SR_B4", "SR_B5", "SR_B6", "ST_B10"
OPTICAL_USED = [RED, NIR, SWIR1]


def sr_reflectance(dn):
    return np.asarray(dn, dtype=float) * SR_MULT + SR_ADD


def st_kelvin(dn):
    return np.asarray(dn, dtype=float) * ST_MULT + ST_ADD


def st_celsius(dn):
    return st_kelvin(dn) - KELVIN_OFFSET


def qa_clear(qa_pixel):
    return (np.asarray(qa_pixel, dtype=np.int64) & QA_CLEAR_MASK) == 0


def normalized_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def ndvi(nir, red):
    """(NIR - Red) / (NIR + Red) on reflectance."""
    return normalized_difference(nir, red)


def ndbi(swir1, nir):
    """(SWIR1 - NIR) / (SWIR1 + NIR) on reflectance."""
    return normalized_difference(swir1, nir)
