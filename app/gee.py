"""Google Earth Engine layer catalogue and raster download."""
from __future__ import annotations

import os
import threading

import ee
import numpy as np

from app import config

NODATA = -9999.0
_init_lock = threading.Lock()
_initialized = False


def init_ee() -> None:
    """Initialise Earth Engine once. Requires `earthengine authenticate` (or a
    service account key) and an EE-enabled Google Cloud project."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        project = os.environ.get("EE_PROJECT", config.EE_PROJECT)
        key_file = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
        sa_email = os.environ.get("EE_SERVICE_ACCOUNT")
        if sa_email and key_file:
            ee.Initialize(ee.ServiceAccountCredentials(sa_email, key_file), project=project)
        else:
            ee.Initialize(project=project)
        _initialized = True


# --- Layer catalogue ------------------------------------------------------

def _dem() -> ee.Image:
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    return col.mosaic().setDefaultProjection(col.first().projection())


def _s2_composite(region: ee.Geometry, start: str, end: str) -> ee.Image:
    def mask_clouds(img):
        scl = img.select("SCL")
        # keep vegetation(4), bare soil(5), water(6), unclassified(7)
        good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return img.updateMask(good)

    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(mask_clouds)
            .median())


LAYERS = {
    "slope": {"label": "Slope (degrees, Copernicus DEM 30 m)",
              "build": lambda r, s, e: ee.Terrain.slope(_dem())},
    "ndvi": {"label": "NDVI (Sentinel-2 median composite)",
             "build": lambda r, s, e: _s2_composite(r, s, e).normalizedDifference(["B8", "B4"])},
    "elevation": {"label": "Elevation (m, Copernicus DEM 30 m)",
                  "build": lambda r, s, e: _dem()},
    "aspect": {"label": "Aspect (degrees, Copernicus DEM 30 m)",
               "build": lambda r, s, e: ee.Terrain.aspect(_dem())},
    "ndwi": {"label": "NDWI (Sentinel-2 median composite)",
             "build": lambda r, s, e: _s2_composite(r, s, e).normalizedDifference(["B3", "B8"])},
    "land_cover": {"label": "Land cover (ESA WorldCover 2021, 10 m; categorical)",
                   "build": lambda r, s, e: ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")},
}


def download_layers(names: list[str], grid: dict, start: str, end: str) -> dict[str, np.ndarray]:
    """Download all layers on one shared EPSG:3857 grid so pixels align.
    Returns float arrays with NaN for no-data."""
    init_ee()
    (s, w), (n, e) = grid["bounds"]
    region = ee.Geometry.Rectangle([w, s, e, n], None, False)

    bands = [LAYERS[name]["build"](region, start, end).rename(name).toFloat() for name in names]
    stacked = ee.Image.cat(bands).unmask(NODATA, False)

    data = ee.data.computePixels({
        "expression": stacked,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": grid["width"], "height": grid["height"]},
            "affineTransform": {
                "scaleX": grid["scale"], "shearX": 0, "translateX": grid["x0"],
                "shearY": 0, "scaleY": -grid["scale"], "translateY": grid["y0"],
            },
            "crsCode": "EPSG:3857",
        },
    })

    out = {}
    for name in names:
        arr = np.asarray(data[name], dtype="float64")
        arr[arr == NODATA] = np.nan
        out[name] = arr
    return out
