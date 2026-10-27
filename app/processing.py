"""Pure NumPy raster processing (no Earth Engine dependency, easy to unit test)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.catalogue import LAYER_CATALOGUE
from app.planner import ExcludeClasses, KeepDirection, KeepRange, Plan, ScoreLinear

EARTH_RADIUS = 6378137.0


def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to 0-1, ignoring NaNs. A constant layer becomes 0."""
    arr = arr.astype("float64")
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan)
    if not finite.any():
        return out
    lo, hi = arr[finite].min(), arr[finite].max()
    out[finite] = 0.0 if hi == lo else (arr[finite] - lo) / (hi - lo)
    return out


def plan_layers(plan: Plan) -> list[str]:
    """Return source layers needed by the plan, including aspect support data."""
    names = list(dict.fromkeys(step.layer for step in plan.criteria))
    if any(isinstance(step, KeepDirection) for step in plan.criteria) and "slope" not in names:
        names.append("slope")
    return names


def _plan_arrays(rasters: dict[str, np.ndarray], plan: Plan) -> dict[str, np.ndarray]:
    required = plan_layers(plan)
    missing = sorted(set(required) - rasters.keys())
    if missing:
        raise ValueError(f"Missing plan raster(s): {missing}")
    arrays = {name: np.asarray(rasters[name], dtype="float64") for name in required}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("Plan rasters must have matching shapes")
    return arrays


def evaluate_criterion(step, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean pass mask for a hard criterion, or a 0-1 score for score_linear."""
    values = arrays[step.layer]
    with np.errstate(invalid="ignore"):
        if isinstance(step, KeepRange):
            passed = np.ones(values.shape, dtype=bool)
            if step.lower is not None:
                passed &= values >= step.lower if step.lower_inclusive else values > step.lower
            if step.upper is not None:
                passed &= values <= step.upper if step.upper_inclusive else values < step.upper
            return passed
        if isinstance(step, KeepDirection):
            distance = np.abs((values - step.centre + 180) % 360 - 180)
            return (arrays["slope"] > 0) & (distance <= step.tolerance)
        if isinstance(step, ExcludeClasses):
            codes = [LAYER_CATALOGUE[step.layer].classes[name] for name in step.classes]
            return ~np.isin(values, codes)
        if isinstance(step, ScoreLinear):
            score = np.clip((values - step.lower) / (step.upper - step.lower), 0, 1)
            return 1 - score if step.preference == "decreasing" else score
    raise ValueError(f"Unsupported operation: {step.operation}")


def apply_plan(rasters: dict[str, np.ndarray], plan: Plan) -> dict[str, np.ndarray]:
    """Apply the plan's hard criteria (ANDed) and weighted linear scores.

    Selection is defined only by the plan's hard criteria; scores rank the
    selected pixels and are never cut off by an extra threshold.
    """
    arrays = _plan_arrays(rasters, plan)
    shape = next(iter(arrays.values())).shape
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays.values()])
    eligible = valid.copy()
    weighted_scores = np.zeros(shape, dtype="float64")
    total_weight = 0.0

    for step in plan.criteria:
        outcome = evaluate_criterion(step, arrays)
        if isinstance(step, ScoreLinear):
            weighted_scores += outcome * step.weight
            total_weight += step.weight
        else:
            eligible &= outcome

    if total_weight:
        result = weighted_scores / total_weight
    else:
        result = np.ones(shape, dtype="float64")
    result[~eligible] = np.nan
    return {"score": result, "valid": valid, "eligible": eligible, "selected": eligible.copy()}


def describe_criterion(step) -> str:
    if isinstance(step, KeepRange):
        lo = "" if step.lower is None else f"{step.lower} {'≤' if step.lower_inclusive else '<'} "
        hi = "" if step.upper is None else f" {'≤' if step.upper_inclusive else '<'} {step.upper}"
        return f"keep {lo}{step.layer}{hi}"
    if isinstance(step, KeepDirection):
        return f"keep aspect {step.centre}° ± {step.tolerance}° (slope > 0)"
    if isinstance(step, ExcludeClasses):
        return f"exclude {step.layer}: {', '.join(step.classes)}"
    return (f"score {step.layer} {step.preference} {step.lower}→{step.upper} "
            f"(weight {step.weight})")


# ESA WorldCover v200 legend colours.
_LAND_COVER_COLOURS = {
    10: (0, 100, 0), 20: (255, 187, 34), 30: (255, 255, 76), 40: (240, 150, 255),
    50: (250, 0, 0), 60: (180, 180, 180), 70: (240, 240, 240), 80: (0, 100, 200),
    90: (0, 150, 160), 95: (0, 207, 117), 100: (250, 230, 160),
}


def layer_rgba(name: str, arr: np.ndarray) -> np.ndarray:
    """Render a raw input layer for visual inspection."""
    if LAYER_CATALOGUE[name].kind == "categorical":
        rgba = np.zeros(arr.shape + (4,), dtype="uint8")
        for code, colour in _LAND_COVER_COLOURS.items():
            rgba[arr == code] = (*colour, 255)
        return rgba
    return colorize(normalize(arr))


def criteria_diagnostics(rasters: dict[str, np.ndarray], plan: Plan, areas: np.ndarray) -> dict:
    """Per-layer and per-criterion images and pass statistics for checking a plan."""
    arrays = _plan_arrays(rasters, plan)
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays.values()])
    area_grid = np.broadcast_to(areas, valid.shape)
    valid_km2 = float(area_grid[valid].sum() / 1e6)

    layers = {}
    for name, arr in arrays.items():
        finite = arr[np.isfinite(arr)]
        layers[name] = {
            "kind": LAYER_CATALOGUE[name].kind,
            "units": LAYER_CATALOGUE[name].units,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "image": to_png_data_url(layer_rgba(name, arr)),
        }

    criteria = []
    for index, step in enumerate(plan.criteria):
        outcome = evaluate_criterion(step, arrays)
        entry = {"index": index, "layer": step.layer, "operation": step.operation,
                 "description": describe_criterion(step), "basis": step.basis,
                 "reason": step.reason}
        if isinstance(step, ScoreLinear):
            score = np.where(valid, outcome, np.nan)
            finite = score[np.isfinite(score)]
            entry["mean_score"] = round(float(finite.mean()), 3) if finite.size else None
            entry["image"] = to_png_data_url(colorize(score))
        else:
            passed = valid & outcome
            passed_km2 = float(area_grid[passed].sum() / 1e6)
            entry["pass_area_km2"] = round(passed_km2, 3)
            entry["pass_pct"] = round(100 * passed_km2 / valid_km2, 2) if valid_km2 else 0
            entry["image"] = to_png_data_url(mask_rgba(passed, color=(0, 170, 90)))
        criteria.append(entry)
    return {"layers": layers, "criteria": criteria}


# --- Web Mercator helpers -------------------------------------------------

def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    x = math.radians(lon) * EARTH_RADIUS
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * EARTH_RADIUS
    return x, y


def merc_to_lonlat(x, y):
    lon = np.degrees(np.asarray(x, dtype="float64") / EARTH_RADIUS)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y, dtype="float64") / EARTH_RADIUS)) - math.pi / 2)
    return lon, lat


def build_grid(west, south, east, north, ground_res_m=30.0, max_dim=1024) -> dict:
    """Pixel grid in EPSG:3857 covering the bbox.

    Mercator pixels are (ground_res / cos(lat)) metres wide; resolution is
    coarsened if needed so neither dimension exceeds max_dim.
    """
    x0, y0 = lonlat_to_merc(west, north)
    x1, y1 = lonlat_to_merc(east, south)
    lat_c = math.radians((south + north) / 2)
    scale = ground_res_m / math.cos(lat_c)
    scale = max(scale, (x1 - x0) / max_dim, (y0 - y1) / max_dim)
    width = max(1, math.ceil((x1 - x0) / scale))
    height = max(1, math.ceil((y0 - y1) / scale))
    w_lon, n_lat = merc_to_lonlat(x0, y0)
    e_lon, s_lat = merc_to_lonlat(x0 + width * scale, y0 - height * scale)
    return {
        "x0": x0, "y0": y0, "scale": scale, "width": width, "height": height,
        "ground_res_m": scale * math.cos(lat_c),
        "bounds": [[float(s_lat), float(w_lon)], [float(n_lat), float(e_lon)]],
    }


def pixel_areas_m2(grid: dict) -> np.ndarray:
    """True ground area of each pixel row (Mercator scale factor = 1/cos(lat))."""
    ys = grid["y0"] - (np.arange(grid["height"]) + 0.5) * grid["scale"]
    _, lats = merc_to_lonlat(np.zeros_like(ys), ys)
    side = grid["scale"] * np.cos(np.radians(lats))
    return (side ** 2)[:, None]


def save_geotiff(path: Path, array: np.ndarray, grid: dict, dtype="float32") -> None:
    """Save one georeferenced raster band in the analysis grid's CRS."""
    import rasterio
    from affine import Affine

    values = np.asarray(array)
    if values.shape != (grid["height"], grid["width"]):
        raise ValueError("Raster shape does not match the analysis grid")
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:3857",
        "transform": Affine(grid["scale"], 0, grid["x0"], 0, -grid["scale"], grid["y0"]),
        "compress": "deflate",
    }
    if dtype == "uint8":
        profile["nodata"] = 255
        output = np.where(np.isfinite(values), values, 255).astype("uint8")
    else:
        profile["nodata"] = -9999.0
        output = np.where(np.isfinite(values), values, -9999.0).astype(dtype)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(output, 1)


def mask_to_geojson(mask: np.ndarray, grid: dict, max_features: int = 5000) -> dict:
    """Vectorise the selected pixels into a lon/lat GeoJSON FeatureCollection."""
    from affine import Affine
    from rasterio import features

    transform = Affine(grid["scale"], 0, grid["x0"], 0, -grid["scale"], grid["y0"])
    feats = []
    for geom, _ in features.shapes(mask.astype("uint8"), mask=mask, transform=transform):
        rings = []
        for ring in geom["coordinates"]:
            xs, ys = zip(*ring)
            lons, lats = merc_to_lonlat(xs, ys)
            rings.append([[round(float(a), 6), round(float(b), 6)] for a, b in zip(lons, lats)])
        feats.append({"type": "Feature", "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": rings}})
        if len(feats) >= max_features:
            break
    return {"type": "FeatureCollection", "features": feats}


# --- Rendering -------------------------------------------------------------

_VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype="float64")


def colorize(arr01: np.ndarray) -> np.ndarray:
    """0-1 array -> RGBA uint8 (viridis), NaN transparent."""
    finite = np.isfinite(arr01)
    v = np.clip(np.nan_to_num(arr01), 0, 1) * (len(_VIRIDIS) - 1)
    i = np.floor(v).astype(int).clip(0, len(_VIRIDIS) - 2)
    f = (v - i)[..., None]
    rgb = _VIRIDIS[i] * (1 - f) + _VIRIDIS[i + 1] * f
    rgba = np.zeros(arr01.shape + (4,), dtype="uint8")
    rgba[..., :3] = rgb.round().astype("uint8")
    rgba[..., 3] = np.where(finite, 255, 0)
    return rgba


def mask_rgba(mask: np.ndarray, color=(255, 0, 80)) -> np.ndarray:
    rgba = np.zeros(mask.shape + (4,), dtype="uint8")
    rgba[mask] = (*color, 255)
    return rgba


def to_png_data_url(rgba: np.ndarray) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
