"""Planner-visible layers and operations. No Earth Engine authentication required."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Layer:
    source: str
    units: str
    kind: Literal["continuous", "circular", "categorical"]
    resolution_m: int
    limitation: str
    minimum: float | None = None
    maximum: float | None = None
    classes: dict[str, int] = field(default_factory=dict)


LAYER_CATALOGUE = {
    "slope": Layer(
        "COPERNICUS/DEM/GLO30: DEM -> slope", "degrees", "continuous", 30,
        "Surface model, not bare earth. Slope alone cannot establish ground stability.",
        0, 90,
    ),
    "aspect": Layer(
        "COPERNICUS/DEM/GLO30: DEM -> aspect", "degrees clockwise from north",
        "circular", 30,
        "Undefined on flat terrain. Exposure depends on hemisphere, season and shading; "
        "not a measurement of available light.",
        0, 360,
    ),
    "elevation": Layer(
        "COPERNICUS/DEM/GLO30: DEM", "metres EGM2008", "continuous", 30,
        "Surface model including vegetation/buildings; not a tidal datum.",
    ),
    "ndvi": Layer(
        "COPERNICUS/S2_SR_HARMONIZED: (B8-B4)/(B8+B4)", "index",
        "continuous", 10,
        "Cloud-masked median over requested dates. Not species, stability or light availability.",
        -1, 1,
    ),
    "ndwi": Layer(
        "COPERNICUS/S2_SR_HARMONIZED: (B3-B8)/(B3+B8)", "index",
        "continuous", 10,
        "Cloud-masked median over requested dates. Not a flood or soil-moisture model.",
        -1, 1,
    ),
    "land_cover": Layer(
        "ESA/WorldCover/v200: Map (2021)", "class code", "categorical", 10,
        "Static 2021 classification; verify current land use and local restrictions.",
        classes={
            "tree_cover": 10, "shrubland": 20, "grassland": 30, "cropland": 40,
            "built_up": 50, "bare_sparse": 60, "snow_ice": 70,
            "permanent_water": 80, "herbaceous_wetland": 90, "mangroves": 95,
            "moss_lichen": 100,
        },
    ),
}

OPERATION_CATALOGUE = {
    "keep_range": {
        "accepts": ["continuous"],
        "effect": "Hard mask: keep values between optional lower and upper bounds, "
                  "with explicit inclusive/exclusive endpoints. At least one bound required.",
    },
    "keep_direction": {
        "accepts": ["circular"],
        "effect": "Hard mask: shortest angular distance to centre <= tolerance; wraps at 360. "
                  "Flat-terrain aspect is unknown, not suitable.",
    },
    "exclude_classes": {
        "accepts": ["categorical"],
        "effect": "Negative mask: exclude named classes from the layer catalogue.",
    },
    "score_linear": {
        "accepts": ["continuous"],
        "effect": "Soft preference: clip((x-lower)/(upper-lower), 0, 1); "
                  "reverse for decreasing preference. Fixed bounds, not local min/max.",
    },
}

COMBINATION_RULES = {
    "masks": "Logical AND of all hard masks; no masks means all data-valid pixels.",
    "scores": "Weighted mean of score_linear outputs, only within the mask. "
              "No final renormalisation. No scores means a mask-only plan.",
    "missing_data": "Unknown if any required input is missing; never assume suitable.",
    "execution": "The plan runner applies these rules to aligned Earth Engine rasters and "
                 "saves suitability, eligibility and selected-area outputs for review.",
}
