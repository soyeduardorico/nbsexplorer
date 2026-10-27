import numpy as np

from app import processing as P
from app.planner import Plan


def test_normalize_range_and_nan():
    out = P.normalize(np.array([[2.0, 4.0], [np.nan, 6.0]]))
    assert np.isnan(out[1, 0])
    np.testing.assert_allclose(out[[0, 0, 1], [0, 1, 1]], [0, 0.5, 1])


def test_normalize_constant_is_zero():
    np.testing.assert_array_equal(P.normalize(np.full((2, 2), 7.0)), np.zeros((2, 2)))


def test_grid_bounds_cover_bbox_and_respect_max_dim():
    g = P.build_grid(-3.6, 50.67, -3.45, 50.77, ground_res_m=30, max_dim=1024)
    (s, w), (n, e) = g["bounds"]
    assert abs(w + 3.6) < 1e-9 and abs(n - 50.77) < 1e-9
    assert e >= -3.45 and s <= 50.67
    assert g["width"] <= 1024 and g["height"] <= 1024
    assert abs(g["ground_res_m"] - 30) < 1e-6
    big = P.build_grid(-10, 40, 10, 60, ground_res_m=30, max_dim=500)
    assert big["width"] <= 500 and big["height"] <= 500


def test_merc_roundtrip():
    x, y = P.lonlat_to_merc(-3.53, 50.72)
    lon, lat = P.merc_to_lonlat(x, y)
    assert abs(lon + 3.53) < 1e-9 and abs(lat - 50.72) < 1e-9


def test_geojson_and_png():
    g = P.build_grid(-3.6, 50.67, -3.45, 50.77, ground_res_m=200)
    mask = np.zeros((g["height"], g["width"]), bool)
    mask[2:5, 3:7] = True
    fc = P.mask_to_geojson(mask, g)
    assert len(fc["features"]) == 1
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert -3.6 <= lon <= -3.45 and 50.67 <= lat <= 50.77
    assert P.to_png_data_url(P.colorize(np.random.rand(4, 4))).startswith("data:image/png;base64,")


def _plan(criteria):
    return Plan.model_validate({
        "title": "Test plan",
        "summary": "A test screening plan.",
        "criteria": criteria,
        "assumptions": [],
        "questions": [],
        "limitations": ["Screening only."],
    })


def test_apply_plan_uses_weighted_scores_without_renormalising_or_cutoff():
    plan = _plan([
        {
            "layer": "ndvi", "operation": "score_linear",
            "lower": 0.0, "upper": 1.0, "preference": "increasing", "weight": 1.0,
            "reason": "Test score.", "basis": "user_requirement",
        },
        {
            "layer": "elevation", "operation": "score_linear",
            "lower": 0.0, "upper": 100.0, "preference": "decreasing", "weight": 3.0,
            "reason": "Test score.", "basis": "user_requirement",
        },
    ])
    result = P.apply_plan({
        "ndvi": np.array([[0.0, 1.0, 0.5]]),
        "elevation": np.array([[100.0, 0.0, 50.0]]),
    }, plan)
    np.testing.assert_allclose(result["score"], [[0.0, 1.0, 0.5]])
    np.testing.assert_array_equal(result["selected"], [[True, True, True]])
    assert result["valid"].all() and result["eligible"].all()


def test_apply_plan_combines_range_direction_and_class_masks():
    plan = _plan([
        {
            "layer": "aspect", "operation": "keep_direction",
            "centre": 0.0, "tolerance": 20.0,
            "reason": "Test direction.", "basis": "user_requirement",
        },
        {
            "layer": "slope", "operation": "keep_range",
            "lower": 1.0, "upper": 45.0,
            "lower_inclusive": True, "upper_inclusive": False,
            "reason": "Test slope.", "basis": "user_requirement",
        },
        {
            "layer": "land_cover", "operation": "exclude_classes",
            "classes": ["built_up"],
            "reason": "Test exclusion.", "basis": "user_requirement",
        },
    ])
    assert P.plan_layers(plan) == ["aspect", "slope", "land_cover"]
    result = P.apply_plan({
        "aspect": np.array([[359.0, 10.0, 180.0, 0.0]]),
        "slope": np.array([[5.0, 0.0, 5.0, 5.0]]),
        "land_cover": np.array([[10.0, 10.0, 10.0, 50.0]]),
    }, plan)
    np.testing.assert_array_equal(result["eligible"], [[True, False, False, False]])
    np.testing.assert_allclose(result["score"], [[1.0, np.nan, np.nan, np.nan]], equal_nan=True)


def test_direction_only_plan_fetches_slope_and_rejects_flat_aspect():
    plan = _plan([{
        "layer": "aspect", "operation": "keep_direction",
        "centre": 0.0, "tolerance": 20.0,
        "reason": "Test direction.", "basis": "user_requirement",
    }])
    assert P.plan_layers(plan) == ["aspect", "slope"]
    result = P.apply_plan({
        "aspect": np.array([[0.0, 0.0]]),
        "slope": np.array([[0.0, 1.0]]),
    }, plan)
    np.testing.assert_array_equal(result["eligible"], [[False, True]])


def test_apply_plan_keeps_missing_data_unknown_and_rejects_bad_inputs():
    plan = _plan([{
        "layer": "slope", "operation": "keep_range",
        "lower": None, "upper": 45.0,
        "lower_inclusive": True, "upper_inclusive": False,
        "reason": "Test slope.", "basis": "user_requirement",
    }])
    result = P.apply_plan({"slope": np.array([[10.0, np.nan]])}, plan)
    np.testing.assert_array_equal(result["valid"], [[True, False]])
    np.testing.assert_array_equal(result["selected"], [[True, False]])
    assert np.isnan(result["score"][0, 1])
    with np.testing.assert_raises_regex(ValueError, "Missing plan raster"):
        P.apply_plan({}, plan)


def test_criteria_diagnostics_reports_each_criterion_separately():
    plan = _plan([
        {
            "layer": "slope", "operation": "keep_range",
            "lower": 1.0, "upper": None,
            "lower_inclusive": True, "upper_inclusive": False,
            "reason": "Test slope.", "basis": "user_requirement",
        },
        {
            "layer": "land_cover", "operation": "exclude_classes",
            "classes": ["built_up"],
            "reason": "Test exclusion.", "basis": "user_requirement",
        },
    ])
    rasters = {
        "slope": np.array([[5.0, 0.0, 5.0, 5.0]]),
        "land_cover": np.array([[10.0, 10.0, 50.0, 30.0]]),
    }
    diag = P.criteria_diagnostics(rasters, plan, np.ones((1, 1)) * 1e6)
    assert [c["pass_pct"] for c in diag["criteria"]] == [75.0, 75.0]
    assert diag["criteria"][1]["description"] == "exclude land_cover: built_up"
    assert diag["layers"]["land_cover"]["kind"] == "categorical"
    assert diag["layers"]["slope"]["image"].startswith("data:image/png;base64,")


def test_save_geotiff_preserves_crs_transform_and_nodata(tmp_path):
    grid = P.build_grid(-4.0, 56.5, -3.99, 56.51, ground_res_m=100)
    values = np.ones((grid["height"], grid["width"]))
    values[0, 0] = np.nan
    path = tmp_path / "result.tif"
    P.save_geotiff(path, values, grid)
    import rasterio

    with rasterio.open(path) as dataset:
        assert dataset.crs.to_string() == "EPSG:3857"
        assert dataset.transform.a == grid["scale"]
        assert dataset.nodata == -9999
        assert dataset.read(1)[0, 0] == dataset.nodata
