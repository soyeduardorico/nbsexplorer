import json
import math

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import main as web, processing as P, run_plan
from app.planner import Plan
from app.run_plan import _dates, square_bbox


def test_square_bbox_makes_approximate_ten_kilometre_test_area():
    west, south, east, north = square_bbox(-4.0, 56.5, 10)
    assert east > west and north > south
    assert (north - south) * 111_320 == pytest.approx(10_000)
    assert (east - west) * 111_320 * math.cos(math.radians(56.5)) == pytest.approx(
        10_000, rel=0.002
    )


@pytest.mark.parametrize("lon,lat,size", [
    (181, 56, 10), (180, 56, 10), (0, 86, 10), (0, 56, 0), (0, 56, float("nan")),
])
def test_square_bbox_rejects_invalid_inputs(lon, lat, size):
    with pytest.raises(ValueError):
        square_bbox(lon, lat, size)


def test_dates_validate_order_and_default_to_last_year():
    start, end = _dates("2024-01-01", "2024-12-31")
    assert (start, end) == ("2024-01-01", "2024-12-31")
    with pytest.raises(ValueError, match="after"):
        _dates("2025-01-01", "2024-12-31")


@pytest.fixture
def screening_plan():
    return Plan.model_validate({
        "title": "Test screening",
        "summary": "Screening only.",
        "criteria": [{
            "layer": "slope", "operation": "keep_range",
            "lower": None, "upper": 45.0,
            "lower_inclusive": True, "upper_inclusive": False,
            "reason": "Test mask.", "basis": "user_requirement",
        }],
        "assumptions": [],
        "questions": [],
        "limitations": ["Not an engineering assessment."],
    })


@pytest.mark.parametrize("use_saved_plan", [False, True])
def test_runner_saves_plan_rasters_and_report(
    tmp_path, monkeypatch, capsys, use_saved_plan, screening_plan,
):
    plan = screening_plan
    monkeypatch.setattr(run_plan, "generate_plan", lambda *args, **kwargs: (plan, "test-model"))
    monkeypatch.setattr(
        "app.gee.download_layers",
        lambda names, grid, start, end: {
            name: np.full((grid["height"], grid["width"]), 20.0) for name in names
        },
    )
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    output = tmp_path / "screening"
    arguments = [
        "Test slope screening", "--bbox", "-4", "56.5", "-3.99", "56.51",
        "--resolution-m", "1000", "--output-dir", str(output),
    ]
    if use_saved_plan:
        source = tmp_path / "saved.json"
        source.write_text(json.dumps({
            "brief": "Saved screening", "model": "saved-model",
            "plan": plan.model_dump(),
            "bounds": [-4, 56.5, -3.99, 56.51],
            "dates": {"start": "2025-01-01", "end": "2026-01-01"},
        }))

        def unexpected_generation(*args, **kwargs):
            pytest.fail("Saved-plan execution must not contact Gemini")

        monkeypatch.setattr(run_plan, "generate_plan", unexpected_generation)
        arguments = [
            "--plan-file", str(source),
            "--resolution-m", "1000", "--output-dir", str(output),
        ]
    assert run_plan.main(arguments) == 0

    assert {path.name for path in output.iterdir()} == {
        "plan.json", "suitability.tif", "eligible.tif", "selected.tif",
        "selected.geojson", "report.json",
    }
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "screening_complete_for_review"
    assert report["scoring_mode"] == "mask_only"
    assert report["areas_km2"]["selected"] > 0
    assert report["grid"]["crs"] == "EPSG:3857"
    assert report["outputs"][1] == "suitability.tif"
    if use_saved_plan:
        assert report["brief"] == "Saved screening"
        assert report["model"] == "saved-model"
        assert "selection_threshold" not in report
        assert report["dates"] == {"start": "2025-01-01", "end": "2026-01-01"}
    assert "suitability.tif" in capsys.readouterr().out


@pytest.mark.parametrize("weighted", [False, True])
def test_web_generates_then_executes_reviewed_plan(monkeypatch, screening_plan, weighted):
    record = screening_plan.model_dump()
    if weighted:
        record["criteria"].append({
            "layer": "slope", "operation": "score_linear", "lower": 0.0, "upper": 40.0,
            "preference": "increasing", "weight": 1.0,
            "reason": "Test score", "basis": "user_requirement",
        })
    plan = Plan.model_validate(record)
    bounds = dict(west=-4, south=56.5, east=-3.99, north=56.51, resolution_m=1000)
    calls = []

    def generate(brief):
        calls.append(brief)
        return plan, "test-gemini"

    def download(names, grid, start, end):
        calls.append(names)
        assert (start, end) == ("2025-01-01", "2026-01-01")
        return {"slope": np.resize([20.0, 50.0, np.nan, 10.0],
                                  (grid["height"], grid["width"]))}

    monkeypatch.setattr(web, "generate_plan", generate)
    monkeypatch.setattr("app.gee.download_layers", download)
    params = dict(**bounds, start_date="2025-01-01", end_date="2026-01-01")
    draft = web.plan(web.PlanRequest(**params, brief="Test woodland restoration"))
    assert len(calls) == 1  # No download until the reviewed plan is submitted.
    assert "west=-4" in calls[0]
    assert draft["model"] == "test-gemini"
    response = web.analyze(web.AnalyzeRequest(**params, plan=draft["plan"]))
    grid = web._analysis_grid(web.AreaRequest(**params))
    expected = P.apply_plan(download(["slope"], grid, params["start_date"], params["end_date"]),
                            plan)
    areas = np.broadcast_to(P.pixel_areas_m2(grid), expected["valid"].shape)
    assert response["stats"]["selected_pixels"] == int(expected["selected"].sum())
    assert response["stats"]["valid_area_km2"] == round(
        float(areas[expected["valid"]].sum() / 1e6), 3
    )
    assert response["stats"]["selected_area_km2"] == round(
        float(areas[expected["selected"]].sum() / 1e6), 3
    )
    assert response["scoring_mode"] == ("weighted_mean" if weighted else "mask_only")
    assert response["images"]["index"].startswith("data:image/png;base64,")
    assert response["selected_geojson"]["type"] == "FeatureCollection"
    assert response["diagnostics"] is None

    diag = web.analyze(web.AnalyzeRequest(**params, plan=draft["plan"], diagnostics=True))
    criteria = diag["diagnostics"]["criteria"]
    assert len(criteria) == len(plan.criteria)
    assert set(diag["diagnostics"]["layers"]) == {"slope"}
    assert all(c["image"].startswith("data:image/png;base64,") for c in criteria)
    assert "pass_area_km2" in criteria[0]
    if weighted:
        assert "mean_score" in criteria[-1]


def test_web_analysis_requires_a_reviewed_plan():
    params = dict(west=-4, south=56.5, east=-3.99, north=56.51)
    with pytest.raises(ValidationError, match="plan"):
        web.AnalyzeRequest(**params)
    assert not any(getattr(r, "path", None) == "/api/layers" for r in web.app.routes)


@pytest.mark.parametrize("updates", [
    {"brief": " "}, {"east": -5}, {"start_date": "invalid"},
    {"start_date": "2026-02-01", "end_date": "2026-01-01"},
])
def test_web_rejects_invalid_inputs(updates):
    params = dict(west=-4, south=56.5, east=-3.99, north=56.51, brief="Test")
    params.update(updates)
    with pytest.raises(ValidationError):
        web.PlanRequest(**params)


def test_web_checks_area_before_contacting_gemini(monkeypatch):
    def unexpected_call(*args):
        pytest.fail("Oversized area must fail before Gemini")

    monkeypatch.setattr(web, "generate_plan", unexpected_call)
    with pytest.raises(HTTPException) as error:
        web.plan(web.PlanRequest(west=-10, south=40, east=10, north=60, brief="Test"))
    assert error.value.status_code == 400


def test_web_reports_gemini_errors(monkeypatch):
    def fail(*args):
        raise ValueError("Missing API key")

    monkeypatch.setattr(web, "generate_plan", fail)
    with pytest.raises(HTTPException, match="Missing API key") as error:
        web.plan(web.PlanRequest(west=-4, south=56.5, east=-3.99, north=56.51, brief="Test"))
    assert error.value.status_code == 502
