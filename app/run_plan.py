"""Generate, review and execute an NBS screening plan against Earth Engine data."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import processing as P
from app.planner import Plan, generate_plan

MAX_AREA_KM2 = 25_000


def square_bbox(lon: float, lat: float, size_km: float) -> tuple[float, float, float, float]:
    """Approximate a local square in longitude/latitude around its centre."""
    if not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError("Center longitude must be between -180 and 180")
    if not math.isfinite(lat) or not -85 <= lat <= 85:
        raise ValueError("Center latitude must be between -85 and 85")
    if not math.isfinite(size_km) or size_km <= 0:
        raise ValueError("Square size must be a positive finite number")
    half_lat = size_km * 1000 / (2 * 111_320)
    half_lon = size_km * 1000 / (2 * 111_320 * math.cos(math.radians(lat)))
    bounds = lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat
    west, south, east, north = bounds
    if west < -180 or east > 180 or south < -85 or north > 85:
        raise ValueError("The requested square extends outside valid analysis coordinates")
    return bounds


def _dates(start: str | None, end: str | None) -> tuple[str, str]:
    end_date = date.fromisoformat(end) if end else date.today()
    start_date = date.fromisoformat(start) if start else end_date - timedelta(days=365)
    if start_date > end_date:
        raise ValueError("Start date must not be after end date")
    return start_date.isoformat(), end_date.isoformat()


def _area(args) -> tuple[float, float, float, float]:
    if args.bbox is not None:
        if args.center_lon is not None or args.center_lat is not None:
            raise ValueError("Use either --bbox or --center-lon/--center-lat, not both")
        west, south, east, north = args.bbox
        if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("Bounding box must be west south east north within valid coordinates")
        return west, south, east, north
    if (args.center_lon is None) != (args.center_lat is None):
        raise ValueError("Provide both --center-lon and --center-lat")
    if args.center_lon is None:
        args.center_lon = float(input("Area centre longitude (decimal degrees): ").strip())
        args.center_lat = float(input("Area centre latitude (decimal degrees): ").strip())
    return square_bbox(args.center_lon, args.center_lat, args.size_km)


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brief", nargs="?", help="NBS need and site context; prompted if omitted")
    parser.add_argument("--plan-file", type=Path,
                        help="Run a saved plan record without contacting Gemini")
    area = parser.add_mutually_exclusive_group()
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    area.add_argument("--center-lon", type=float, help="Centre longitude for a square test area")
    parser.add_argument("--center-lat", type=float, help="Centre latitude; required with --center-lon")
    parser.add_argument("--size-km", type=float, default=10, help="Square side length (default: 10 km)")
    parser.add_argument("--resolution-m", type=float, default=30, help="Target ground resolution (default: 30 m)")
    parser.add_argument("--start-date", help="Sentinel-2 start date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Sentinel-2 end date, YYYY-MM-DD")
    parser.add_argument("--model", help="Gemini model ID (default: config.GEMINI_MODEL)")
    parser.add_argument("--timeout", type=float, default=120, help="Gemini request timeout in seconds")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory (default: timestamped nbs-run directory)")
    args = parser.parse_args(argv)

    try:
        saved_plan = None
        if args.plan_file is not None:
            with args.plan_file.open(encoding="utf-8") as source:
                record = json.load(source)
            if not isinstance(record, dict):
                raise ValueError("Saved plan record must be a JSON object")
            saved_plan = Plan.model_validate(record.get("plan"))
            bounds_record = record.get("bounds")
            dates_record = record.get("dates")
            if (
                not isinstance(bounds_record, list) or len(bounds_record) != 4
                or any(not isinstance(value, (int, float)) for value in bounds_record)
                or not isinstance(dates_record, dict)
                or not isinstance(record.get("brief"), str)
                or not isinstance(record.get("model"), str)
            ):
                raise ValueError(
                    "Saved plan needs numeric bounds, dates, brief and model metadata"
                )
            args.bbox = [float(value) for value in bounds_record]
            args.brief = record["brief"]
            args.start_date = dates_record.get("start")
            args.end_date = dates_record.get("end")
            if not isinstance(args.start_date, str) or not isinstance(args.end_date, str):
                raise ValueError("Saved plan needs start and end dates")
        bounds = _area(args)
        start, end = _dates(args.start_date, args.end_date)
        if not math.isfinite(args.resolution_m) or not 10 <= args.resolution_m <= 1000:
            raise ValueError("Resolution must be between 10 and 1000 metres")
        if args.output_dir is None:
            args.output_dir = Path(f"nbs-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        if args.output_dir.exists():
            raise ValueError(f"Output directory already exists; choose another: {args.output_dir}")
        brief = args.brief
        if brief is None:
            brief = input("Describe the NBS need and site context:\n> ").strip()
        if not brief.strip():
            raise ValueError("The NBS brief cannot be empty")

        print(f"Analysis bounds (west, south, east, north): {bounds}")
        print(f"Sentinel-2 composite dates: {start} to {end}")
        if saved_plan is not None:
            plan, model = saved_plan, record["model"]
            print(f"Loaded saved plan from {args.plan_file}. Gemini is not used.", flush=True)
        else:
            print(f"Generating plan with Gemini (socket timeout: {args.timeout:g} seconds)...",
                  flush=True)
            plan, model = generate_plan(
                brief,
                model=args.model,
                timeout=args.timeout,
            )
        print(f"\nDraft screening plan ({model}) - not an engineering assessment")
        print(plan.model_dump_json(indent=2))
        has_scores = any(step.operation == "score_linear" for step in plan.criteria)
        if has_scores:
            print("\nSelected areas are those passing all hard criteria in the plan; "
                  "scores only rank pixels within them.")
        else:
            print("\nThis plan has no soft scores; its output is a binary eligibility screen, "
                  "not a ranking of eligible sites.")
        approval = input("Run this plan and save the screening outputs? [y/N]: ").strip().lower()
        if approval not in {"y", "yes"}:
            print("Not approved. No layers downloaded and nothing saved.")
            return 0

        grid = P.build_grid(*bounds, ground_res_m=args.resolution_m)
        requested_area_km2 = float(
            P.pixel_areas_m2(grid).sum() * grid["width"] / 1e6
        )
        if requested_area_km2 > MAX_AREA_KM2:
            raise ValueError(
                f"Area too large ({requested_area_km2:,.0f} km2); "
                f"maximum is {MAX_AREA_KM2:,} km2"
            )

        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(args.output_dir / "plan.json", {
            "brief": brief,
            "model": model,
            "plan": plan.model_dump(),
            "dates": {"start": start, "end": end},
            "bounds": bounds,
        })
        layer_names = P.plan_layers(plan)
        from app.gee import download_layers

        print(f"Downloading Earth Engine layers: {', '.join(layer_names)}...", flush=True)
        try:
            rasters = download_layers(layer_names, grid, start, end)
        except Exception as exc:
            raise RuntimeError(f"Earth Engine download failed: {exc}") from exc

        print("Applying plan criteria and saving screening outputs...", flush=True)
        result = P.apply_plan(rasters, plan)
        areas = P.pixel_areas_m2(grid)
        area_grid = np.broadcast_to(areas, result["valid"].shape)
        valid_area = float(area_grid[result["valid"]].sum() / 1e6)
        eligible_area = float(area_grid[result["eligible"]].sum() / 1e6)
        selected_area = float(area_grid[result["selected"]].sum() / 1e6)
        valid_scores = result["score"][np.isfinite(result["score"])]

        P.save_geotiff(args.output_dir / "suitability.tif", result["score"], grid)
        P.save_geotiff(
            args.output_dir / "eligible.tif",
            np.where(result["valid"], result["eligible"], np.nan),
            grid,
            dtype="uint8",
        )
        P.save_geotiff(
            args.output_dir / "selected.tif",
            np.where(result["valid"], result["selected"], np.nan),
            grid,
            dtype="uint8",
        )
        _write_json(
            args.output_dir / "selected.geojson",
            P.mask_to_geojson(result["selected"], grid),
        )
        layer_stats = {}
        for name, raster in rasters.items():
            values = raster[np.isfinite(raster)]
            layer_stats[name] = (
                {"min": float(values.min()), "max": float(values.max()),
                 "mean": float(values.mean())}
                if values.size else None
            )
        report = {
            "status": "screening_complete_for_review",
            "brief": brief,
            "model": model,
            "plan": plan.model_dump(),
            "bounds": grid["bounds"],
            "grid": {
                "width": grid["width"],
                "height": grid["height"],
                "target_ground_resolution_m": round(grid["ground_res_m"], 2),
                "crs": "EPSG:3857",
            },
            "dates": {"start": start, "end": end},
            "scoring_mode": "weighted_mean" if has_scores else "mask_only",
            "requested_area_km2": round(requested_area_km2, 3),
            "areas_km2": {
                "valid": round(valid_area, 3),
                "eligible": round(eligible_area, 3),
                "selected": round(selected_area, 3),
            },
            "selected_percent_of_valid": (
                round(100 * selected_area / valid_area, 2) if valid_area else 0
            ),
            "score_stats": (
                {"min": float(valid_scores.min()), "max": float(valid_scores.max()),
                 "mean": float(valid_scores.mean())}
                if has_scores and valid_scores.size else None
            ),
            "layer_stats": layer_stats,
            "outputs": [
                "plan.json", "suitability.tif", "eligible.tif",
                "selected.tif", "selected.geojson",
            ],
            "limitations": [
                "This is preliminary screening, not an engineering safety or ecological feasibility assessment.",
                "Verify source data, dates, assumptions, land access and local constraints with qualified practitioners.",
            ],
        }
        _write_json(args.output_dir / "report.json", report)
        print(f"\nScreening complete. Results saved in {args.output_dir.resolve()}:")
        print(f"  Suitability raster: {args.output_dir / 'suitability.tif'}")
        print(f"  Selected areas:     {args.output_dir / 'selected.geojson'}")
        print(f"  Report:             {args.output_dir / 'report.json'}")
        print(f"  Selected area:      {selected_area:.3f} km2")
        print("Open the GeoTIFF in QGIS or another GIS to inspect the result.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Planner error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. Nothing further was saved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
