"""FastAPI backend: screen an NBS brief and selected area with Gemini and GEE."""
from __future__ import annotations

import datetime as dt
from http.client import HTTPException as HTTPClientException
import sys
from pathlib import Path
from typing import Literal

# Allow running this file directly (python app/main.py / VS Code Run button),
# not only as `uvicorn app.main:app`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from app import processing as P
from app.planner import Plan, generate_plan, reply_to_issue_conversation

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_AREA_KM2 = 25_000

app = FastAPI(title="NBS screening map")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AreaRequest(BaseModel):
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-85, le=85)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-85, le=85)
    resolution_m: float = Field(30, ge=10, le=1000)
    start_date: str | None = None
    end_date: str | None = None

    @model_validator(mode="after")
    def _check_area(self):
        if self.east <= self.west or self.north <= self.south:
            raise ValueError("Invalid bounding box")
        end = dt.date.fromisoformat(self.end_date) if self.end_date else dt.date.today()
        start = (dt.date.fromisoformat(self.start_date) if self.start_date
                 else end - dt.timedelta(days=365))
        if start > end:
            raise ValueError("Start date must not be after end date")
        self.start_date, self.end_date = start.isoformat(), end.isoformat()
        return self


class PlanRequest(AreaRequest):
    brief: str = Field(min_length=1, max_length=10000)

    @field_validator("brief")
    @classmethod
    def _check_brief(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The NBS brief cannot be empty")
        return value.strip()


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def _check_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Conversation messages cannot be empty")
        return value


class ConversationRequest(BaseModel):
    messages: list[ConversationMessage] = Field(min_length=1, max_length=40)
    summarize: bool = False

    @model_validator(mode="after")
    def _check_size(self):
        if sum(len(message.content) for message in self.messages) > 30000:
            raise ValueError("Conversation is too long; start a new discussion")
        if self.messages[0].role != "user":
            raise ValueError("A discussion must start with a user message")
        if any(
            self.messages[index - 1].role == message.role
            for index, message in enumerate(self.messages[1:], start=1)
        ):
            raise ValueError("Conversation messages must alternate between user and assistant")
        if not self.summarize and self.messages[-1].role != "user":
            raise ValueError("Send a message before requesting a reply")
        return self


class AnalyzeRequest(AreaRequest):
    plan: Plan
    diagnostics: bool = False


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    from app.gee import download_layers

    grid = _analysis_grid(req)
    areas = P.pixel_areas_m2(grid)

    end, start = req.end_date, req.start_date

    try:
        rasters = download_layers(P.plan_layers(req.plan), grid, start, end)
    except Exception as exc:  # EE errors (auth, quota, empty collections, ...)
        raise HTTPException(502, f"Earth Engine error: {exc}") from exc

    # Selection comes only from the plan's criteria; there is no extra score cutoff.
    result = P.apply_plan(rasters, req.plan)
    index, mask, valid = result["score"], result["selected"], result["valid"]
    scoring_mode = ("weighted_mean" if any(
        step.operation == "score_linear" for step in req.plan.criteria
    ) else "mask_only")
    diagnostics = (P.criteria_diagnostics(rasters, req.plan, areas)
                   if req.diagnostics else None)
    area_grid = np.broadcast_to(areas, mask.shape)
    selected_km2 = float(area_grid[mask].sum() / 1e6)
    valid_km2 = float(area_grid[valid].sum() / 1e6)

    layer_stats = {}
    for name, arr in rasters.items():
        finite = arr[np.isfinite(arr)]
        layer_stats[name] = (
            {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}
            if finite.size else None
        )

    return {
        "bounds": grid["bounds"],
        "grid": {"width": grid["width"], "height": grid["height"],
                 "resolution_m": round(grid["ground_res_m"], 2)},
        "dates": {"start": start, "end": end},
        "scoring_mode": scoring_mode,
        "layer_stats": layer_stats,
        "stats": {
            "valid_area_km2": round(valid_km2, 3),
            "selected_area_km2": round(selected_km2, 3),
            "selected_pct": round(100 * selected_km2 / valid_km2, 2) if valid_km2 else 0,
            "selected_pixels": int(mask.sum()),
        },
        "images": {
            "index": P.to_png_data_url(P.colorize(index)),
            "mask": P.to_png_data_url(P.mask_rgba(mask)),
        },
        "selected_geojson": P.mask_to_geojson(mask, grid),
        "diagnostics": diagnostics,
    }


def _analysis_grid(req: AreaRequest) -> dict:
    grid = P.build_grid(req.west, req.south, req.east, req.north, req.resolution_m)
    total_km2 = float(P.pixel_areas_m2(grid).sum() * grid["width"] / 1e6)
    if total_km2 > MAX_AREA_KM2:
        raise HTTPException(400, f"Area too large ({total_km2:,.0f} km²); max {MAX_AREA_KM2:,} km².")
    return grid


@app.post("/api/plan")
def plan(req: PlanRequest):
    _analysis_grid(req)
    brief = (
        f"{req.brief}\n\nSelected area (WGS84 longitude/latitude): "
        f"west={req.west}, south={req.south}, east={req.east}, north={req.north}."
    )
    try:
        draft, model = generate_plan(brief)
    except (OSError, ValueError, HTTPClientException) as exc:
        raise HTTPException(502, f"Gemini planning error: {exc}") from exc
    return {"plan": draft.model_dump(), "model": model}


@app.post("/api/conversation")
def conversation(req: ConversationRequest):
    try:
        reply = reply_to_issue_conversation(
            [message.model_dump() for message in req.messages],
            summarize=req.summarize,
        )
    except (OSError, ValueError, HTTPClientException) as exc:
        raise HTTPException(502, f"Gemini conversation error: {exc}") from exc
    return {"reply": reply}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
