# NBS Screening Map

Start the app, open http://127.0.0.1:8000, and:

1. Search for a place or draw/edit a rectangular area on the map.
2. Enter the NBS brief and optionally change the resolution and dates. Tick
   **Diagnostic mode** to inspect how the plan was applied (see below).
3. Click **Generate screening plan**. Gemini 3.1 Flash-Lite proposes the required layers
   and criteria using the existing planner.
4. Review the criteria, assumptions, unanswered questions and limitations, then click
   **Approve plan and run screening**.
5. The backend downloads Earth Engine layers and applies the same screening operations
   as the existing command-line runner. Results appear as map overlays and area statistics;
   selected polygons can be downloaded as GeoJSON.

Plans without soft scores are labelled as binary eligibility screens, not rankings.
Selected areas are exactly the pixels that pass all of the plan's hard criteria (the
limits proposed by Gemini and reviewed by you); there is no extra score cutoff. Soft
scores only rank pixels inside the selected areas.

**Diagnostic mode** adds, for the approved plan, an "Input" overlay for each downloaded
layer and a "Criterion N" overlay for each criterion (green = pixels passing that
criterion on its own; score criteria show their 0–1 score), plus a table of the area
passing each criterion. Use it to check that Gemini picked sensible layers and limits.
Editing the brief, area or parameters requires generating and reviewing a new plan.
The web workflow does not save GeoTIFFs or run directories; use the command-line runner below
if those are needed. `/api/analyze` only accepts a reviewed plan; the old fixed-cutoff
layer combiner and `/api/layers` have been removed.

Available layers: slope, elevation, aspect (Copernicus DEM GLO-30), NDVI, NDWI
(Sentinel-2 SR cloud-masked median composite), and ESA WorldCover 2021 land cover.

## Gemini recipe planner

The planner proposes an NBS screening plan. The separate plan runner can download the
required Earth Engine rasters, apply the plan's hard masks and weighted scores, and save
georeferenced screening outputs. Neither workflow certifies engineering/ecological feasibility.

1. Set the `GEMINI_API_KEY` environment variable before starting the app. The default
   `GEMINI_MODEL` is `gemini-3.1-flash-lite`; set `GEMINI_MODEL` to override it.
   Keep the key in an environment variable or hosting-provider secret, never in source
   control. The key stays on the backend; it is not sent to the browser.
2. With the project environment activated, run:

```powershell
python -m app.planner "I need to stabilise a slope in a cold environment with little light."
```

With no brief argument, the script prompts for one; Enter uses the example above.
You can override the Gemini model or choose a different output path:

```powershell
python -m app.planner "Cold slope with little light" --model "your-model-id" --output "my-plan.json"
python -m app.planner --list-catalogues
```

You can also run `python .\app\planner.py`. `--timeout` is in seconds (default 120).
No Earth Engine credentials are needed for planning. The brief, catalogues and output schema
are sent to Google's Gemini API over HTTPS; the web workflow also includes the selected bounds.
Raster data and Earth Engine credentials are not sent to Gemini. API usage may incur charges.
LM Studio is no longer used.

If Gemini times out, add `--timeout 600` to either planner command
to allow a slower response. This option applies only to Gemini, not Earth Engine
downloads. The plan runner prints separate generation, download and processing stages.

The lookup tables live in [app/catalogue.py](app/catalogue.py):

- Layers: slope, aspect, elevation, NDVI, NDWI and land cover, with sources, units,
  resolutions, limitations and class names.
- Operations: `keep_range`, `keep_direction`, `exclude_classes`, `score_linear`.
  The plan runner applies these as hard masks and weighted scores to aligned rasters.
- Combination: AND all masks, weighted mean of optional scores, missing required data
  remains unknown. This is separate from the existing web combiner's min-max workflow.

[app/planner.py](app/planner.py) sends the catalogues and a JSON schema to Gemini.
It validates the response against allowed layers, operation types, class names, numeric
domains and parameter bounds. Invalid, incomplete or failed responses produce an error,
not a fallback plan. The model must support JSON-schema structured output.

Review the printed criteria, assumptions, unanswered questions and limitations.
For example, the model should ask for location/hemisphere rather than silently requiring
south-facing slopes. Thresholds not supplied by the user must be marked provisional.
Schema validation cannot prove that the model's ecological reasoning is correct.

Only `y` or `yes` at the approval prompt saves the draft (default `nbs-plan.json`).
An existing file is never overwritten. The record includes the brief, model ID, approval
timestamp, plan, combination rules and catalogue snapshot. Approval means **draft approved
for further review**, not approval to construct, execute GEE operations or treat missing
site information as resolved. Any other answer declines without saving.

## Execute a screening plan

Run the end-to-end prototype with a site brief and the centre coordinates of a rural
10 km × 10 km test square. The runner asks for review and approval before downloading
anything. If the coordinates are omitted, it prompts for them:

```powershell
python -m app.run_plan "Identify candidate areas for native woodland restoration"
```

Alternatively, provide the centre as decimal longitude/latitude:

```powershell
python -m app.run_plan "Identify candidate areas for native woodland restoration" `
  --center-lon -4.0 --center-lat 56.5 --size-km 10
```

The example coordinates are only a usage example; choose the centre of the test area
yourself. Or pass an explicit `--bbox WEST SOUTH EAST NORTH`. The default Sentinel-2
date range is the previous 365 days; `--start-date` and `--end-date` override it.
`--resolution-m` defaults to 30 m. Selection is defined only by the plan's hard criteria;
there is no additional score cutoff, and soft scores only rank the selected pixels.

After you approve the generated plan, the runner creates a new timestamped output
directory (or the path supplied with `--output-dir`) containing:

- `plan.json`: the brief, generated plan, dates and bounds.
- `suitability.tif`: 0–1 weighted score, with ineligible or unknown pixels as NoData.
- `eligible.tif` and `selected.tif`: hard-mask eligibility (identical selection).
- `selected.geojson`: polygons of selected pixels.
- `report.json`: plan, layer/score statistics, area totals, data dates and limitations.

Open the GeoTIFFs in QGIS or another GIS. The grid uses EPSG:3857 and may be coarsened
to keep each dimension at or below 1024 pixels. The runner requires a Gemini API key, Earth
Engine authentication and `EE_PROJECT` as described below. Plan criteria only use
available proxy layers; verify assumptions, source data, land access and local
constraints with qualified practitioners before using the results.
If a plan contains no `score_linear` criteria, the report marks it `mask_only`:
eligible cells are not ranked, and the selected raster is not a comparative suitability map.

### Test a saved plan without Gemini

The root script `conenction_test.py` loads `nbs-run-20261022-125628/plan.json`,
including its bounds and dates, and runs the same Earth Engine
download and processing flow after confirmation:

```powershell
python .\conenction_test.py
```

Copy the project and that saved-plan directory to the other computer, install
`requirements.txt` in its own Python environment, and authenticate Earth Engine there.
Do not copy the old virtual environment or authentication credentials. A Gemini key is
not required for a saved plan. Results go into a new timestamped directory, leaving the source untouched.
You can choose a new directory with `--output-dir connection-results`.
The equivalent general-purpose command is:

```powershell
python -m app.run_plan --plan-file .\nbs-run-20261022-125628\plan.json
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
earthengine authenticate                 # one-off, opens browser
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

The default Earth Engine Cloud project (`exeter-gee-research`) is stored in
[app/config.py](app/config.py), so it does not need to be set for each run. To override
it for a session, set `$env:EE_PROJECT = "your-other-project-id"` before starting
the app or plan runner.

Service account alternative: set `EE_SERVICE_ACCOUNT` (email) and `EE_SERVICE_ACCOUNT_KEY` (path to JSON key).

## Deploy on Render

The repository includes a Render Blueprint in `render.yaml`. Create a new Blueprint
instance from the GitHub repository and provide `GEMINI_API_KEY` when prompted. Keep
Earth Engine credentials out of GitHub: add the service-account JSON as a Render secret
file, then set `EE_SERVICE_ACCOUNT` to its service-account email and
`EE_SERVICE_ACCOUNT_KEY` to the secret file's path. The service account must be enabled
for the Earth Engine project configured by `EE_PROJECT`.

The Blueprint uses Render's free web-service plan, which may sleep when idle and has
limits on outbound traffic. Gemini and Earth Engine usage are separate from hosting.
Restrict access to the service while prototyping; its API endpoints can trigger those
external services.

### Windows: `_curses` error when importing `ee`

If the traceback shows `ee` importing `blessings` and then `curses`, Python is loading
an unrelated package named `ee`, not Google's `earthengine-api`. Do not install
`curses` to work around this. Use the project environment, which installs the correct
package from `requirements.txt`:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys, ee; print(sys.executable); print(ee.__file__); print(ee.Initialize)"
```

Alternatively, run the planner explicitly with `.\.venv\Scripts\python.exe -m app.run_plan`
to bypass a different `python` on PATH. In VS Code, select
`.venv\Scripts\python.exe` using **Python: Select Interpreter**.

## Tests

```powershell
pytest
```

## Notes

- Resolution auto-coarsens so the grid is at most 1024×1024 px; max area 25,000 km².
- Pixels with no data in a required layer are excluded. Plans use the criteria's physical
  units and explicit increasing/decreasing preferences, without local min-max normalisation.
