"""Run the saved Scotland screening plan without LM Studio."""
from pathlib import Path
import sys

from app.run_plan import main


if __name__ == "__main__":
    plan_file = Path(__file__).resolve().parent / "nbs-run-20261022-141138" / "plan.json"
    raise SystemExit(main(["--plan-file", str(plan_file), *sys.argv[1:]]))
