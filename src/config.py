"""Shared settings. Study area: Indonesian peat cells in Sumatra and Kalimantan, 0.25 degree cells, weekly samples."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from the project .env (for FIRMS_MAP_KEY). A variable already set in the shell wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

# lon_min, lat_min, lon_max, lat_max
BBOX = (95.0, -6.0, 119.0, 6.0)
CELL = 0.25
HORIZON_DAYS = 7
START = "2012-01-20"   # first day of VIIRS SNPP 375 m data in FIRMS
END = "2026-10-27"     # last day of hotspot data used
SP_END = "2026-07-31"  # last day of the archived (standard processing) hotspots; near-real-time after this
PEAT_MIN_FRAC = 0.10   # keep cells whose mean Peat-ML peat probability is at least this
EL_NINO_YEARS = [2015, 2023, 2026]  # fire-season El Nino: ONI for Jul-Sep at least +0.5 (2019 was +0.20)
SEED = 42

# Association rule mining splits (docs/RESEARCH.md, section 5.3)
MINE_YEARS = list(range(2012, 2020))   # rules and bin edges are learned here only
VALID_YEARS = [2020, 2021, 2022]       # choose bins and thresholds
TEST_YEARS = [2023, 2024, 2025, 2026]  # used once, at the end
