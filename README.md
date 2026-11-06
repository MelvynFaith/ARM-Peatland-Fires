# ARM Peatland Fires

Data Mining course project: association rule mining on weekly peatland fire data from Sumatra and Kalimantan, Indonesia.

**Working title:** Association Rule Mining of Conditions Preceding New Ignitions and Fire Extinction in Indonesian Peatlands Using Apriori and FP-Growth

## Idea in short

Each transaction is one peat cell (0.25°, about 28 km) in one week, from 2012 to 2026. Its items describe rainfall, dry spells, nearby fires, peat, season and El Niño. Two rule sets are mined for two field decisions:

- **Ignition rules** on cells with no hotspot in the last 30 days. They show which conditions come before a new fire next week, to guide prevention patrols.
- **Extinction rules** on cells that are burning now. They show which conditions come before the fire going out, to guide when crews can move.

Rules are mined on 2012 to 2019 and tested on 2023 to 2026. They must beat a simple baseline. The final output is a short rule card that officers can check from free weekly data.

## Documents

| File | Contents |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | Course requirements, decisions made, next steps, open questions. Start here. |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Literature, novelty, BOD and the research design for mining and evaluation |
| [docs/DATA.md](docs/DATA.md) | Sources, pipeline, cleaning rules, column meanings, known issues |
| `CLAUDE.md` | Git authorship rules for AI-assisted commits |

## Layout

```
src/config.py           settings and year splits
src/download_*.py       data downloads
src/regions.py          country, province and island for each cell
src/build_dataset.py    weekly cell table: data/samples.parquet
data/                   downloaded and built data (not in git)
docs/                   documentation
```

## Data sources

| Data | Source |
|---|---|
| Hotspots | NASA FIRMS, VIIRS S-NPP 375 m (free map key needed) |
| Rainfall | CHIRPS v2.0 daily |
| Peat map | Peat-ML v0.9 (Zenodo, CC BY 4.0) |
| El Niño index | NOAA ONI |
| Boundaries | Natural Earth |

The data pipeline in `src/` (downloads, `regions.py`, `build_dataset.py`) is copied unchanged from the team's earlier competition project, SIGAP Gambut (commit `9870036`). Only `config.py` was trimmed and given the rule-mining splits. Raw and processed data are not stored in git.

## Building the dataset

Needs Python 3.10 or later and a free [FIRMS map key](https://firms.modaps.eosdis.nasa.gov/api/map_key/) in the `FIRMS_MAP_KEY` environment variable. The downloads take about an hour and fill `data/` (about 450 MB).

```
pip install -r requirements.txt
cd src
python download_misc.py
python download_chirps.py
python download_firms.py
python build_dataset.py
```

`build_dataset.py` writes `data/samples.parquet`: one row per peat cell per week, with rainfall, dry-spell, hotspot-history, peat, season and El Niño features, and the label for the next 7 days.

## Timeline

| Week | Work | Status |
|---|---|---|
| 10 | Dataset, data characteristics, BOD, algorithms | Done |
| 11 to 12 | Code: preprocessing, mining, evaluation | Next |
| 13 | Report 1 | |
| 14 | Extra experiments (if requested) | |
| 15 | IMRAD article (Indonesian) | |
| 16 | Presentation | |
