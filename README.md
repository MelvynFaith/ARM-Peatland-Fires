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

## Week 11 update: preprocessing, mining and evaluation

The progress report from the start to week 11 is [docs/REPORT.md](docs/REPORT.md). Three scripts were added after `build_dataset.py`:

| Script | Step | Output |
|---|---|---|
| `src/items.py` | Preprocessing: numeric features become 25 category items (bin edges from 2012-2019 only); seasonal history; one transaction per cell-week | `data/transactions.parquet`, `results/item_support.csv`, `results/data_overview.csv`, `results/bin_edges.json` |
| `src/mine.py` | Mining: Apriori and FP-Growth (`mlxtend`) and ECLAT (own code), same itemsets checked; run time and memory at min support 5%, 2%, 1%, 0.5%; class rules `X -> API_MINGGU_DEPAN` for all cells (`fire`) and quiet cells (`quiet`) | `results/benchmark.csv`, `results/benchmark.png`, `results/rules_fire.csv`, `results/rules_quiet.csv` |
| `src/evaluate_rules.py` | Evaluation on 2020-2026: holdout lift, lift per year, block-bootstrap intervals, El Niño contrast, rule card versus baselines, decision tree comparison | `results/eval_rules_*.csv`, `results/rule_card_*.csv`, `results/card_vs_baselines.csv`, `results/summary.md`, plots |

### Setup (Windows, PowerShell)

1. Get a free FIRMS map key at <https://firms.modaps.eosdis.nasa.gov/api/map_key/>. Enter your email and the 32-character key arrives by email.
2. Copy `.env.example` to `.env` in the project root and paste the key after `FIRMS_MAP_KEY=` (no quotes, no spaces). `src/config.py` reads it, so every script sees it. Never commit `.env`.
3. Create the environment and run everything:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd src
..\.venv\Scripts\python download_misc.py
..\.venv\Scripts\python download_chirps.py
..\.venv\Scripts\python download_firms.py
..\.venv\Scripts\python build_dataset.py
..\.venv\Scripts\python items.py
..\.venv\Scripts\python mine.py
..\.venv\Scripts\python evaluate_rules.py
```

The downloads resume from their caches when rerun. The week 11 fixes to the pipeline (CHIRPS URL and rate limit, Windows encoding, `.env`) are listed at the end of [docs/DATA.md](docs/DATA.md).

### How to check that preprocessing, mining and evaluation work

**1. Without real data (about 2 minutes): smoke test.** Builds a synthetic `samples.parquet` with one planted pattern ("burned last week and 14+ dry days, then fire is likely") in a temporary folder and runs `items.py`, `mine.py` and `evaluate_rules.py` on it. Your real `data/` and `results/` are not touched.

```
.venv\Scripts\python tests\smoke_test.py
```

It must end with `SMOKE TEST PASSED` and print the planted rule with lift above 1.5 on both 2012-2019 and 2020-2026. It fails (with an `AssertionError`) if rows are lost in preprocessing, if Apriori, FP-Growth and ECLAT disagree, if the planted rule is not found or not stable, or if a result file is missing. The temporary results folder is printed at the end, so you can open the files.

**2. With real data: what each step prints and writes.**

| Step | Look at the terminal | Open these files |
|---|---|---|
| `build_dataset.py` | `(286622, …) positive rate 0.21` or close (the earlier project's run) | `data/samples.parquet` |
| `items.py` | missing values (expected none), rows kept, data overview, bin edges, item support | `results/data_overview.csv` (row count, cells, provinces, base rates), `results/item_support.csv` (support and single-item lift of all 25 items), `results/bin_edges.json` |
| `mine.py` | run time and memory per algorithm and minimum support; `same_itemsets True` for every row; rule counts and top 15 rules | `results/benchmark.png`, `results/benchmark.csv`, `results/algorithm_check.csv` (must be all `True`), `results/rules_fire.csv`, `results/rules_quiet.csv` (with a plain-language `sentence` column) |
| `evaluate_rules.py` | the full summary (also saved as markdown) | `results/summary.md` (start here), `results/rule_card_fire.csv`, `results/rule_card_quiet.csv`, `results/card_vs_baselines.csv`, `results/lift_train_vs_holdout.png`, `results/lift_per_year_card.png`, `results/tree_fire.txt` |

What a good result looks like, following the BOD:

- `algorithm_check.csv`: every row `same_itemsets = True` and `max_support_difference = 0`.
- `rules_fire.csv`: all rules have `lift >= 1.5` and `support >= 0.01`.
- `eval_rules_fire.csv`: compare `lift_2012-2019` with `lift_2020-2026`. The column `stable` is `True` when the holdout lift stays at least 1.5, the lower end of its bootstrap interval is above 1, and the lift is at least 1 in all holdout years but one.
- `card_vs_baselines.csv`: the rule card should have a higher precision (and lift) than `Random` and than the single-item baselines at a similar share of flagged cell-weeks.

The CSV files open in Excel. The `.png` files are the charts for the report.
