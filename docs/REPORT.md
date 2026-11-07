# Progress report (weeks 10 to 11)

**Association Rule Mining of Conditions Preceding Peatland Fires in Sumatra and Kalimantan Using Apriori and FP-Growth**

Data Mining course project, topic association rule mining (ARM). This report covers the project from the start (week 10: data, data characteristics, BOD, algorithms) to the end of week 11 (code for preprocessing, mining and evaluation, and the first results). Supporting documents: [RESEARCH.md](RESEARCH.md) (literature and novelty), [DATA.md](DATA.md) (data pipeline), [STATUS.md](STATUS.md) (course plan and decisions).

Every number in sections 7 and 8 is produced by the code in `src/` and can be found in `results/` (section 10 shows how to rerun it).

## 1. Background

Peatland fires in Sumatra and Kalimantan return every dry season. The 2015 crisis cost Indonesia US$16.1 billion and burned 2.6 million hectares (World Bank, 2016). The agencies that respond (BPBD, Manggala Agni, plantation managers) have limited crews. Their current tools mostly show fires that are already burning: hotspot maps such as SiPongi and NASA FIRMS. BMKG's fire-weather indices rate how easily fire could start from weather alone.

Association rule mining fits this problem because its output is a list of if-then statements. An officer can read a rule such as "if the cell burned last week and there has been a long dry spell, fire next week is likely" and check it from free data, without running a model.

## 2. Business Objective Determination (BOD)

**Business objective.** Help fire agencies understand which combinations of conditions most often come right before a peat fire, so they can prioritise patrols and prevention before fires start or spread.

**Business questions.**

1. Which combinations of dryness, recent fire activity, season and El Niño come before a fire in the following week?
2. How much more likely is a fire under these combinations than on an average week (lift)?
3. Do the patterns stay the same over time, or only in some years?
4. Are there patterns that come before fires in areas that have been quiet for a month (new ignitions)?

**Business success criteria.**

- The rules are easy to read as plain if-then statements an officer can act on, for example "if burned last week and a long dry spell, fire is likely".
- The top rules point to conditions that can be checked from free data every week.

**Data mining goal.** Mine association rules that end in "fire next week", from the weekly peat-cell transactions (2012 to 2026), using Apriori and FP-Growth, and compare the two algorithms.

**Data mining success criteria.**

- Rules with lift clearly above 1 (at least 1.5) and enough support to matter (at least 1% of transactions).
- Rules found on 2012 to 2019 keep a similar lift on 2020 to 2026.
- Apriori and FP-Growth find the same rules, with their running time compared.

How the questions map to the work:

| Question | Where it is answered |
|---|---|
| 1. Which combinations | Rule set `fire` (section 7.3), rule card (7.5) |
| 2. How much more likely | Lift of each rule, train and holdout (7.3, 7.4) |
| 3. Stable over time | Lift per year and holdout lift with a bootstrap interval (7.4) |
| 4. New ignitions | Rule set `quiet`, mined only on cells with no hotspot for 30 days (7.6) |
| Algorithm comparison | Apriori, FP-Growth and ECLAT, same itemsets, run time and memory (7.2) |

Note. RESEARCH.md (week 10) also proposed a second rule set for fire extinction. It is not part of the BOD above, so it is not in this report. It can be added as an extra experiment in week 14.

## 3. Data (week 10)

### 3.1 Sources

All data is public and free. No Kaggle dataset is used.

| Data | Source | What is used |
|---|---|---|
| Hotspots | NASA FIRMS, VIIRS S-NPP 375 m, <https://firms.modaps.eosdis.nasa.gov/> | Vegetation-fire detections, archived (SP) data to 31 July 2026, near-real-time (NRT) after that |
| Rainfall | CHIRPS v2.0 daily, <https://www.chc.ucsb.edu/data/chirps> | Daily rain at 0.05°, averaged to 0.25° cells |
| Peat map | Peat-ML v0.9, CC BY 4.0, <https://zenodo.org/records/7352284> (DOI 10.5281/zenodo.7352284) | Peat probability per cell |
| El Niño index | NOAA ONI, <https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/> | 3-month ONI value, taken 2 months before the week |
| Boundaries | Natural Earth, public domain, <https://github.com/nvkelso/natural-earth-vector> | Country, province and island of each cell |

The download and dataset scripts (`src/download_*.py`, `src/regions.py`, `src/build_dataset.py`) come from the team's earlier competition project, SIGAP Gambut. DATA.md explains every step.

### 3.2 Transactions

- **One transaction = one peat cell in one week.** A cell is 0.25° (about 28 km). A cell is kept when its mean Peat-ML probability is at least 0.10 and it lies in Indonesian Sumatra or Kalimantan. That leaves 413 cells in 13 provinces.
- **Lead time.** The features of a transaction use data up to and including the week start day `t`. The label "fire next week" covers days `t+1` to `t+7`. The two windows never overlap, so every rule says what comes *before* a fire.
- **Period.** April 2012 to October 2026, one transaction per cell every 7 days.

### 3.3 Data characteristics

- **Numeric raw data.** Rainfall (mm), dry-day counts, hotspot counts, peat probability and the ONI index are all numbers. ARM needs items, so each feature is turned into category items (section 4). There are 25 items in total, including the consequent.
- **Imbalanced.** Only about one transaction in five has a fire the following week. This is why rules are judged by lift (how much the fire rate rises above the base rate) and not by confidence alone.
- **Satellite outages removed.** A day with no VIIRS detection anywhere in the study box is an outage, not a fire-free day. Weeks touched by outages are dropped (rules in DATA.md).
- **Not independent.** Neighbouring cells and consecutive weeks are related. With hundreds of thousands of transactions almost any rule would get a tiny p-value, so significance tests would mislead. Rules are judged on later years they were not mined from instead (section 6).
- **Missing variables.** There is no temperature, humidity, wind, land use or human activity. Rules can only use rain, fire history, peat, season, place and El Niño.

The exact counts from this run are in section 7.1.

## 4. Preprocessing (week 11) — `src/items.py`

`build_dataset.py` already does the heavy cleaning: vegetation fires only (FIRMS type 0, no low-confidence detections, no gas flares or volcanoes), Indonesian cells only, outage weeks dropped. `items.py` turns that table into transactions in five steps.

1. **Checks.** Drop rows with missing feature values and duplicate (week, cell) rows. Both counts are printed. Neither is expected, because `build_dataset.py` already fills or drops them.
2. **Seasonal history.** For each cell and calendar month, the code marks whether the cell had a hotspot in that month in any **earlier** year. Only past years are used, so no future information enters.
3. **Bin edges.** Physical thresholds are used where one exists. Where none exists, the edge is a quantile computed **only on 2012 to 2019**, so the holdout years cannot influence the items. The edges are saved in `results/bin_edges.json`.
4. **Items.** Each feature becomes one or more boolean items (table below). Items in the same group are exclusive bins: every transaction has exactly one rain-30 item, one season item, one ENSO item, one fire-history state and one island. The code checks this with assertions.
5. **Output.** `data/transactions.parquet` (one boolean column per item, plus week, year, cell, province and label), `results/item_support.csv` (support of every item and its single-item lift), `results/data_overview.csv`.

### 4.1 Item catalogue

| Group | Item | True when | Edge type |
|---|---|---|---|
| Rain | `HUJAN30_SANGAT_RENDAH` | rain in the last 30 days < 50 mm | physical (under about 1.7 mm/day) |
| | `HUJAN30_RENDAH` | 50 to 120 mm in 30 days | physical (120 mm = about 4 mm/day, below which severe burning becomes likely, Field et al.) |
| | `HUJAN30_NORMAL` | ≥ 120 mm in 30 days | physical |
| | `HUJAN7_SANGAT_RENDAH` | rain in the last 7 days < 5 mm | physical |
| | `HUJAN90_RENDAH` | 90-day rain in the driest 25% | quantile, 2012-2019 |
| Dry spell | `KERING_7_13HARI` | 7 to 13 consecutive days under 1 mm | physical |
| | `KERING_PANJANG` | 14 or more consecutive days under 1 mm | physical |
| Fire history (cell) | `TERBAKAR_7HARI` | at least one hotspot in the cell in the last 7 days | any hotspot |
| | `TERBAKAR_8_30HARI` | no hotspot in the last 7 days, but one in days 8 to 30 | any hotspot |
| | `SENYAP_30HARI` | no hotspot in the cell for 30 days ("quiet") | any hotspot |
| | `HOTSPOT_7HARI_TINGGI` | hotspot count in the last 7 days at least the median of burning cells | quantile, 2012-2019 |
| | `RIWAYAT_API_TINGGI` | share of fire days in the last 3 years in the top 25% | quantile, 2012-2019 |
| | `PERNAH_TERBAKAR_BULAN_INI` | the cell burned in this calendar month in an earlier year | past years only |
| Fire nearby | `TETANGGA_TERBAKAR_7HARI` | a hotspot in one of the 8 neighbouring cells in the last 7 days | any hotspot |
| | `TETANGGA_TERBAKAR_30HARI` | neighbour hotspot in days 8 to 30, none in the last 7 | any hotspot |
| Peat | `GAMBUT_TINGGI` | Peat-ML probability at least the median of the cells | quantile, 2012-2019 |
| Season | `MUSIM_KEMARAU` | June to October | calendar |
| | `MUSIM_HUJAN` | November to March | calendar |
| | `MUSIM_PERALIHAN` | April and May | calendar |
| El Niño | `EL_NINO` | ONI ≥ +0.5 | NOAA definition |
| | `LA_NINA` | ONI ≤ −0.5 | NOAA definition |
| | `ENSO_NETRAL` | otherwise | NOAA definition |
| Place | `PULAU_SUMATRA`, `PULAU_KALIMANTAN` | island of the cell | |
| Consequent | `API_MINGGU_DEPAN` | at least one hotspot in the cell in days `t+1` to `t+7` (label `y = 1`) | |

Each item also has a plain Indonesian description in `items.py` (`ITEM_TEXT`), so every rule can be printed as a sentence for officers. Provinces are not items, because 13 extra items would make the rules longer and more place-specific. They stay in the transaction file for later breakdowns.

## 5. Mining (week 11) — `src/mine.py`

### 5.1 Measures

For a rule A → `API_MINGGU_DEPAN` (A is the antecedent, a set of items, and N is the number of transactions):

| Measure | Formula | Meaning |
|---|---|---|
| Support | count(A and fire next week) / N | How common the whole pattern is |
| Confidence | count(A and fire next week) / count(A) = P(fire \| A) | Fire rate in the weeks where A holds |
| Lift | confidence / P(fire) | How many times more likely fire is than on an average week. Above 1 means A comes before fire more often than chance |
| Improvement | confidence − the highest confidence of any rule made from a subset of A (the empty set gives P(fire)) | Whether the extra items add anything over a shorter rule |

Only rules whose consequent is `API_MINGGU_DEPAN` are generated (class association rules). Rules such as "El Niño → dry season" would be true but useless for the business question.

### 5.2 Algorithms

All three algorithms answer the same question: which itemsets reach the minimum support? They differ only in how they search.

- **Apriori** (Agrawal and Srikant, 1994; `mlxtend`). Works breadth-first, level by level. Frequent itemsets of size k−1 are joined into candidates of size k. A candidate is dropped at once if any of its subsets is not frequent (the Apriori property: every subset of a frequent itemset is frequent). The remaining candidates are counted with a scan of the transactions. It is simple, but it builds many candidates and scans the data once per level.
- **FP-Growth** (Han, Pei and Yin, 2000; `mlxtend`). Compresses all transactions into one prefix tree, the FP-tree, ordered by item frequency. Itemsets are then grown from each item's conditional tree, without generating candidates and without rescanning the data.
- **ECLAT** (Zaki, 2000; own implementation in `mine.py`). Uses the vertical layout: each item stores the set of transactions that contain it, as a bitset. The support of an itemset is the size of the intersection of its items' bitsets (a bitwise AND, then a bit count). Search is depth-first.

Because the three algorithms are exact, they must return identical itemsets and supports. The code checks this with an assertion at every minimum-support level.

### 5.3 Settings

| Setting | Value | Reason |
|---|---|---|
| Mining years | 2012 to 2019 | BOD. Includes the 2015 El Niño |
| Holdout years | 2020 to 2026 | BOD. Never used for mining or bin edges. Includes the 2023 and 2026 El Niño seasons |
| Maximum itemset size | 5 (at most 4 conditions per rule) | Rules stay readable as one sentence |
| Minimum support, rule set `fire` | 1% | BOD |
| Minimum support, rule set `quiet` | 0.5% of quiet transactions | The subset is smaller and fire is rarer there. Lift is measured against the quiet-cell base rate |
| Minimum lift | 1.5 | BOD |
| Minimum improvement | 0.01 | Drops rules that only add items to a shorter rule without raising confidence by at least 1 point |
| Benchmark minimum supports | 5%, 2%, 1%, 0.5% | Standard comparison of run time and memory |

Two rule sets are mined:

- **`fire`**: all transactions. Answers questions 1 to 3.
- **`quiet`**: only transactions with `SENYAP_30HARI` (no hotspot in the cell for 30 days). The fire-history items that are constant in this subset are removed. Answers question 4 (new ignitions), where the hotspot map is no help.

## 6. Evaluation (week 11) — `src/evaluate_rules.py`

| Level | What is measured |
|---|---|
| Algorithms | Same frequent itemsets and supports from Apriori, FP-Growth and ECLAT; run time (median of 3 runs) and peak memory (`tracemalloc`) at each minimum support |
| Each rule | Support, confidence and lift on 2012-2019 and on 2020-2026 (also 2020-2022 and 2023-2026 separately); lift per year; lift in El Niño years versus other years |
| Uncertainty | 95% interval of the holdout confidence and lift by **block bootstrap over weeks**: 1,000 resamples of whole weeks, so cells of the same week stay together and their dependence is kept |
| Success criteria | `pass_train`: train lift ≥ 1.5 and train support ≥ minimum support. `pass_holdout`: holdout lift ≥ 1.5 and the lower bound of its interval > 1. `stable`: both, and lift ≥ 1 in all holdout years except at most one |
| Rule card | Up to 10 rules picked greedily by train lift. A rule is skipped if it contains an already picked rule or covers almost the same cell-weeks (Jaccard overlap > 0.7). A cell-week is flagged when any card rule matches. Precision, recall and share of flagged cell-weeks on 2020-2026 |
| Baselines | Random flags at the same rate (precision = base rate); the single best rule; a single item an officer already knows (`TERBAKAR_7HARI`, i.e. "it burned last week", and seasonal history) |
| Decision tree | A depth-4 tree on the same items, trained on 2012-2019. Each leaf is also an if-then rule, so its leaves are compared with the ARM rules (same measures, same holdout) |
| NRT check | Holdout lift again on archived (SP, `quality = final`) rows only, to show that the August to October 2026 NRT weeks do not drive the result |

Rules are always selected on 2012-2019 only. The holdout years are used once, to measure.

## 7. Results

### 7.0 Status at the end of week 11

| Part | Status |
|---|---|
| Environment, FIRMS key in `.env` | Done |
| Hotspots (FIRMS) | Downloaded: 3,114,710 detections, 20 January 2012 to 27 October 2026 |
| Peat map, ONI, boundaries, cell regions | Downloaded and built (cells in Indonesian Sumatra and Kalimantan assigned) |
| Rainfall (CHIRPS) | **Blocked.** On 7 November 2026 `data.chc.ucsb.edu` first answered `429 Too Many Requests` (fixed in the code, see DATA.md) and then refused all connections, also from outside our network. The NOAA mirror misses whole years (including 2015), so it cannot replace it. The download resumes from its cache once the server is back |
| Code: preprocessing, mining, evaluation | Done and tested on synthetic data (7.1) |
| Results on the real data | Waiting for CHIRPS. Sections 7.2 onwards are filled by rerunning `items.py`, `mine.py` and `evaluate_rules.py`. All tables come out in `results/summary.md` |

CHIRPS v3.0 was considered as a fallback. It is on the same server, so it was blocked too. It also builds daily rain differently from v2.0, so the dry-spell items would change. CHC states that CHIRPS v2.0 is produced until December 2026, so this project stays on v2.0.

### 7.1 Verification of the code (smoke test)

`tests/smoke_test.py` builds a synthetic table with the same columns as `samples.parquet` (36 cells, April 2012 to October 2026, 27,216 transactions). One pattern is planted: a cell that burned last week during 14 or more dry days burns next week with probability 0.75, against 0.35 for burned cells otherwise and 0.08 for the rest. The three scripts then run on it unchanged.

| Check | Result |
|---|---|
| Rows kept by preprocessing; every exclusive item group has exactly one true item | Passed |
| Apriori, FP-Growth and ECLAT: identical itemsets and supports at 5%, 2%, 1% and 0.5% | Passed (1,922 / 7,005 / 13,699 / 21,583 itemsets) |
| The same class rules from all three algorithms | Passed |
| Planted rule found | `HUJAN30_NORMAL & HUJAN7_SANGAT_RENDAH & KERING_PANJANG & TERBAKAR_7HARI → API_MINGGU_DEPAN`, lift 3.17 on 2012-2019 and 3.01 on 2020-2026 |
| All result files written | Passed |

On synthetic data the rule found is longer than the planted one, because random items such as `HUJAN30_NORMAL` add noise. The core `KERING_PANJANG & TERBAKAR_7HARI` is the part that carries the lift. This is the expected behaviour of the improvement filter on random data, not a problem with the code.

**Memory.** With its default settings, mlxtend Apriori built one large candidate array: about 1.5 GB at 0.5% support on only 14,000 rows. The laptop has 7.4 GB of RAM. `mine.py` therefore runs Apriori with `low_memory=True`. On a synthetic table of 231,552 rows (about the size of the real 2012-2019 data) it then took 3.7 s and 8 MB at 5% support, and 22.9 s and 28 MB at 0.5%. This is already a first algorithm-comparison finding: Apriori's candidate generation is the costly step that FP-Growth and ECLAT avoid.

### 7.2 to 7.6 Real-data results

To be filled after the full run: data overview (7.2), algorithm comparison (7.3), rule set `fire` (7.4), stability over time and El Niño (7.5), rule card versus baselines and the decision tree (7.6), and rule set `quiet` for new ignitions.

## 8. Limitations

- **Associations, not causes.** A rule says a condition *comes before* fire more often than chance. Most peat fires are lit by people, and the data has no human, land-use, wind or humidity variables.
- **Fire history dominates.** Rules with `TERBAKAR_7HARI` are expected to be the strongest, and a hotspot map already shows them. This is why the `quiet` rule set exists (question 4) and why the card is compared with the single item `TERBAKAR_7HARI`.
- **Base rate changes between periods.** Satellite outages remove many weeks in 2024 to 2026, and the fire rate differs between years. Lift is always computed against the base rate of the same period, so rules stay comparable.
- **Bins are a choice.** Physical thresholds and 2012-2019 quantiles were fixed before looking at the holdout years. Other bins could give other rules. The validation years 2020 to 2022 can be used to test alternatives without touching 2023 to 2026.
- **NRT data at the very end.** August to October 2026 uses near-real-time hotspots with an approximate gas-flare filter. The evaluation also reports holdout lift on archived rows only.

## 9. Next steps

| Week | Work |
|---|---|
| 11 (rest) | When CHIRPS is reachable: `download_chirps.py`, `build_dataset.py`, then `items.py`, `mine.py`, `evaluate_rules.py`. Check the row count against 286,622 and fill sections 7.2 to 7.6 from `results/summary.md` |
| 12 | Discuss the rule card. Check the bins on 2020-2022. Optionally apply the card to the latest week (`data/forecast_features.parquet`) as a demo |
| 13 | Report 1 |
| 14 | Extra experiments if requested: extinction rules, CBA-style classifier, ERA5 humidity |
| 15 | IMRAD article in Indonesian |
| 16 | Presentation |

## 10. How to reproduce

See the README section "Week 11 update". In short: put the FIRMS key in `.env`, install `requirements.txt` in a Python 3.12 virtual environment, run the download scripts and `build_dataset.py`, then `items.py`, `mine.py` and `evaluate_rules.py` from `src/`. Run `tests/smoke_test.py` to check the code without any download.

## References

- Agrawal, R. and Srikant, R. (1994). Fast algorithms for mining association rules. VLDB.
- Han, J., Pei, J. and Yin, Y. (2000). Mining frequent patterns without candidate generation. SIGMOD.
- Zaki, M. J. (2000). Scalable algorithms for association mining. IEEE TKDE 12(3).
- Field, R. D. et al. (2016). Indonesian fire activity and smoke pollution in 2015 show persistent nonlinear sensitivity to El Niño-induced drought. PNAS.
- World Bank (2016). The cost of fire: an economic analysis of Indonesia's 2015 fire crisis.
- Funk, C. et al. (2015). The climate hazards infrared precipitation with stations (CHIRPS). Scientific Data 2, 150066.
- Melton, J. R. et al. (2022). Peat-ML: a global map of peatland extent (v0.9), Zenodo, doi:10.5281/zenodo.7352284.
- Kralj Novak, P., Lavrač, N. and Webb, G. I. (2009). Supervised descriptive rule discovery. JMLR 10.
- Literature on ARM for Indonesian fires: see RESEARCH.md, section 1.
