# Data

How the dataset is built, what each column means, and the known caveats. The code is in `src/`. It was copied from the team's earlier competition project, SIGAP Gambut. That repository (`MelvynFaith/Larper-IRSDC2026`) is private, so everything needed is described here.

## Sources

| Data | What is used | Script | Access |
|---|---|---|---|
| NASA FIRMS hotspots | VIIRS S-NPP 375 m. Standard processing (SP) archive to `SP_END`, then near-real-time (NRT) to `END` | `download_firms.py` | Free map key in `FIRMS_MAP_KEY` |
| CHIRPS v2.0 rainfall | Daily 0.05°, averaged to 0.25°. Final data per year, then preliminary daily files | `download_chirps.py` | Open |
| Peat-ML v0.9 | Global peatland probability map, Zenodo record 7352284, CC BY 4.0 | `download_misc.py` | Open |
| NOAA ONI | El Niño index, 3-month seasons | `download_misc.py` | Open |
| Natural Earth 1:10m | Country and province boundaries, public domain | `download_misc.py` | Open |

## Pipeline

Run from `src/` in this order. Times are from the earlier project.

1. `download_misc.py`: ONI, Peat-ML and boundaries. A few minutes.
2. `download_chirps.py`: rainfall 2011 to `END`, cached per year in `data/chirps/`. About 30 minutes. Reruns resume.
3. `download_firms.py`: hotspots in 5-day windows, cached in `data/firms_cache/`. 20 to 40 minutes. Reruns resume.
4. `build_dataset.py`: builds the weekly table. A few minutes.

`regions.py` is called by `build_dataset.py`. It assigns each cell a country, province and island, and caches the result in `data/cells.parquet`.

## Study area and sample

- Grid: 0.25° cells (about 28 km, about 770 km²) inside the box 95°E to 119°E and 6°S to 6°N.
- Cells kept:
  - mean Peat-ML probability of at least 0.10;
  - on land;
  - in Indonesian Sumatra or Kalimantan.

  This leaves 413 cells in 13 provinces. 73 peat cells in Malaysia and Brunei are dropped.
- One sample is one cell on one week start day `t`. Samples are taken every 7 days, starting 90 days after `START` so every feature has history.
- Features use data up to and including day `t`. The label covers days `t+1` to `t+7`. Feature and label windows never overlap.
- Size in the earlier project's last run: 286,622 rows from April 2012 to October 2026. About 21% of rows are positive (fire next week).

## Cleaning rules

- **Fires only.** SP hotspots keep FIRMS `type` 0 (vegetation fire). Types 1 to 3 (volcano, static land source such as a gas flare, offshore platform) are dropped. NRT data has no `type`. NRT hotspots are dropped when they lie within about 1 km of a spot the SP archive ever flagged as type 1 to 3. Tested on 2025 SP data, this removed 97% of non-fire sources and 4% of real fires.
- **Confidence.** Low-confidence (`l`) hotspots are dropped. Nominal and high are kept.
- **Satellite outages.** A day with no detection of any kind in the whole box counts as an outage, not a fire-free day. A week is dropped when:
  - an outage falls in its label window or its last 14 days; or
  - more than 2 outages fall in its last 30 days.

  Outages affect, among others, late July to mid August 2022, July 2023, and many weeks in 2024 to 2026. Only about 30 weeks of 2024 and 23 weeks of 2026 remain.
- **El Niño index lag.** ONI is taken from the season centred 2 months before `t`, so no future value is used.
- **Rain at the very end.** The last days without published preliminary CHIRPS are filled with each cell's mean of the 7 days before. In the last run this was 26 to 27 October 2026.

## Columns of `data/samples.parquet`

| Column | Meaning |
|---|---|
| `week` | Week start day `t` (features end here) |
| `year` | Year of `week` |
| `j`, `i` | Cell row and column in the grid |
| `island`, `province` | Sumatra or Kalimantan, and province name (Natural Earth spelling) |
| `peat` | Mean Peat-ML peat probability of the cell, 0 to 1 |
| `month_sin`, `month_cos` | Month of `t` as a cycle |
| `oni` | ONI value, 2-month lag |
| `rain_7d` … `rain_90d` | Rain in mm over the last 7, 14, 30, 60 and 90 days |
| `dry_days` | Consecutive days with under 1 mm of rain up to `t`, capped at 120 |
| `hot_7d` … `hot_365d` | Hotspots in the cell over the last 7, 14, 30, 90 and 365 days |
| `hot_nbr_7d`, `hot_nbr_14d`, `hot_nbr_30d` | Hotspots in the 8 neighbouring cells over the last 7, 14 and 30 days |
| `days_since_fire` | Days since the last hotspot in the cell, capped at 1000 |
| `fire_day_share_3y` | Share of the last 3 years' days with at least one hotspot in the cell |
| `y` | Label: 1 if at least one hotspot in days `t+1` to `t+7` |
| `hot_next_7d` | Number of hotspots in days `t+1` to `t+7` |
| `quality` | `final`: features and label from archived data only. `nrt`: some near-real-time data |

`data/forecast_features.parquet` has the same features, without labels, for the last day of data and the week before. It is for applying the rule card to the latest week.

## Other files in `data/`

`firms_viirs.parquet` (all hotspots), `chirps_025.nc` and `chirps_prelim_025.nc` (rain on the grid), `peat_ml.nc`, `oni.txt`, `ne_countries.geojson`, `ne_provinces.geojson`, `cells.parquet`, plus the caches `firms_cache/` and `chirps/`.

## Known issues when rebuilding

- **FIRMS quota.** A key allows 5,000 transactions per 10 minutes. When the quota is spent, FIRMS answers `Invalid MAP_KEY` even for a valid key. The script then waits for the quota to clear and continues. The SP source accepts at most 5 days per request.
- **NRT versus SP.** FIRMS moves near-real-time data into the SP archive a few months later. If the NRT download for August to October 2026 comes back empty, check how far SP now reaches and move `SP_END` in `config.py`. This has not been tested.
- **CHIRPS.** A server error (for example a one-off 403) is usually temporary. Rerun and the cache resumes.
- **Missing variables.** There is no temperature, humidity, wind, land cover or human activity data. Rules can only use rain, fire history, peat, season, place and El Niño.
