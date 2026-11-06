# ARM Peatland Fires

Data Mining course project: association rule mining on weekly peatland fire data from Sumatra and Kalimantan, Indonesia.

**Working title:** Association Rule Mining of Conditions Preceding New Ignitions and Fire Extinction in Indonesian Peatlands Using Apriori and FP-Growth

## Idea in short

Each transaction is one peat cell (0.25°, about 28 km) in one week, from 2012 to 2026. Its items describe rainfall, dry spells, nearby fires, peat, season and El Niño. Two rule sets are mined for two field decisions:

- **Ignition rules** on cells with no hotspot in the last 30 days. They show which conditions come before a new fire next week, to guide prevention patrols.
- **Extinction rules** on cells that are burning now. They show which conditions come before the fire going out, to guide when crews can move.

Rules are mined on 2012 to 2019 and tested on 2023 to 2026. They must beat a simple baseline. The final output is a short rule card that officers can check from free weekly data.

The research, the novelty and the Business Objective Determination are in [docs/RESEARCH.md](docs/RESEARCH.md).

## Data sources

| Data | Source |
|---|---|
| Hotspots | NASA FIRMS, VIIRS S-NPP 375 m (free map key needed) |
| Rainfall | CHIRPS v2.0 daily |
| Peat map | Peat-ML v0.9 (Zenodo, CC BY 4.0) |
| El Niño index | NOAA ONI |
| Boundaries | Natural Earth |

The data pipeline is reused from the team's earlier competition project. Raw and processed data are not stored in git.

## Timeline

| Week | Work | Status |
|---|---|---|
| 10 | Dataset, data characteristics, BOD, algorithms | Done |
| 11 to 12 | Code: preprocessing, mining, evaluation | Next |
| 13 | Report 1 | |
| 14 | Extra experiments (if requested) | |
| 15 | IMRAD article (Indonesian) | |
| 16 | Presentation | |
