# Status

Last updated: 7 November 2026.

## Course requirements

Data Mining course project, topic association rule mining (ARM). Weekly plan from the lecturer:

| Week | Requirement | Status |
|---|---|---|
| 10 | Find a dataset. Show the data, its characteristics, the BOD (Business Objective Determination) and the algorithms (more than one suggested) | Done, see [RESEARCH.md](RESEARCH.md) and [DATA.md](DATA.md) |
| 11 to 12 | Code: preprocessing, mining, evaluation | Next |
| 13 | Report 1 | |
| 14 | Extra experiments, if requested | |
| 15 | Article in Indonesian, IMRAD structure | |
| 16 | Presentation | |

## Decisions made

- **Data.** Reuse the SIGAP Gambut weekly peat-cell dataset. It is built from public satellite and climate data, not from Kaggle.
- **Direction.** Two rule sets with a 7-day lead time:
  - ignition rules on quiet cells (no hotspot in 30 days);
  - extinction rules on burning cells (hotspot in the last 7 days).

  The reasons and the novelty are in RESEARCH.md, sections 2 and 3.
- **Algorithms.** Apriori and FP-Growth, plus ECLAT.
- **Splits.** Mine on 2012 to 2019, tune on 2020 to 2022, test once on 2023 to 2026 (`config.py`).
- **Output.** A rule card of about 10 ignition and 10 extinction rules, each tested against a baseline, then applied to the latest week as a demo.

## Done

- Research, novelty and BOD: `docs/RESEARCH.md`.
- Data pipeline copied into `src/`, documented in `docs/DATA.md`. It has not been run in this repo yet.

## Next steps (week 11)

1. Build the data. Set `FIRMS_MAP_KEY`, then run the pipeline (README). Check the row count and positive rate against DATA.md.
2. Item building, `src/items.py` (planned):
   - discretise features into items, with bin edges from 2012 to 2019 only;
   - add seasonal history per cell and month;
   - make the quiet and burning subsets.
3. Mining, `src/mine.py` (planned): Apriori, FP-Growth and ECLAT, keeping only rules whose consequent is `API_BARU` or `PADAM`. Record run time and memory at minimum support 5%, 2%, 1% and 0.5%. Add `mlxtend` to `requirements.txt`.
4. Evaluation, `src/evaluate_rules.py` (planned):
   - per-rule lift relative to the subset, the improvement over sub-rules, lift per year, and block-bootstrap intervals;
   - the rule card against random and seasonal-history baselines;
   - the El Niño versus normal contrast.

## Open questions

- Do the success thresholds in RESEARCH.md section 4 (holdout lift at least 1.5, support at least 0.5%) match the lecturer's expectations?
- Does the course allow reusing the competition project's data pipeline, and how should AI assistance be disclosed? The report should state both.
- Who are the team members for this course project?

## Week 11 progress (7 November 2026)

Full write-up: [REPORT.md](REPORT.md).

- **Code added:** `src/items.py` (preprocessing into 25 items), `src/mine.py` (Apriori, FP-Growth, ECLAT, class rules, benchmark), `src/evaluate_rules.py` (holdout and per-year lift, block bootstrap, El Niño contrast, rule card versus baselines, decision tree), `tests/smoke_test.py`.
- **BOD followed:** the week 11 BOD from the team: rules ending in "fire next week" on all cells, plus quiet cells for new ignitions (question 4). Extinction rules from RESEARCH.md are left for week 14.
- **Splits used:** mine on 2012 to 2019, holdout 2020 to 2026, also reported as 2020-2022 and 2023-2026 (`evaluate_rules.py`).
- **Data:** FIRMS, Peat-ML, ONI and boundaries downloaded. CHIRPS is blocked by a server outage on `data.chc.ucsb.edu`. Pipeline fixes are listed in DATA.md, "Week 11 changes to the pipeline".
- **Verified:** smoke test passes, the three algorithms give identical itemsets, Apriori runs with `low_memory=True` (the default needed too much RAM).
- **Next:** run the pipeline when CHIRPS is back and fill REPORT.md sections 7.2 to 7.6 from `results/summary.md`.
