"""Smoke test for preprocessing, mining and evaluation, without downloading anything.

It writes a small synthetic data/samples.parquet (same columns as build_dataset.py) into a temporary folder,
with one pattern planted on purpose: a cell that burned last week (hot_7d > 0) during a long dry spell
(dry_days >= 14) has a much higher chance of fire next week. Then it runs items.py, mine.py and
evaluate_rules.py on it and checks that:
- every item group is exclusive and the transactions keep all rows;
- Apriori, FP-Growth and ECLAT return identical itemsets (mine.py asserts this);
- the planted rule KERING_PANJANG & TERBAKAR_7HARI -> API_MINGGU_DEPAN is found with lift >= 1.5,
  and keeps lift >= 1.5 on the holdout years;
- all result files are written.
The real data/ and results/ folders are not touched.

Run from the project root:  .venv\\Scripts\\python tests\\smoke_test.py      (about 1 to 3 minutes)
"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="arm_smoke_"))
config.DATA, config.RESULTS = tmp / "data", tmp / "results"
config.DATA.mkdir(); config.RESULTS.mkdir()

import items, mine, evaluate_rules  # noqa: E402

for mod in (items, mine, evaluate_rules):
    mod.RESULTS = config.RESULTS
items.DATA = config.DATA


def synthetic_samples(seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2012-04-20", "2026-10-13", freq="7D")
    cells = [(j, i) for j in range(6) for i in range(6)]
    n = len(weeks) * len(cells)
    w = np.repeat(weeks, len(cells))
    j = np.tile([c[0] for c in cells], len(weeks)); i = np.tile([c[1] for c in cells], len(weeks))
    month = w.month.to_numpy()
    dry_season = np.isin(month, [6, 7, 8, 9, 10])
    dry = np.where(dry_season, rng.integers(0, 40, n), rng.integers(0, 12, n))
    hot7 = rng.poisson(np.where(dry >= 14, 1.5, 0.15))
    p = np.where((hot7 > 0) & (dry >= 14), 0.75, np.where(hot7 > 0, 0.35, 0.08))  # the planted pattern
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({
        "week": w, "j": j, "i": i, "island": np.where(j < 3, "Sumatra", "Kalimantan"), "province": [f"P{a}" for a in j],
        "peat": 0.1 + 0.12 * i, "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "oni": rng.normal(0, 0.8, n), "rain_7d": rng.gamma(1, 20, n), "rain_14d": 0.0, "rain_30d": rng.gamma(2, 70, n),
        "rain_60d": 0.0, "rain_90d": rng.gamma(4, 90, n), "dry_days": dry, "hot_7d": hot7, "hot_14d": hot7,
        "hot_30d": hot7 + rng.poisson(0.4, n), "hot_90d": 0, "hot_365d": 0, "hot_nbr_7d": rng.poisson(0.8, n),
        "hot_nbr_14d": 0, "hot_nbr_30d": rng.poisson(2, n), "days_since_fire": 0, "fire_day_share_3y": rng.random(n) * 0.1,
        "y": y, "hot_next_7d": y * 2, "quality": np.where(w > pd.Timestamp("2026-07-31"), "nrt", "final")})
    df["year"] = df.week.dt.year
    return df


def main():
    s = synthetic_samples()
    s.to_parquet(config.DATA / "samples.parquet")
    print(f"synthetic samples: {len(s):,} rows in {tmp}")

    items.main()
    tx = items.load_transactions()
    assert len(tx) == len(s), "transactions lost rows"
    assert set(items.ITEMS) <= set(tx.columns)

    sys.argv = ["mine.py", "--quick"]
    mine.main()  # asserts that the three algorithms agree
    evaluate_rules.main()

    rules = pd.read_csv(config.RESULTS / "rules_fire.csv")
    ev = pd.read_csv(config.RESULTS / "eval_rules_fire.csv")
    planted = {"KERING_PANJANG", "TERBAKAR_7HARI"}
    hit = ev[ev.antecedent.map(lambda a: planted <= set(a.split(" & ")))]
    assert len(hit), "planted rule not found"
    best = hit.sort_values("lift_2012-2019", ascending=False).iloc[0]
    assert best["lift_2012-2019"] >= 1.5 and best["lift_2020-2026"] >= 1.5, best
    assert (rules.lift >= mine.MIN_LIFT).all() and (rules.improvement >= mine.MIN_IMPROVEMENT).all()
    expected = ["item_support.csv", "data_overview.csv", "bin_edges.json", "benchmark.csv", "benchmark.png",
                "algorithm_check.csv", "rules_fire.csv", "rules_quiet.csv", "eval_rules_fire.csv", "eval_rules_quiet.csv",
                "rule_card_fire.csv", "card_vs_baselines.csv", "tree_rules_fire.csv", "summary.md",
                "lift_train_vs_holdout.png", "lift_per_year_card.png"]
    missing = [f for f in expected if not (config.RESULTS / f).exists()]
    assert not missing, missing
    print("\nSMOKE TEST PASSED")
    print(f"planted rule found: {best.antecedent}  lift train {best['lift_2012-2019']:.2f}, holdout {best['lift_2020-2026']:.2f}")
    print(f"results written to {config.RESULTS}")


if __name__ == "__main__":
    main()
