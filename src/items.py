"""Preprocessing: turn the weekly cell table (data/samples.parquet) into ARM transactions.

One transaction = one peat cell in one week. Every numeric feature is discretised into category items
(about 25 to 30 in total), so a transaction is the set of items that are true for that cell-week.
The consequent item API_MINGGU_DEPAN ("fire next week") is true when the label y is 1.

Rules for building items:
- Physical thresholds are used where one exists (ONI +-0.5 from NOAA, about 4 mm/day = 120 mm per 30 days
  from Field et al., 7 and 14 dry days, any hotspot at all).
- Where no physical threshold exists, the edge is a quantile computed on the mining years (2012 to 2019)
  only, so the holdout years 2020 to 2026 never influence the items.
- Items in one group are mutually exclusive bins (for example exactly one of the three rain-30 items is true).
- Seasonal history (PERNAH_TERBAKAR_BULAN_INI) only looks at earlier years, so it never uses the future.

Output: data/transactions.parquet (one boolean column per item, plus week, year, j, i, island, province, y)
        results/item_support.csv, results/bin_edges.json, results/data_overview.csv
Run:    python items.py
"""
import json
import numpy as np, pandas as pd
from config import DATA, RESULTS, MINE_YEARS

TARGET = "API_MINGGU_DEPAN"
META = ["week", "year", "j", "i", "island", "province", "y", "hot_next_7d", "quality"]

# Groups of items, used in the report and to check that the bins in a group are exclusive.
GROUPS = {
    "Rain": ["HUJAN30_SANGAT_RENDAH", "HUJAN30_RENDAH", "HUJAN30_NORMAL", "HUJAN7_SANGAT_RENDAH", "HUJAN90_RENDAH"],
    "Dry spell": ["KERING_7_13HARI", "KERING_PANJANG"],
    "Fire history (cell)": ["TERBAKAR_7HARI", "TERBAKAR_8_30HARI", "SENYAP_30HARI", "HOTSPOT_7HARI_TINGGI",
                            "RIWAYAT_API_TINGGI", "PERNAH_TERBAKAR_BULAN_INI"],
    "Fire nearby": ["TETANGGA_TERBAKAR_7HARI", "TETANGGA_TERBAKAR_30HARI"],
    "Peat": ["GAMBUT_TINGGI"],
    "Season": ["MUSIM_KEMARAU", "MUSIM_HUJAN", "MUSIM_PERALIHAN"],
    "El Nino (ONI)": ["EL_NINO", "LA_NINA", "ENSO_NETRAL"],
    "Place": ["PULAU_SUMATRA", "PULAU_KALIMANTAN"],
    "Consequent": [TARGET],
}
ITEMS = [it for g in GROUPS.values() for it in g]

# Plain Indonesian wording, so each rule can be printed as a sentence an officer can check every week.
ITEM_TEXT = {
    "HUJAN30_SANGAT_RENDAH": "hujan 30 hari terakhir < 50 mm",
    "HUJAN30_RENDAH": "hujan 30 hari terakhir 50-120 mm",
    "HUJAN30_NORMAL": "hujan 30 hari terakhir >= 120 mm",
    "HUJAN7_SANGAT_RENDAH": "hujan 7 hari terakhir < 5 mm",
    "HUJAN90_RENDAH": "hujan 90 hari terakhir termasuk 25% terendah",
    "KERING_7_13HARI": "7-13 hari berturut-turut tanpa hujan",
    "KERING_PANJANG": "14 hari atau lebih berturut-turut tanpa hujan",
    "TERBAKAR_7HARI": "ada hotspot di sel ini dalam 7 hari terakhir",
    "TERBAKAR_8_30HARI": "hotspot terakhir di sel ini 8-30 hari lalu",
    "SENYAP_30HARI": "tidak ada hotspot di sel ini selama 30 hari",
    "HOTSPOT_7HARI_TINGGI": "banyak hotspot di sel ini dalam 7 hari terakhir",
    "RIWAYAT_API_TINGGI": "sel termasuk 25% paling sering terbakar dalam 3 tahun",
    "PERNAH_TERBAKAR_BULAN_INI": "sel pernah terbakar pada bulan yang sama di tahun sebelumnya",
    "TETANGGA_TERBAKAR_7HARI": "sel tetangga ada hotspot dalam 7 hari terakhir",
    "TETANGGA_TERBAKAR_30HARI": "sel tetangga ada hotspot 8-30 hari lalu",
    "GAMBUT_TINGGI": "proporsi gambut sel tinggi",
    "MUSIM_KEMARAU": "bulan Juni-Oktober",
    "MUSIM_HUJAN": "bulan November-Maret",
    "MUSIM_PERALIHAN": "bulan April-Mei",
    "EL_NINO": "El Nino (ONI >= 0.5)",
    "LA_NINA": "La Nina (ONI <= -0.5)",
    "ENSO_NETRAL": "ENSO netral",
    "PULAU_SUMATRA": "di Sumatra",
    "PULAU_KALIMANTAN": "di Kalimantan",
    TARGET: "ada hotspot di sel ini minggu depan",
}


def seasonal_history(df):
    """True if the cell had a hotspot in the same calendar month in any EARLIER year (no look-ahead).

    A month counts as burned when any sampled week of that month had a hotspot in its last 7 days (hot_7d > 0)."""
    m = df.assign(month=df.week.dt.month, burned=(df.hot_7d > 0))
    per = m.groupby(["j", "i", "month", "year"]).burned.max().reset_index().sort_values("year")
    # cumulative "ever burned in this month" up to and including the year, then shift one year back
    per["ever"] = per.groupby(["j", "i", "month"]).burned.cummax()
    per["ever_before"] = per.groupby(["j", "i", "month"]).ever.shift(1, fill_value=False).astype(bool)
    out = m.merge(per[["j", "i", "month", "year", "ever_before"]], on=["j", "i", "month", "year"], how="left")
    return out.ever_before.fillna(False).to_numpy()


def bin_edges(df):
    """Edges that need data are learned on the mining years only."""
    tr = df[df.year.isin(MINE_YEARS)]
    cells = tr.drop_duplicates(["j", "i"])
    return {
        "rain30_very_low_mm": 50.0,           # under about 1.7 mm/day
        "rain30_low_mm": 120.0,               # under about 4 mm/day (Field et al.)
        "rain7_very_low_mm": 5.0,             # almost no rain in the last week
        "rain90_low_mm": float(tr.rain_90d.quantile(0.25)),        # driest quarter of 90-day totals
        "dry_days_week": 7, "dry_days_long": 14,
        "hot7_high": float(tr.loc[tr.hot_7d > 0, "hot_7d"].median()),  # busier half of burning cells
        "fire_share_high": float(tr.fire_day_share_3y.quantile(0.75)),  # most fire-prone quarter
        "peat_high": float(cells.peat.median()),                    # peatier half of the cells
        "oni_el_nino": 0.5, "oni_la_nina": -0.5,                    # NOAA definition
        "dry_season_months": [6, 7, 8, 9, 10], "wet_season_months": [11, 12, 1, 2, 3],
    }


def make_items(df, e):
    """Boolean item table for any rows that have the feature columns (also usable on new weeks)."""
    month = df.week.dt.month
    r30, hot7, nbr7 = df.rain_30d, df.hot_7d, df.hot_nbr_7d
    it = pd.DataFrame(index=df.index)
    it["HUJAN30_SANGAT_RENDAH"] = r30 < e["rain30_very_low_mm"]
    it["HUJAN30_RENDAH"] = (r30 >= e["rain30_very_low_mm"]) & (r30 < e["rain30_low_mm"])
    it["HUJAN30_NORMAL"] = r30 >= e["rain30_low_mm"]
    it["HUJAN7_SANGAT_RENDAH"] = df.rain_7d < e["rain7_very_low_mm"]
    it["HUJAN90_RENDAH"] = df.rain_90d <= e["rain90_low_mm"]
    it["KERING_7_13HARI"] = (df.dry_days >= e["dry_days_week"]) & (df.dry_days < e["dry_days_long"])
    it["KERING_PANJANG"] = df.dry_days >= e["dry_days_long"]
    it["TERBAKAR_7HARI"] = hot7 > 0
    it["TERBAKAR_8_30HARI"] = (hot7 == 0) & (df.hot_30d > 0)
    it["SENYAP_30HARI"] = df.hot_30d == 0
    it["HOTSPOT_7HARI_TINGGI"] = hot7 >= e["hot7_high"]
    it["RIWAYAT_API_TINGGI"] = df.fire_day_share_3y >= e["fire_share_high"]
    it["PERNAH_TERBAKAR_BULAN_INI"] = df["seasonal_hist"].astype(bool)
    it["TETANGGA_TERBAKAR_7HARI"] = nbr7 > 0
    it["TETANGGA_TERBAKAR_30HARI"] = (nbr7 == 0) & (df.hot_nbr_30d > 0)
    it["GAMBUT_TINGGI"] = df.peat >= e["peat_high"]
    it["MUSIM_KEMARAU"] = month.isin(e["dry_season_months"])
    it["MUSIM_HUJAN"] = month.isin(e["wet_season_months"])
    it["MUSIM_PERALIHAN"] = ~(it.MUSIM_KEMARAU | it.MUSIM_HUJAN)
    it["EL_NINO"] = df.oni >= e["oni_el_nino"]
    it["LA_NINA"] = df.oni <= e["oni_la_nina"]
    it["ENSO_NETRAL"] = ~(it.EL_NINO | it.LA_NINA)
    it["PULAU_SUMATRA"] = df.island == "Sumatra"
    it["PULAU_KALIMANTAN"] = df.island == "Kalimantan"
    if "y" in df:
        it[TARGET] = df.y == 1
    return it.astype(bool)


def load_transactions():
    """Transactions written by this script: item columns (bool) plus META columns."""
    return pd.read_parquet(DATA / "transactions.parquet")


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA / "samples.parquet")
    n0 = len(df)

    # --- cleaning checks (the heavy cleaning is in build_dataset.py: fires only, outages dropped) ---
    feat = ["rain_7d", "rain_30d", "rain_90d", "dry_days", "hot_7d", "hot_30d", "hot_nbr_7d", "hot_nbr_30d",
            "fire_day_share_3y", "peat", "oni"]
    missing = df[feat + ["y"]].isna().sum()
    print("missing values per column:\n", missing[missing > 0].to_string() if missing.any() else "  none")
    df = df.dropna(subset=feat + ["y"]).copy()
    dup = df.duplicated(["week", "j", "i"]).sum()
    df = df.drop_duplicates(["week", "j", "i"])
    print(f"rows {n0:,} -> {len(df):,} after dropping missing ({n0 - len(df) - dup}) and duplicates ({dup})")

    df["seasonal_hist"] = seasonal_history(df)
    e = bin_edges(df)
    it = make_items(df, e)

    # every row must have exactly one item from each exclusive group
    for g in (["HUJAN30_SANGAT_RENDAH", "HUJAN30_RENDAH", "HUJAN30_NORMAL"],
              ["MUSIM_KEMARAU", "MUSIM_HUJAN", "MUSIM_PERALIHAN"], ["EL_NINO", "LA_NINA", "ENSO_NETRAL"],
              ["TERBAKAR_7HARI", "TERBAKAR_8_30HARI", "SENYAP_30HARI"], ["PULAU_SUMATRA", "PULAU_KALIMANTAN"]):
        assert (it[g].sum(axis=1) == 1).all(), g

    tx = pd.concat([df[META].reset_index(drop=True), it[ITEMS].reset_index(drop=True)], axis=1)
    tx.to_parquet(DATA / "transactions.parquet")
    (RESULTS / "bin_edges.json").write_text(json.dumps(e, indent=2))

    # --- data characteristics for the report ---
    mine = tx.year.isin(MINE_YEARS)
    sup = pd.DataFrame({
        "group": [g for g, its in GROUPS.items() for _ in its],
        "item": ITEMS,
        "support_all": [tx[c].mean() for c in ITEMS],
        "support_2012_2019": [tx.loc[mine, c].mean() for c in ITEMS],
        "support_2020_2026": [tx.loc[~mine, c].mean() for c in ITEMS],
        "fire_rate_when_true": [tx.loc[tx[c], "y"].mean() if tx[c].any() else np.nan for c in ITEMS],
    })
    sup["lift_single_item"] = sup.fire_rate_when_true / tx.y.mean()
    sup.round(4).to_csv(RESULTS / "item_support.csv", index=False)

    overview = pd.DataFrame([
        ("transactions", len(tx)), ("cells", tx[["j", "i"]].drop_duplicates().shape[0]),
        ("provinces", tx.province.nunique()), ("weeks", tx.week.nunique()),
        ("first week", tx.week.min().date()), ("last week", tx.week.max().date()),
        ("items (incl. consequent)", len(ITEMS)), ("mean items per transaction", round(it[ITEMS].sum(1).mean(), 2)),
        ("fire next week, all", round(tx.y.mean(), 4)),
        ("fire next week, 2012-2019", round(tx.loc[mine, "y"].mean(), 4)),
        ("fire next week, 2020-2026", round(tx.loc[~mine, "y"].mean(), 4)),
        ("quiet 30 days share", round(tx.SENYAP_30HARI.mean(), 4)),
        ("fire next week | quiet 30 days", round(tx.loc[tx.SENYAP_30HARI, "y"].mean(), 4)),
        ("transactions 2012-2019", int(mine.sum())), ("transactions 2020-2026", int((~mine).sum())),
    ], columns=["measure", "value"])
    overview.to_csv(RESULTS / "data_overview.csv", index=False)
    print(overview.to_string(index=False))
    print("\nbin edges:", json.dumps({k: v for k, v in e.items() if not isinstance(v, list)}))
    print("\nitem support (2012-2019) and single-item lift:")
    print(sup[["item", "support_2012_2019", "fire_rate_when_true", "lift_single_item"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
