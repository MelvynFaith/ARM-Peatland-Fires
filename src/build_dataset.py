"""Grid hotspots, rainfall, peat and ENSO into weekly cell samples.

Sample = (cell, week start t). Features use data up to and including day t.
Label = at least one VIIRS vegetation-fire hotspot (type 0, confidence not low) in the cell during days t+1..t+7.
Gas flares, industrial sites, offshore platforms and volcanoes (FIRMS type 1 to 3) are not fires and are dropped.
Days with no VIIRS detection at all in the study box are satellite outages, not fire-free days: weeks whose
label window or last 14 days touch one (or more than 2 of the last 30 days) are dropped. Only Indonesian cells in Sumatra and Kalimantan are kept.

After SP_END the hotspots are near-real-time (NRT) data from the same satellite. NRT has no `type` column, so
detections within about 1 km of a spot the SP archive ever flagged as a non-vegetation source are dropped.
After the last final CHIRPS day the rainfall is preliminary CHIRPS; days not yet published at the very end are
filled with each cell's mean of the 7 days before. Samples carry `quality`: "final" when features and label
come from archived data only, "nrt" otherwise. The rows for the latest day (the live forecast, no label yet)
are written to data/forecast_features.parquet, with the rows for a week earlier so the bulletin can show what changed.
"""
import numpy as np, pandas as pd, xarray as xr
from config import BBOX, CELL, DATA, HORIZON_DAYS, PEAT_MIN_FRAC, START, END, SP_END
from regions import cell_regions

LON0, LAT0 = BBOX[0], BBOX[1]
NX = int(round((BBOX[2] - BBOX[0]) / CELL))
NY = int(round((BBOX[3] - BBOX[1]) / CELL))


def _pixel_key(lat, lon):
    """Integer key of the 0.01 degree pixel (about 1 km) a detection falls in."""
    return np.round(np.asarray(lat) / 0.01).astype(np.int64) * 100000 + np.round(np.asarray(lon) / 0.01).astype(np.int64)


def vegetation_fires(df):
    """Keep vegetation-fire detections with confidence nominal or high.
    SP rows: FIRMS type 0. NRT rows (no type): not within one pixel of a spot SP ever flagged as type 1 to 3."""
    if "source" not in df:
        df = df.assign(source="SP")
    typ = pd.to_numeric(df["type"], errors="coerce")
    st = df[(df.source == "SP") & typ.isin([1, 2, 3])]
    k = _pixel_key(st.latitude, st.longitude)
    static = np.unique(np.concatenate([k + da * 100000 + db for da in (-1, 0, 1) for db in (-1, 0, 1)]))
    conf = df.confidence.astype(str).str.lower() != "l"
    sp = (df.source == "SP") & typ.eq(0)
    nrt = (df.source == "NRT") & ~np.isin(_pixel_key(df.latitude, df.longitude), static)
    return df[conf & (sp | nrt)]


def hotspot_cube(days):
    """Daily vegetation-fire hotspot counts per cell, and a mask of outage days (no detection of any kind)."""
    df = pd.read_parquet(DATA / "firms_viirs.parquet")
    df = df[(df.longitude >= BBOX[0]) & (df.longitude < BBOX[2]) & (df.latitude >= BBOX[1]) & (df.latitude < BBOX[3])]
    t_all = (pd.to_datetime(df.acq_date) - days[0]).dt.days.to_numpy()
    seen = np.zeros(len(days), bool)
    seen[t_all[(t_all >= 0) & (t_all < len(days))]] = True
    df = vegetation_fires(df)
    t = (pd.to_datetime(df.acq_date) - days[0]).dt.days.to_numpy()
    j = ((df.latitude - LAT0) // CELL).astype(int).to_numpy()
    i = ((df.longitude - LON0) // CELL).astype(int).to_numpy()
    ok = (t >= 0) & (t < len(days))
    cube = np.zeros((len(days), NY, NX), np.float32)
    np.add.at(cube, (t[ok], j[ok], i[ok]), 1)
    return cube, ~seen


def rain_cube(days):
    """Daily rain per cell (NaN over ocean): final CHIRPS, then preliminary CHIRPS, then (only for the last
    few days not yet published) each cell's mean of the 7 days before."""
    p = xr.open_dataset(DATA / "chirps_025.nc").precip
    pre = DATA / "chirps_prelim_025.nc"
    if pre.exists():
        q = xr.open_dataset(pre).precip
        p = xr.concat([p, q.sel(time=q.time > p.time[-1])], "time")
    a = p.reindex(time=days).to_numpy().astype(np.float32)
    have = np.where(~np.all(np.isnan(a), axis=(1, 2)))[0]
    if have[-1] < len(days) - 1:
        a[have[-1] + 1:] = np.nanmean(a[have[-1] - 6:have[-1] + 1], axis=0)
    return a


def peat_grid():
    """Mean Peat-ML probability per study cell. Binned by cell index so the grid matches rainfall and hotspots."""
    p = xr.open_dataset(DATA / "peat_ml.nc").PEATLAND_P
    p = p.sel(lat=slice(BBOX[1], BBOX[3]), lon=slice(BBOX[0], BBOX[2]))
    v = p.to_numpy()
    if np.nanmax(v) > 1.5:
        v = v / 100.0
    j = np.floor((p.lat.to_numpy() - LAT0) / CELL).astype(int)
    i = np.floor((p.lon.to_numpy() - LON0) / CELL).astype(int)
    ok_j, ok_i = (j >= 0) & (j < NY), (i >= 0) & (i < NX)
    J, I = np.meshgrid(j[ok_j], i[ok_i], indexing="ij")
    vv = v[np.ix_(ok_j, ok_i)]
    good = ~np.isnan(vv)
    tot = np.zeros((NY, NX)); cnt = np.zeros((NY, NX))
    np.add.at(tot, (J[good], I[good]), vv[good])
    np.add.at(cnt, (J[good], I[good]), 1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def roll_sum(x, w):
    c = np.cumsum(np.nan_to_num(x), axis=0, dtype=np.float64)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out.astype(np.float32)


def dry_run(rain, thr=1.0):
    dry = (np.nan_to_num(rain, nan=0.0) < thr)
    run = np.zeros_like(rain)
    for t in range(len(rain)):
        run[t] = (run[t - 1] + 1 if t else 1) * dry[t]
    return np.minimum(run, 120)


def days_since(x, cap=1000):
    """Days since the last day with at least one hotspot (capped)."""
    out = np.zeros(x.shape, np.float32)
    last = np.full(x.shape[1:], -cap, dtype=np.int64)
    for t in range(len(x)):
        last = np.where(x[t] > 0, t, last)
        out[t] = np.minimum(t - last, cap)
    return out


def neighbour_sum(x):
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    return sum(pad[:, a:a + NY, b:b + NX] for a in range(3) for b in range(3)) - x


def oni_by_month():
    rows = [l.split() for l in (DATA / "oni.txt").read_text().splitlines()[1:]]
    seas = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    d = {}
    for s, y, _, a in rows:
        m = seas.index(s) + 1  # month at the centre of the season
        d[(int(y), m)] = float(a)
    return d


def main():
    days = pd.date_range(START, END)
    hot, outage = hotspot_cube(days)
    rain = rain_cube(days)
    peat = peat_grid()
    oni = oni_by_month()
    reg = cell_regions().set_index(["j", "i"])
    island = reg.island.unstack().to_numpy()
    province = reg.province.unstack().to_numpy()
    print("outage days:", [str(d.date()) for d in days[outage]])

    feats = {f"rain_{w}d": roll_sum(rain, w) for w in (7, 14, 30, 60, 90)}
    feats["dry_days"] = dry_run(rain)
    for w in (7, 14, 30):
        feats[f"hot_{w}d"] = roll_sum(hot, w)
    feats["hot_nbr_14d"] = neighbour_sum(feats["hot_14d"])
    # longer fire memory: helps rank quiet cells for new ignitions
    for w in (90, 365):
        feats[f"hot_{w}d"] = roll_sum(hot, w)
    feats["hot_nbr_7d"] = neighbour_sum(feats["hot_7d"])
    feats["hot_nbr_30d"] = neighbour_sum(feats["hot_30d"])
    feats["days_since_fire"] = days_since(hot)
    fire_day = (hot > 0).astype(np.float32)
    feats["fire_day_share_3y"] = roll_sum(fire_day, 1095) / np.minimum(np.arange(len(days)) + 1, 1095)[:, None, None]
    fut = roll_sum(hot, HORIZON_DAYS)  # fut[t] = hotspots in t-6..t

    valid = ~np.isnan(rain[0]) & (peat >= PEAT_MIN_FRAC) & np.isin(island, ["Sumatra", "Kalimantan"])
    print("cells kept:", int(valid.sum()), "| peat cells dropped outside Sumatra/Kalimantan Indonesia:",
          int((~np.isnan(rain[0]) & (peat >= PEAT_MIN_FRAC) & ~valid).sum()))
    jj, ii = np.where(valid)
    ts = np.arange(90, len(days) - HORIZON_DAYS, 7)  # weekly, need 90 days of history
    out_cum = np.concatenate([[0], np.cumsum(outage)])
    # drop a week if an outage hits its label window or last 14 days, or more than 2 of its last 30 days
    clean = np.array([(out_cum[t + HORIZON_DAYS + 1] - out_cum[t - 13] == 0) and (out_cum[t + 1] - out_cum[t - 29] <= 2)
                      for t in ts])
    print("weeks dropped for outages:", [str(days[t].date()) for t in ts[~clean]])
    ts = ts[clean]
    sp_end = (pd.Timestamp(SP_END) - days[0]).days

    def rows_for(t, labelled=True):
        d = days[t]
        lag = d - pd.DateOffset(months=2)  # ONI season centred 2 months earlier, so no future months leak in
        o = oni.get((lag.year, lag.month), np.nan)
        r = {"week": np.full(len(jj), d), "j": jj, "i": ii, "island": island[jj, ii], "province": province[jj, ii],
             "peat": peat[jj, ii],
             "month_sin": np.full(len(jj), np.sin(2 * np.pi * d.month / 12)),
             "month_cos": np.full(len(jj), np.cos(2 * np.pi * d.month / 12)),
             "oni": np.full(len(jj), o),
             "y": (fut[t + HORIZON_DAYS][jj, ii] > 0).astype(int) if labelled else np.full(len(jj), np.nan),
             # count, used only to measure how much fire activity alerts cover
             "hot_next_7d": fut[t + HORIZON_DAYS][jj, ii] if labelled else np.full(len(jj), np.nan),
             "quality": "final" if labelled and t + HORIZON_DAYS <= sp_end else ("nrt" if labelled else "forecast")}
        for k, v in feats.items():
            r[k] = v[t][jj, ii]
        return pd.DataFrame(r)

    df = pd.concat([rows_for(t) for t in ts], ignore_index=True)
    df["year"] = df.week.dt.year
    df.to_parquet(DATA / "samples.parquet")
    tf = len(days) - 1
    if out_cum[tf + 1] - out_cum[tf - 13] > 0:
        print("warning: a satellite outage falls in the last 14 days; the live forecast is less reliable")
    fc = pd.concat([rows_for(t, labelled=False) for t in (tf - HORIZON_DAYS, tf)], ignore_index=True)  # latest day and a week earlier
    fc["year"] = fc.week.dt.year
    fc.to_parquet(DATA / "forecast_features.parquet")
    print("forecast rows for", days[tf].date(), "and", days[tf - HORIZON_DAYS].date(), "->", len(fc), "rows")
    print("quality:", df.quality.value_counts().to_dict())
    print(df.shape, "positive rate", df.y.mean().round(4))
    print(df.groupby("year").y.mean().round(3).to_string())


if __name__ == "__main__":
    main()
