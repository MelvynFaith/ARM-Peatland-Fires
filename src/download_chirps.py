"""Subset CHIRPS v2 daily 0.05 deg rainfall to the study area and aggregate to 0.25 deg cells.

Final CHIRPS (netCDF per year) up to the last day published; each year is cached in data/chirps/ and
retried on network errors, so reruns resume. Days after the last final day come from preliminary
CHIRPS (daily GeoTIFF, published a few days after the fact) and are written separately to
data/chirps_prelim_025.nc, so it is always clear which rainfall is final.
"""
import gzip, io, re, time
import numpy as np, pandas as pd, requests, xarray as xr
import fsspec
from PIL import Image
from config import BBOX, CELL, DATA, END

URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.{y}.days_p05.nc"
PRELIM = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/{y}/"
K = int(round(CELL / 0.05))


def one_year(y, tries=6):
    for i in range(tries):
        try:
            fs = fsspec.filesystem("http", client_kwargs={"trust_env": True}, block_size=8 * 1024 * 1024)
            with fs.open(URL.format(y=y)) as f:
                ds = xr.open_dataset(f, engine="h5netcdf")
                p = ds.precip.sel(latitude=slice(BBOX[1], BBOX[3]), longitude=slice(BBOX[0], BBOX[2])).load()
            p = p.where(p >= 0)
            return p.coarsen(latitude=K, longitude=K, boundary="trim").mean()
        except Exception as e:
            print(f"{y} try {i + 1} failed: {type(e).__name__}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"CHIRPS {y} failed")


def prelim_day(day, lat, lon):
    """One preliminary daily GeoTIFF (global 0.05 deg, north-up from lat 50, lon -180) on the 0.25 deg grid."""
    url = PRELIM.format(y=day.year) + f"chirps-v2.0.{day:%Y.%m.%d}.tif.gz"
    r = requests.get(url, timeout=300); r.raise_for_status()
    a = np.array(Image.open(io.BytesIO(gzip.decompress(r.content))), dtype=np.float32)
    rows = slice(int(round((50 - BBOX[3]) / 0.05)), int(round((50 - BBOX[1]) / 0.05)))
    cols = slice(int(round((BBOX[0] + 180) / 0.05)), int(round((BBOX[2] + 180) / 0.05)))
    a = a[rows, cols][::-1]                      # south to north, like the netCDF files
    a[a < 0] = np.nan
    a = np.nanmean(a.reshape(len(lat), K, len(lon), K), axis=(1, 3))
    return xr.DataArray(a[None], coords={"time": [day], "latitude": lat, "longitude": lon},
                        dims=("time", "latitude", "longitude"))


def main():
    cache = DATA / "chirps"
    cache.mkdir(parents=True, exist_ok=True)
    old = DATA / "chirps_025.nc"
    if old.exists() and not any(cache.glob("*.nc")):  # seed the per-year cache from an earlier combined file
        p = xr.open_dataset(old).precip.load()
        for y, g in p.groupby("time.year"):
            if len(g.time) >= 365:
                g.to_dataset(name="precip").to_netcdf(cache / f"{y}.nc")
    end = pd.Timestamp(END)
    out = []
    for y in range(2011, end.year + 1):
        full = cache / f"{y}.nc"
        if full.exists():
            p = xr.open_dataset(full).precip
        else:  # the current year is published in parts: re-read until it is complete
            p = one_year(y)
            if len(p.time) >= 365:
                p.to_dataset(name="precip").to_netcdf(full)
        print(y, "final to", str(p.time.values[-1])[:10], flush=True)
        out.append(p)
    final = xr.concat(out, "time")
    final.to_dataset(name="precip").to_netcdf(DATA / "chirps_025.nc")
    last = pd.Timestamp(final.time.values[-1])

    have = set(re.findall(r"chirps-v2\.0\.(\d{4}\.\d{2}\.\d{2})\.tif\.gz", requests.get(PRELIM.format(y=end.year), timeout=60).text))
    days = [d for d in pd.date_range(last + pd.Timedelta(days=1), end) if f"{d:%Y.%m.%d}" in have]
    pre = [prelim_day(d, final.latitude.values, final.longitude.values) for d in days]
    if pre:
        xr.concat(pre, "time").to_dataset(name="precip").to_netcdf(DATA / "chirps_prelim_025.nc")
        print("prelim", str(days[0].date()), "to", str(days[-1].date()))


if __name__ == "__main__":
    main()
