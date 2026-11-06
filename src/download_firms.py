"""Download VIIRS 375 m hotspots (FIRMS area API) in 5-day windows. Needs env FIRMS_MAP_KEY.

Archived standard-processing data (SP) up to SP_END, then near-real-time data (NRT) from the same
satellite (S-NPP) up to END. NRT rows have no `type` column; build_dataset.py removes gas flares and
other static heat sources from them using the locations flagged in the SP archive."""
import io, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
from config import BBOX, DATA, START, END, SP_END

SOURCES = {"SP": "VIIRS_SNPP_SP", "NRT": "VIIRS_SNPP_NRT"}
WINDOW = 5  # SP source allows 1..5 days per request
URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}/{date}"


STATUS = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}"
LIMIT_MSGS = ("Invalid MAP_KEY", "Exceeding", "Exceeded")  # FIRMS answers these while the 10-min quota is spent


def wait_for_quota(key, free_below=3000):
    """Block until the key's rolling 10-minute transaction count is low enough to continue."""
    while True:
        try:
            used = requests.get(STATUS.format(key=key), timeout=60).json()["current_transactions"]
        except (requests.RequestException, ValueError, KeyError):
            used = free_below
        if used < free_below:
            return
        print(f"quota used {used}, waiting", file=sys.stderr)
        time.sleep(60)


def fetch(date, key, src="SP", days=WINDOW, tries=8):
    folder = DATA / "firms_cache" if src == "SP" else DATA / "firms_cache" / src.lower()
    cache = folder / f"{date}_{days}d.csv" if days != WINDOW else folder / f"{date}.csv"
    if cache.exists():
        return pd.read_csv(cache) if cache.stat().st_size else pd.DataFrame()
    url = URL.format(key=key, src=SOURCES[src], bbox=",".join(map(str, BBOX)), date=date, days=days)
    for i in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and not r.text.startswith(("Invalid", "Exceed")):
                cache.write_text(r.text.strip() and r.text)
                return pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()
            print(f"{date}: {r.status_code} {r.text[:80]!r}", file=sys.stderr)
            if any(m in r.text for m in LIMIT_MSGS):
                wait_for_quota(key)
                continue
        except requests.RequestException as e:
            print(f"{date}: {e}", file=sys.stderr)
        time.sleep(2 ** i)
    raise RuntimeError(f"failed {date}")


def main():
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        sys.exit("Set FIRMS_MAP_KEY")
    (DATA / "firms_cache" / "nrt").mkdir(parents=True, exist_ok=True)
    jobs = []
    for src, a, b in [("SP", START, SP_END), ("NRT", pd.Timestamp(SP_END) + pd.Timedelta(days=1), END)]:
        for d in pd.date_range(a, b, freq=f"{WINDOW}D"):
            days = min(WINDOW, (pd.Timestamp(b) - d).days + 1)  # last window must not run past b
            jobs.append((src, d.strftime("%Y-%m-%d"), days))
    with ThreadPoolExecutor(2) as ex:
        parts = list(ex.map(lambda j: fetch(j[1], key, j[0], j[2]).assign(source=j[0]), jobs))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if "type" not in df:
        df["type"] = pd.NA
    df = df[["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "source"]]  # type 0 = vegetation fire
    df = df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
    df = df[(df.source == "SP") | (pd.to_datetime(df.acq_date) > pd.Timestamp(SP_END))]  # no overlap between sources
    df.to_parquet(DATA / "firms_viirs.parquet")
    print(len(df), "hotspots", df.acq_date.min(), df.acq_date.max())


if __name__ == "__main__":
    main()
