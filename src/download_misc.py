"""Download ONI (ENSO index), Peat-ML global peatland extent (CC BY 4.0, Zenodo 7352284) and
Natural Earth 1:10m country and province boundaries (public domain)."""
import requests
from config import DATA

DATA.mkdir(exist_ok=True)
(DATA / "oni.txt").write_text(requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=60).text)
rec = requests.get("https://zenodo.org/api/records/7352284", timeout=60).json()
f = next(x for x in rec["files"] if x["key"].endswith(".nc"))
url = f["links"]["self"]
with requests.get(url, stream=True, timeout=600) as r, open(DATA / "peat_ml.nc", "wb") as out:
    for chunk in r.iter_content(1 << 20):
        out.write(chunk)
print("done", f["key"], f["size"])

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_{}.geojson"
for name, out in [("0_countries", "ne_countries.geojson"), ("1_states_provinces", "ne_provinces.geojson")]:
    (DATA / out).write_bytes(requests.get(NE.format(name), timeout=600).content)
    print("done", out)
