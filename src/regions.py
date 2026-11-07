"""Assign every 0.25 degree cell to a country and province (Natural Earth 1:10m, public domain).

A cell takes the province that holds most of its land, sampled on a 5 x 5 grid of points
(0.05 degree, the CHIRPS resolution). Cells with no sampled land take the nearest province.
"""
import json
import numpy as np, pandas as pd
from shapely import points
from shapely.geometry import shape
from shapely.strtree import STRtree
from config import BBOX, CELL, DATA

NX = int(round((BBOX[2] - BBOX[0]) / CELL))
NY = int(round((BBOX[3] - BBOX[1]) / CELL))
SUMATRA = {"Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau", "Jambi", "Bengkulu",
           "Sumatera Selatan", "Bangka-Belitung", "Lampung"}


def provinces():
    gj = json.loads((DATA / "ne_provinces.geojson").read_text(encoding="utf-8"))
    keep = [f for f in gj["features"] if f["properties"]["admin"] in ("Indonesia", "Malaysia", "Brunei", "Singapore")]
    return [shape(f["geometry"]) for f in keep], [(f["properties"]["admin"], f["properties"]["name"]) for f in keep]


def cell_regions():
    """DataFrame with one row per cell: j, i, lon, lat, country, province, island."""
    path = DATA / "cells.parquet"
    if path.exists():
        return pd.read_parquet(path)
    geoms, names = provinces()
    tree = STRtree(geoms)
    off = (np.arange(5) + 0.5) * CELL / 5
    rows = []
    for j in range(NY):
        for i in range(NX):
            lon0, lat0 = BBOX[0] + CELL * i, BBOX[1] + CELL * j
            xx, yy = np.meshgrid(lon0 + off, lat0 + off)
            pts = points(xx.ravel(), yy.ravel())
            _, hit = tree.query(pts, predicate="within")
            if len(hit):
                k = np.bincount(hit, minlength=len(geoms)).argmax()
            else:
                k = tree.nearest(points(lon0 + CELL / 2, lat0 + CELL / 2))
            country, prov = names[k]
            island = ("Sumatra" if prov in SUMATRA else "Kalimantan" if prov.startswith("Kalimantan") else "Other") \
                if country == "Indonesia" else "Outside Indonesia"
            rows.append((j, i, lon0 + CELL / 2, lat0 + CELL / 2, country, prov, island, len(hit) / 25))
    df = pd.DataFrame(rows, columns=["j", "i", "lon", "lat", "country", "province", "island", "land_frac"])
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    c = cell_regions()
    print(c[c.land_frac > 0].groupby(["country", "island"]).size())
