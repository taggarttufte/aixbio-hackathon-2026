"""Pull NWSS Influenza A wastewater data + HHS flu hospital admissions.

NWSS Flu A: data.cdc.gov dataset 'ymmh-divb', updated weekly Fridays.
HHS flu admissions: via Delphi covidcast, signal=confirmed_admissions_influenza_1d.
"""
import time, requests, pandas as pd
from pathlib import Path

OUT = Path("data/raw") ; OUT.mkdir(parents=True, exist_ok=True)

# ---- NWSS Flu A wastewater -------------------------------------------------
print("=== NWSS Flu A wastewater (ymmh-divb) ===")
url = "https://data.cdc.gov/api/views/ymmh-divb/rows.csv?accessType=DOWNLOAD"
t0 = time.time()
r = requests.get(url, timeout=120, stream=True)
out_path = OUT / "nwss_flu_a.csv"
with open(out_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=1 << 20):
        f.write(chunk)
print(f"  saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {time.time()-t0:.1f}s)")

# Quick peek
flu = pd.read_csv(out_path, nrows=5000)
print(f"  cols: {list(flu.columns)}")
print(f"  first rows:\n{flu.head(3)}\n")

# Reload full, parse dates if available
flu = pd.read_csv(out_path)
date_col = None
for c in ["date", "sample_date", "date_end", "Date"]:
    if c in flu.columns:
        date_col = c
        break
if date_col:
    flu[date_col] = pd.to_datetime(flu[date_col], errors="coerce")
    print(f"  total rows: {len(flu):,}  date range: "
          f"{flu[date_col].min()} -> {flu[date_col].max()}")
else:
    print(f"  no date column found; cols are {list(flu.columns)[:8]}")

# ---- HHS flu admissions via Delphi -----------------------------------------
print("\n=== HHS flu admissions (Delphi covidcast) ===")
URL = "https://api.delphi.cmu.edu/epidata/covidcast/"

def fetch(geo_type, geo_value):
    params = {
        "data_source": "hhs",
        "signal": "confirmed_admissions_influenza_1d",
        "geo_type": geo_type, "geo_value": geo_value,
        "time_type": "day", "time_values": "20210901-20250930",
    }
    r = requests.get(URL, params=params, timeout=60)
    js = r.json()
    if js.get("result") != 1:
        print(f"  FAIL ({geo_type}={geo_value}): {js.get('message')}")
        return pd.DataFrame()
    df = pd.DataFrame(js["epidata"])
    df["date"] = pd.to_datetime(df.time_value.astype(str), format="%Y%m%d")
    return df.sort_values("date")

frames = {}
for geo_type, geo_value, label in [("state", "nv", "NV"), ("nation", "us", "US")]:
    print(f"  HHS flu admits {label}...", end=" ")
    df = fetch(geo_type, geo_value)
    if not df.empty:
        s = df.set_index("date")["value"].rename(f"hhs_flu_{label}")
        frames[s.name] = s
        print(f"{len(s):,} rows  range={s.index.min().date()} -> {s.index.max().date()}")

if frames:
    flu_admits = pd.concat(frames.values(), axis=1).sort_index()
    flu_admits.to_csv(OUT / "hhs_flu_admissions.csv")
    print(f"\n  saved {OUT/'hhs_flu_admissions.csv'}  shape={flu_admits.shape}")
