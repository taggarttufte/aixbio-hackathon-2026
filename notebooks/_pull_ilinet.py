"""Pull CDC ILINet weekly outpatient ILI surveillance via Delphi epidata API.

CMU Delphi epidata — free, no auth, JSON response. Wraps CDC FluView.
  https://api.delphi.cmu.edu/epidata/fluview/

Region codes:
  - 'nat'  = national
  - 'hhs9' = HHS region 9 (AZ, CA, HI, NV)  ← matches our NWSS Reno site
  - other: hhs1..10, cen1..9, state codes via different endpoint
"""
import requests, pandas as pd
from pathlib import Path

OUT = Path("data/raw") ; OUT.mkdir(parents=True, exist_ok=True)
URL = "https://api.delphi.cmu.edu/epidata/fluview/"

# Epiweek format: YYYYWW. Our window: ~MMWR week 2021-36 to 2025-38.
EPIWEEKS = "202136-202538"
REGIONS = ["nat", "hhs9"]

frames = []
for region in REGIONS:
    print(f"  region={region}...", end=" ")
    r = requests.get(URL, params={"regions": region, "epiweeks": EPIWEEKS}, timeout=30)
    js = r.json()
    if js.get("result") != 1:
        print(f"FAIL: {js.get('message')}")
        continue
    df = pd.DataFrame(js["epidata"])
    print(f"{len(df):,} rows")
    df["region_code"] = region
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
# Convert epiweek (YYYYWW) to a proper date — start of MMWR week (Sunday)
out["epiweek"] = out["epiweek"].astype(str)
out["year"] = out["epiweek"].str[:4].astype(int)
out["week"] = out["epiweek"].str[4:].astype(int)
# MMWR week start ≈ Sunday of that ISO-ish week. Use pandas:
out["week_start"] = pd.to_datetime(
    out.year.astype(str) + "-W" + out.week.astype(str).str.zfill(2) + "-0",
    format="%G-W%V-%w",
)
out = out.sort_values(["region_code", "week_start"])
out.to_csv(OUT / "ilinet.csv", index=False)
print(f"\nsaved {OUT/'ilinet.csv'}  shape={out.shape}")
print(f"date range: {out.week_start.min().date()} -> {out.week_start.max().date()}")
print("\ncolumns:", list(out.columns))
print("\nlatest 5 rows for hhs9:")
hhs9 = out[out.region_code == "hhs9"].tail(5)[["week_start", "wili", "ili", "num_ili", "num_patients"]]
print(hhs9.to_string(index=False))
