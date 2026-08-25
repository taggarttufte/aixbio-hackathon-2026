"""Pull COVID-specific ground-truth signals via CMU Delphi covidcast API.

Two signals at state level (NV) and one national:
  - HHS hospital admissions (covid-confirmed): daily, 2020-present
  - JHU CSSE confirmed incidence (cases): daily, ends March 2023
  - JHU CSSE deaths: daily, ends March 2023

The HHS series is the most reliable COVID-specific ground truth that extends
into 2024. JHU cases/deaths cover the early period as a complementary anchor.
"""
import requests, pandas as pd
from pathlib import Path

OUT = Path("data/raw") ; OUT.mkdir(parents=True, exist_ok=True)
URL = "https://api.delphi.cmu.edu/epidata/covidcast/"

# YYYYMMDD time strings via Delphi's range syntax
TIME_RANGE = "20210901-20250930"

def fetch(source, signal, geo_type, geo_value, time_range=TIME_RANGE):
    params = {
        "data_source": source, "signal": signal,
        "geo_type": geo_type, "geo_value": geo_value,
        "time_type": "day", "time_values": time_range,
    }
    r = requests.get(URL, params=params, timeout=60)
    js = r.json()
    if js.get("result") != 1:
        print(f"  FAIL {source}:{signal} ({geo_type}={geo_value}): {js.get('message')}")
        return pd.DataFrame()
    df = pd.DataFrame(js["epidata"])
    df["date"] = pd.to_datetime(df["time_value"].astype(str), format="%Y%m%d")
    return df.sort_values("date")

frames = {}

# ---- HHS hospital admissions (Nevada + national) ----
for geo_type, geo_value, label in [("state", "nv", "NV"), ("nation", "us", "US")]:
    print(f"  HHS admissions {label}...", end=" ")
    df = fetch("hhs", "confirmed_admissions_covid_1d", geo_type, geo_value)
    if not df.empty:
        s = df.set_index("date")["value"].rename(f"hhs_admits_{label}")
        frames[s.name] = s
        print(f"{len(s):,} rows  range={s.index.min().date()} -> {s.index.max().date()}")

# ---- JHU CSSE cases (NV + US, ends Mar 2023) ----
for geo_type, geo_value, label in [("state", "nv", "NV"), ("nation", "us", "US")]:
    print(f"  JHU cases {label}...", end=" ")
    df = fetch("jhu-csse", "confirmed_incidence_num", geo_type, geo_value)
    if not df.empty:
        s = df.set_index("date")["value"].rename(f"jhu_cases_{label}")
        frames[s.name] = s
        print(f"{len(s):,} rows  range={s.index.min().date()} -> {s.index.max().date()}")

# ---- JHU deaths ----
for geo_type, geo_value, label in [("state", "nv", "NV"), ("nation", "us", "US")]:
    print(f"  JHU deaths {label}...", end=" ")
    df = fetch("jhu-csse", "deaths_incidence_num", geo_type, geo_value)
    if not df.empty:
        s = df.set_index("date")["value"].rename(f"jhu_deaths_{label}")
        frames[s.name] = s
        print(f"{len(s):,} rows  range={s.index.min().date()} -> {s.index.max().date()}")

if not frames:
    raise SystemExit("nothing fetched")

out = pd.concat(frames.values(), axis=1).sort_index()
out.to_csv(OUT / "covid_ground_truth.csv")
print(f"\nsaved {OUT / 'covid_ground_truth.csv'}  shape={out.shape}")
print(f"columns: {list(out.columns)}")
print("\ntail of HHS admissions:")
print(out[[c for c in out.columns if c.startswith('hhs_')]].dropna().tail(5).to_string())
