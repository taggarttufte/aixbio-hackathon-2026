"""Pull Google Trends for COVID symptom queries, Nevada scope, 2021-09 to 2025-09.

pytrends quirks:
- 5 keywords max per request.
- timeframe strings: "YYYY-MM-DD YYYY-MM-DD" for custom range.
- geo="US-NV" for Nevada; "US" for national (fallback).
- Rate limits hit hard — pause between calls.
- Returns weekly data for >9-month windows (daily unavailable at this span).
"""
import time
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

# Group symptoms conservatively. "covid" itself is dominated by media cycles;
# specific symptoms are closer to actual illness behavior.
GROUPS = [
    ["covid symptoms", "loss of taste", "loss of smell", "covid test", "sore throat"],
    ["fever", "cough", "congestion", "runny nose", "body aches"],
]
TIMEFRAME = "2021-09-01 2025-09-15"

frames = []
for geo in ["US-NV", "US"]:
    for i, grp in enumerate(GROUPS):
        print(f"[{geo}] group {i+1}: {grp}")
        try:
            pytrends.build_payload(grp, timeframe=TIMEFRAME, geo=geo)
            df = pytrends.interest_over_time()
            if df.empty:
                print(f"  empty response")
                continue
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df.columns = [f"{geo}::{c}" for c in df.columns]
            frames.append(df)
            print(f"  {len(df)} rows, cols={list(df.columns)}")
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(8)  # back off to avoid blocks

if not frames:
    raise SystemExit("pytrends returned no data — bailing, fall back to NWSS-only")

out = pd.concat(frames, axis=1)
out.index.name = "date"
out.to_csv(OUT / "trends_nv_us.csv")
print(f"\nsaved {OUT / 'trends_nv_us.csv'}  shape={out.shape}")
print(out.head(3))
print(out.tail(3))
