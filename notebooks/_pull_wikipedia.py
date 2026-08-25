"""Pull Wikipedia daily pageviews for respiratory-illness articles.

Wikimedia REST API. No auth. ~100 req/sec limit (we're well under).
Schema:
  GET /metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}
  -> { "items": [{ "article": str, "views": int, "timestamp": "YYYYMMDDHH" }, ...] }
"""
import time, requests, pandas as pd
from pathlib import Path

OUT = Path("data/raw") ; OUT.mkdir(parents=True, exist_ok=True)
ARTICLES = [
    "COVID-19",
    "Symptoms_of_COVID-19",
    "Influenza",
    "Common_cold",
    "Cough",
    "Fever",
    "Sore_throat",
    "Anosmia",            # loss of smell article
    "Ageusia",            # loss of taste article
    "Respiratory_syncytial_virus",
]
START = "2021090100"
END   = "2025093000"

base = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}")
HEADERS = {"User-Agent": "AIxBio-Hackathon/1.0 (taggarttufte@gmail.com)"}

frames = []
for art in ARTICLES:
    url = base.format(article=art, start=START, end=END)
    print(f"  {art}...", end=" ")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"FAIL {r.status_code}")
        continue
    items = r.json().get("items", [])
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df.timestamp.str[:8], format="%Y%m%d")
    df = df[["date", "views"]].rename(columns={"views": art})
    frames.append(df.set_index("date"))
    print(f"{len(df):,} rows")
    time.sleep(0.1)

out = pd.concat(frames, axis=1).sort_index()
out.to_csv(OUT / "wikipedia_pageviews.csv")
print(f"\nsaved {OUT/'wikipedia_pageviews.csv'}  shape={out.shape}")
print(f"date range: {out.index.min().date()} -> {out.index.max().date()}")
print("daily-mean pageviews per article:")
print(out.mean().sort_values(ascending=False).round(0).to_string())
