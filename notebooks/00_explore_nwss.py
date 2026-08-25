"""
00_explore_nwss.py — First-pass exploration of NWSS SARS-CoV-2 data.

Goal: pick one densely-sampled site, plot concentration over time, overlay
known US COVID wave peaks, sanity-check that NWSS signal tracks them.

Run: `python notebooks/00_explore_nwss.py`
Outputs: notebooks/fig/00_*.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FIG_DIR = Path("notebooks/fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Known US COVID wave peaks (approximate, from CDC reporting)
WAVE_PEAKS = {
    "Delta":    "2021-09-01",
    "BA.1":     "2022-01-15",
    "BA.2":     "2022-04-20",
    "BA.5":     "2022-07-20",
    "XBB/BQ":   "2023-01-05",
    "EG.5":     "2023-09-05",
    "JN.1":     "2024-01-05",
    "KP.3":     "2024-08-10",
}

# ---- load ------------------------------------------------------------------
print("loading concentration...")
conc = pd.read_csv(
    "data/raw/nwss_concentration.csv",
    parse_dates=["date"],
)
# Keep flow-population normalization (compatible with CDC's own reporting)
conc = conc[conc.normalization == "flow-population"].copy()

print("loading metric (for geo metadata)...")
meta_cols = ["key_plot_id", "wwtp_jurisdiction", "county_names",
             "population_served", "first_sample_date"]
m = pd.read_csv(
    "data/raw/nwss_metric.csv",
    usecols=meta_cols,
    parse_dates=["first_sample_date"],
)
# One row per site for metadata
site_meta = m.drop_duplicates("key_plot_id").set_index("key_plot_id")

# ---- pick site -------------------------------------------------------------
site = "NWSS_nv_554_Treatment plant_90_raw wastewater"
s = conc[conc.key_plot_id == site].sort_values("date").reset_index(drop=True)
meta = site_meta.loc[site]
print(f"\nSITE: {site}")
print(f"  state={meta.wwtp_jurisdiction}  county={meta.county_names}")
print(f"  pop_served={meta.population_served:,.0f}")
print(f"  n_samples={len(s):,}  "
      f"range={s.date.min().date()} -> {s.date.max().date()}")

# Deduplicate dates (sometimes multiple samples/day)
s = s.groupby("date", as_index=False).pcr_conc_lin.mean()

# ---- plot 1: linear + log concentration with wave peaks --------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax in axes:
    for name, d in WAVE_PEAKS.items():
        d = pd.Timestamp(d)
        if s.date.min() <= d <= s.date.max():
            ax.axvline(d, color="grey", lw=0.8, alpha=0.5, ls="--")
            ax.text(d, ax.get_ylim()[1] if False else 0.97, name,
                    rotation=90, va="top", ha="right",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8, color="grey")

axes[0].plot(s.date, s.pcr_conc_lin, "o-", ms=2, lw=0.6)
axes[0].set_ylabel("PCR conc (copies / ...)\n[linear]")
axes[0].set_title(f"NWSS SARS-CoV-2, site {site.split('_')[2].upper()} "
                  f"({meta.wwtp_jurisdiction}, pop {meta.population_served:,.0f})")

# Log needs strictly positive values; replace 0/neg with NaN
y = s.pcr_conc_lin.where(s.pcr_conc_lin > 0)
axes[1].semilogy(s.date, y, "o-", ms=2, lw=0.6)
axes[1].set_ylabel("PCR conc [log scale]")
axes[1].set_xlabel("date")
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].xaxis.set_minor_locator(mdates.MonthLocator())

fig.tight_layout()
fig.savefig(FIG_DIR / "00_single_site_timeseries.png", dpi=140)
print(f"saved {FIG_DIR / '00_single_site_timeseries.png'}")

# ---- plot 2: rolling mean + std (baseline visual for control chart) --------
# Resample to daily, forward-fill up to 7 days (sampling cadence is ~2x/wk)
daily = (
    s.set_index("date")
     .asfreq("D")
     .pcr_conc_lin
     .interpolate(method="time", limit=7)
)
# 28-day rolling baseline
mu = daily.rolling("28D", min_periods=10).mean()
sd = daily.rolling("28D", min_periods=10).std()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily.values, lw=0.6, alpha=0.5, label="daily (interp)")
ax.plot(mu.index, mu.values, lw=1.8, label="28d mean")
ax.fill_between(mu.index, mu - 2 * sd, mu + 2 * sd, alpha=0.15,
                label="±2σ band")
for name, d in WAVE_PEAKS.items():
    d = pd.Timestamp(d)
    if daily.index.min() <= d <= daily.index.max():
        ax.axvline(d, color="grey", lw=0.8, alpha=0.4, ls="--")
ax.set_yscale("log")
ax.set_title("Rolling baseline (28-day) with ±2σ band — what a control chart sees")
ax.set_ylabel("PCR conc [log]")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "00_rolling_baseline.png", dpi=140)
print(f"saved {FIG_DIR / '00_rolling_baseline.png'}")

# ---- summary stats ---------------------------------------------------------
print("\n=== descriptive stats ===")
print(f"non-null days (daily grid, after fill): {daily.notna().sum():,}")
print(f"total days in span: {(daily.index.max() - daily.index.min()).days:,}")
print(f"pct zeros in raw: {(s.pcr_conc_lin <= 0).mean():.1%}")
print(f"raw pcr_conc_lin: min={s.pcr_conc_lin.min():.2g}  "
      f"med={s.pcr_conc_lin.median():.2g}  "
      f"max={s.pcr_conc_lin.max():.2g}")
