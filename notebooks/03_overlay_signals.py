"""03_overlay_signals.py — overlay NWSS, Google Trends, Wikipedia pageviews,
and ILINet on a single timeline. Each signal z-scored for visual comparison.

This is the "what does each signal look like across waves" plot. We're
checking visually that:
  - NWSS leads
  - Trends + Wikipedia track illness behavior
  - ILINet confirms waves but lags
  - All four show coherent wave structure
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import WAVE_PEAKS

FIG_DIR = Path("notebooks/fig") ; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. NWSS daily, log10, smoothed ---------------------------------------
SITE = "NWSS_nv_554_Treatment plant_90_raw wastewater"
conc = pd.read_csv("data/raw/nwss_concentration.csv", parse_dates=["date"])
conc = conc[(conc.key_plot_id == SITE) & (conc.normalization == "flow-population")]
nwss = (conc.groupby("date").pcr_conc_lin.mean()
            .asfreq("D")
            .clip(lower=1.0)
            .interpolate(method="time", limit=7))
nwss = np.log10(nwss).rolling("14D", min_periods=3).mean()
nwss.name = "NWSS_log10_14d"

# ---- 2. Google Trends weekly, NV "covid symptoms" -------------------------
trends = pd.read_csv("data/raw/trends_nv_us.csv", parse_dates=["date"])
trends = trends.set_index("date")["US-NV::covid symptoms"].astype(float)
# Upsample weekly -> daily by forward-fill within the week
trends_d = trends.resample("D").ffill()
trends_d.name = "Trends_NV_covid_symptoms"

# ---- 3. Wikipedia daily pageviews, multiple articles, smoothed -------------
wiki = pd.read_csv("data/raw/wikipedia_pageviews.csv", parse_dates=["date"]).set_index("date")
# Combine the more sensitive articles into one signal (sum of disease-specific)
wiki_signal = (wiki["Symptoms_of_COVID-19"] +
               wiki["Anosmia"] +
               wiki["Ageusia"] +
               wiki["COVID-19"])
wiki_signal = wiki_signal.rolling("14D", min_periods=3).mean()
wiki_signal.name = "Wikipedia_covid_articles"

# ---- 4. ILINet weekly wILI for HHS9 ---------------------------------------
ili = pd.read_csv("data/raw/ilinet.csv", parse_dates=["week_start"])
ili9 = ili[ili.region_code == "hhs9"].set_index("week_start").wili.astype(float)
ili9_d = ili9.resample("D").ffill()
ili9_d.name = "ILINet_HHS9_wili"

# ---- z-score each on its own range for plotting ---------------------------
def z(s, win="365D"):
    """Robust z-score: subtract median, divide by IQR-based std equivalent."""
    return (s - s.median()) / (s.quantile(0.75) - s.quantile(0.25)).clip(min=1e-9)

signals = pd.DataFrame({
    "NWSS (Reno wastewater, log10)":          z(nwss),
    "Google Trends (NV, 'covid symptoms')":   z(trends_d),
    "Wikipedia (covid articles, 14d)":        z(wiki_signal),
    "ILINet (HHS9 wILI %)":                   z(ili9_d),
})

# Trim to common range
common_start = pd.Timestamp("2021-09-15")
common_end   = pd.Timestamp("2025-07-01")
signals = signals.loc[common_start:common_end]

# ---- plot -----------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
colors = ["C0", "C2", "C3", "C4"]
for ax, (name, vals), color in zip(axes, signals.items(), colors):
    ax.plot(vals.index, vals.values, color=color, lw=1.0, alpha=0.85)
    ax.set_ylabel(name.split(" (")[0], fontsize=9)
    ax.set_title(name, fontsize=9, loc="left", color=color)
    ax.axhline(0, color="grey", lw=0.4, alpha=0.5)
    for n, d in WAVE_PEAKS.items():
        d = pd.Timestamp(d)
        if signals.index.min() <= d <= signals.index.max():
            ax.axvline(d, color="grey", lw=0.6, alpha=0.4, ls="--")
            if ax is axes[0]:
                ax.text(d, 1.02, n, transform=ax.get_xaxis_transform(),
                        rotation=90, va="bottom", ha="right",
                        fontsize=7, color="grey")
fig.suptitle("Multi-signal overlay (each signal z-scored to its own scale)",
             fontsize=11, y=0.995)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_overlay_signals.png", dpi=140)
print(f"saved {FIG_DIR / '03_overlay_signals.png'}")

# ---- pairwise correlations on weekly resample ------------------------------
weekly = signals.resample("W").mean()
corr = weekly.corr().round(3)
print("\nweekly cross-correlation:")
print(corr.to_string())
corr.to_csv(FIG_DIR / "03_corr_weekly.csv")

# ---- lead/lag analysis (cross-correlation) --------------------------------
def lag_correlation(a: pd.Series, b: pd.Series, max_lag_weeks=12):
    """Return DataFrame of corr(a shifted by lag, b) for lag in [-max,+max]
    weekly bins. Positive lag = a leads b."""
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    lags, corrs = [], []
    for L in range(-max_lag_weeks, max_lag_weeks + 1):
        if L >= 0:
            c = a.shift(L).corr(b)
        else:
            c = a.shift(L).corr(b)
        lags.append(L) ; corrs.append(c)
    return pd.DataFrame({"lag_weeks": lags, "corr": corrs})

print("\n=== lead/lag of each signal vs ILINet (positive lag = signal leads ILINet) ===")
for col in signals.columns:
    if "ILINet" in col:
        continue
    df = lag_correlation(weekly[col], weekly["ILINet (HHS9 wILI %)"])
    best = df.loc[df["corr"].idxmax()]
    print(f"  {col:<45}  best lag = {int(best.lag_weeks):+d} wk  corr = {best['corr']:.3f}")
