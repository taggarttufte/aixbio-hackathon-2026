"""05_attention_decay_national.py — re-run the attention-decay analysis using
population-weighted national NWSS aggregate (instead of just Reno).

Compares national-scale signals against national-scale ground truth, removing
the geographic mismatch from the previous analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import WAVE_PEAKS

FIG_DIR = Path("notebooks/fig") ; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- load aggregated NWSS --------------------------------------------------
agg = pd.read_csv("data/processed/nwss_aggregated.csv", parse_dates=["date"]).set_index("date")
nwss_nat = agg["national"].rolling("14D", min_periods=3).mean()
nwss_nv  = agg["nevada"].rolling("14D", min_periods=3).mean()

# ---- other signals ---------------------------------------------------------
trends = pd.read_csv("data/raw/trends_nv_us.csv", parse_dates=["date"]).set_index("date")
trends_us = trends["US::covid symptoms"].astype(float)
trends_nv = trends["US-NV::covid symptoms"].astype(float)

wiki = pd.read_csv("data/raw/wikipedia_pageviews.csv", parse_dates=["date"]).set_index("date")
wiki_signal = (wiki["Symptoms_of_COVID-19"] + wiki["Anosmia"] + wiki["Ageusia"])
wiki_signal = wiki_signal.rolling("14D", min_periods=3).mean()

truth = pd.read_csv("data/raw/covid_ground_truth.csv", parse_dates=["date"]).set_index("date")
hhs_us = truth["hhs_admits_US"].rolling("7D", min_periods=2).mean()
hhs_nv = truth["hhs_admits_NV"].rolling("7D", min_periods=2).mean()

# ---- weekly resample -------------------------------------------------------
def to_weekly(s, name=None):
    name = name or s.name
    out = s.resample("W-SUN").mean() ; out.name = name ; return out

weekly = pd.concat([
    to_weekly(nwss_nat,    "NWSS_national"),
    to_weekly(nwss_nv,     "NWSS_nevada"),
    to_weekly(trends_us,   "Trends_US"),
    to_weekly(trends_nv,   "Trends_NV"),
    to_weekly(wiki_signal, "Wiki"),
    to_weekly(hhs_us,      "HHS_US"),
    to_weekly(hhs_nv,      "HHS_NV"),
], axis=1).dropna(how="all")

# Trim to HHS-trustworthy window (HHS reporting reliable through ~Apr 2024)
ANALYSIS_END = pd.Timestamp("2024-04-30")
analysis = weekly.loc[:ANALYSIS_END].copy()

# ---- per-year correlation: NATIONAL ground truth ---------------------------
print("=== per-year correlation with HHS_US (national) ===\n")
years = [2022, 2023, 2024]
truth_col = "HHS_US"

rows = []
for year in years:
    yr = analysis.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(yr) < 10:
        continue
    row = {"year": year, "n_weeks": len(yr.dropna(subset=[truth_col]))}
    for sig in ["NWSS_national", "Trends_US", "Wiki"]:
        valid = yr[[sig, truth_col]].dropna()
        if len(valid) >= 8:
            row[sig] = valid[sig].corr(valid[truth_col])
        else:
            row[sig] = np.nan
    rows.append(row)
corr_table = pd.DataFrame(rows).set_index("year")
print(corr_table.round(3).to_string())
corr_table.to_csv(FIG_DIR / "05_corr_by_year_national.csv")

# ---- variance ratio: BA.1 era vs post-BA.1 ---------------------------------
print("\n=== variance ratio: BA.1 era vs post-BA.1 ===\n")
BA1_LO = pd.Timestamp("2021-12-01")  ; BA1_HI = pd.Timestamp("2022-02-28")
POST_LO = pd.Timestamp("2022-03-01") ; POST_HI = ANALYSIS_END
print(f"  BA.1 window:  {BA1_LO.date()} -> {BA1_HI.date()}")
print(f"  post window:  {POST_LO.date()} -> {POST_HI.date()}\n")

vr_rows = []
for sig in ["NWSS_national", "Trends_US", "Wiki", "HHS_US"]:
    s = analysis[sig].dropna()
    ba1 = s.loc[BA1_LO:BA1_HI]
    post = s.loc[POST_LO:POST_HI]
    var_ratio = ba1.var(ddof=1) / post.var(ddof=1) if post.var(ddof=1) > 0 else np.nan
    amp_ratio = (ba1.max() - ba1.min()) / max(post.max() - post.min(), 1e-9)
    vr_rows.append({"signal": sig, "var_ratio_BA1_to_post": var_ratio,
                    "amp_ratio_BA1_to_post": amp_ratio,
                    "ba1_max": ba1.max(), "post_max": post.max()})
vr = pd.DataFrame(vr_rows)
print(vr.round(2).to_string(index=False))
vr.to_csv(FIG_DIR / "05_variance_ratio.csv", index=False)

# ---- overlay plot, national signals ----------------------------------------
def z_full(s):
    s = s.dropna()
    return (s - s.mean()) / s.std(ddof=0)

z = pd.DataFrame({
    "NWSS_national": z_full(analysis["NWSS_national"]),
    "Trends_US":     z_full(analysis["Trends_US"]),
    "Wiki":          z_full(analysis["Wiki"]),
    "HHS_US":        z_full(analysis["HHS_US"]),
})

fig, ax = plt.subplots(figsize=(13, 5.5))
plot_signals = [
    ("HHS_US",        "HHS COVID admissions (US)", "k",  2.2, 0.95),
    ("NWSS_national", "NWSS national wastewater",  "C0", 1.4, 0.85),
    ("Trends_US",     "Google Trends (US)",        "C2", 1.0, 0.80),
    ("Wiki",          "Wikipedia COVID articles",  "C3", 1.0, 0.80),
]
for col, lab, color, lw, alpha in plot_signals:
    ax.plot(z.index, z[col], color=color, lw=lw, alpha=alpha, label=lab)

for n, d in WAVE_PEAKS.items():
    d = pd.Timestamp(d)
    if z.index.min() <= d <= z.index.max():
        ax.axvline(d, color="grey", lw=0.5, alpha=0.4, ls="--")
        ax.text(d, 0.99, n, transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=7, color="grey")

ax.axhline(2, color="red", lw=0.8, ls=":", alpha=0.7, label="z=2")
ax.set_ylabel("z-score (computed over the analysis window)")
ax.set_xlabel("week")
ax.set_title(f"Attention-decay (national-level): NWSS aggregates 621 sites; "
             f"Trends/Wiki spike at BA.1 then decay despite continuing waves")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "05_attention_decay_national.png", dpi=140)
print(f"\nsaved {FIG_DIR / '05_attention_decay_national.png'}")

# ---- correlation bar chart -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_cols = ["NWSS_national", "Trends_US", "Wiki"]
labels    = ["NWSS national", "Google Trends US", "Wikipedia"]
colors    = ["C0", "C2", "C3"]
xpos = np.arange(len(corr_table.index))
w = 0.25
for i, (col, lab, c) in enumerate(zip(plot_cols, labels, colors)):
    ax.bar(xpos + i * w, corr_table[col], width=w, label=lab, color=c)
ax.set_xticks(xpos + w)
ax.set_xticklabels(corr_table.index)
ax.set_ylabel(f"Pearson r vs. HHS_US (national)")
ax.set_title("Per-year correlation with national COVID admissions")
ax.axhline(0, color="grey", lw=0.5)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "05_corr_by_year_national.png", dpi=140)
print(f"saved {FIG_DIR / '05_corr_by_year_national.png'}")
