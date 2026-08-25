"""04_attention_decay.py — quantify the central claim:

  "Attention-based surveillance signals (Google Trends, Wikipedia) lose
  correlation with the COVID ground truth over time, while wastewater (NWSS)
  maintains its correlation. This is the attention-decay failure mode."

Method:
  - Use HHS COVID hospital admissions (Nevada + US) as ground truth.
  - For each signal, compute weekly correlation with HHS admissions per year.
  - Plot signals z-scored on shared axes for visual confirmation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import WAVE_PEAKS

FIG_DIR = Path("notebooks/fig") ; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- load all signals onto a daily index, weekly resample later ------------
SITE = "NWSS_nv_554_Treatment plant_90_raw wastewater"
conc = pd.read_csv("data/raw/nwss_concentration.csv", parse_dates=["date"])
conc = conc[(conc.key_plot_id == SITE) & (conc.normalization == "flow-population")]
nwss = (conc.groupby("date").pcr_conc_lin.mean().asfreq("D")
            .clip(lower=1.0)
            .interpolate(method="time", limit=7))
nwss = np.log10(nwss).rolling("14D", min_periods=3).mean()

trends = (pd.read_csv("data/raw/trends_nv_us.csv", parse_dates=["date"])
            .set_index("date"))
trends_nv = trends["US-NV::covid symptoms"].astype(float)
trends_us = trends["US::covid symptoms"].astype(float)

wiki = pd.read_csv("data/raw/wikipedia_pageviews.csv", parse_dates=["date"]).set_index("date")
wiki_signal = (wiki["Symptoms_of_COVID-19"] + wiki["Anosmia"] + wiki["Ageusia"])
wiki_signal_smooth = wiki_signal.rolling("14D", min_periods=3).mean()

truth = pd.read_csv("data/raw/covid_ground_truth.csv", parse_dates=["date"]).set_index("date")
hhs_nv = truth["hhs_admits_NV"].rolling("7D", min_periods=2).mean()
hhs_us = truth["hhs_admits_US"].rolling("7D", min_periods=2).mean()

# ---- weekly resample (Sunday-anchored) -------------------------------------
def to_weekly(s, name=None):
    name = name or s.name
    out = s.resample("W-SUN").mean()
    out.name = name
    return out

weekly = pd.concat([
    to_weekly(nwss,             "NWSS_log10"),
    to_weekly(trends_nv,        "Trends_NV"),
    to_weekly(trends_us,        "Trends_US"),
    to_weekly(wiki_signal_smooth, "Wiki_signal"),
    to_weekly(hhs_nv,           "HHS_NV"),
    to_weekly(hhs_us,           "HHS_US"),
], axis=1).dropna(how="all")

# ---- per-year correlation against HHS admissions ---------------------------
print("=== per-year correlation with HHS COVID admissions ===\n")
years = [2022, 2023, 2024]
truth_signal = "HHS_NV"          # primary ground truth
print(f"Ground truth: {truth_signal}\n")

rows = []
for year in years:
    yr = weekly.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(yr) < 10:
        continue
    row = {"year": year, "n_weeks": len(yr.dropna(subset=[truth_signal]))}
    for sig in ["NWSS_log10", "Trends_NV", "Trends_US", "Wiki_signal"]:
        valid = yr[[sig, truth_signal]].dropna()
        if len(valid) >= 8:
            row[sig] = valid[sig].corr(valid[truth_signal])
        else:
            row[sig] = np.nan
    rows.append(row)

corr_table = pd.DataFrame(rows).set_index("year")
print(corr_table.round(3).to_string())
corr_table.to_csv(FIG_DIR / "04_corr_by_year.csv")

# ---- per-wave detection at fixed threshold (z-score >= 2) ------------------
print("\n=== per-wave detection at z>=2 threshold (each signal z-scored on full window) ===\n")

def z_full(s):
    return (s - s.mean()) / s.std(ddof=0)

z = pd.DataFrame({
    "NWSS_log10":  z_full(weekly["NWSS_log10"]),
    "Trends_NV":   z_full(weekly["Trends_NV"]),
    "Wiki_signal": z_full(weekly["Wiki_signal"]),
    "HHS_NV":      z_full(weekly["HHS_NV"]),
})

WAVE_PEAKS_TS = {k: pd.Timestamp(v) for k, v in WAVE_PEAKS.items()}
detect_rows = []
for name, peak in WAVE_PEAKS_TS.items():
    win_lo = peak - pd.Timedelta(days=60)
    win_hi = peak + pd.Timedelta(days=14)
    if peak < z.index.min() or peak > z.index.max():
        continue
    row = {"wave": name, "peak": peak.date()}
    for sig in z.columns:
        wnd = z.loc[win_lo:win_hi, sig].dropna()
        if len(wnd) == 0:
            row[sig] = "—"
            continue
        crossed = wnd >= 2.0
        if crossed.any():
            first = wnd.index[crossed][0]
            lead = (peak - first).days
            row[sig] = f"YES (z>=2 at -{lead}d)"
        else:
            row[sig] = "no"
    detect_rows.append(row)

detect = pd.DataFrame(detect_rows)
print(detect.to_string(index=False))
detect.to_csv(FIG_DIR / "04_detection_by_wave.csv", index=False)

# ---- plot: per-year correlation bar chart ----------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_cols = ["NWSS_log10", "Trends_NV", "Wiki_signal"]
labels    = ["NWSS wastewater", "Google Trends NV", "Wikipedia COVID"]
colors    = ["C0", "C2", "C3"]
x = np.arange(len(corr_table.index))
w = 0.25
for i, (col, lab, c) in enumerate(zip(plot_cols, labels, colors)):
    ax.bar(x + i * w, corr_table[col], width=w, label=lab, color=c)
ax.set_xticks(x + w)
ax.set_xticklabels(corr_table.index)
ax.set_ylabel(f"Pearson r vs. {truth_signal}")
ax.set_title("Attention-decay: signal correlation with COVID hospital admissions, by year")
ax.axhline(0, color="grey", lw=0.5)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_corr_by_year.png", dpi=140)
print(f"\nsaved {FIG_DIR / '04_corr_by_year.png'}")

# ---- plot: z-scored signals overlaid with HHS admissions -------------------
fig, ax = plt.subplots(figsize=(13, 5))
plot_signals = [
    ("HHS_NV",      "HHS COVID admissions (NV)", "k",  2.0, 0.9),
    ("NWSS_log10",  "NWSS wastewater",           "C0", 1.2, 0.7),
    ("Trends_NV",   "Google Trends",             "C2", 1.0, 0.7),
    ("Wiki_signal", "Wikipedia COVID",           "C3", 1.0, 0.7),
]
for col, lab, color, lw, alpha in plot_signals:
    ax.plot(z.index, z[col], color=color, lw=lw, alpha=alpha, label=lab)

for n, d in WAVE_PEAKS_TS.items():
    if z.index.min() <= d <= z.index.max():
        ax.axvline(d, color="grey", lw=0.5, alpha=0.4, ls="--")
        ax.text(d, 0.99, n, transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=7, color="grey")

ax.axhline(2, color="red", lw=0.8, ls=":", alpha=0.7, label="z=2 detection threshold")
ax.set_ylabel("z-score")
ax.set_xlabel("week")
ax.set_title("Attention-decay visualization: BA.1 spike trains the threshold; later waves "
             "fire on NWSS but not on Trends/Wikipedia")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(z.index.min(), pd.Timestamp("2024-05-01"))  # HHS truth ends here
fig.tight_layout()
fig.savefig(FIG_DIR / "04_attention_decay_plot.png", dpi=140)
print(f"saved {FIG_DIR / '04_attention_decay_plot.png'}")
