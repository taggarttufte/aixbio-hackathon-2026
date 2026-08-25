"""07_polish_figures.py — re-render the headline figures with consistent
publication-quality aesthetics for the paper. Output goes directly to
paper/figs/.

Figure mapping:
  Fig 1 — single-site MA28 control chart (NWSS Reno COVID)
  Fig 2 — national multi-signal overlay (COVID)
  Fig 3 — COVID vs flu side-by-side (HEADLINE)
  Fig 4 — variance ratio bar chart
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = Path("paper/figs") ; OUT.mkdir(parents=True, exist_ok=True)

# ---- consistent style ------------------------------------------------------
mpl.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize":  9.5,
    "figure.dpi":      130,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "lines.linewidth":   1.3,
})
# Palette: deliberate consistent assignment per signal type
COL = {
    "nwss":   "#1f77b4",  # blue
    "trends": "#2ca02c",  # green
    "wiki":   "#d62728",  # red
    "ili":    "#9467bd",  # purple
    "hhs":    "#000000",  # black for ground truth
    "alert":  "#e63946",
}

WAVE_PEAKS_COVID = {
    "Delta":  "2021-09-01", "BA.1": "2022-01-15", "BA.2": "2022-04-20",
    "BA.5":   "2022-07-20", "XBB":  "2023-01-05", "EG.5": "2023-09-05",
    "JN.1":   "2024-01-05", "KP.3": "2024-08-10",
}

# ---- load all signals once -------------------------------------------------
print("loading...")
SITE = "NWSS_nv_554_Treatment plant_90_raw wastewater"
conc = pd.read_csv("data/raw/nwss_concentration.csv", parse_dates=["date"])
conc_reno = conc[(conc.key_plot_id == SITE) & (conc.normalization == "flow-population")]
nwss_reno = (conc_reno.groupby("date").pcr_conc_lin.mean().asfreq("D")
                   .clip(lower=1.0).interpolate(method="time", limit=7))
nwss_reno_log = np.log10(nwss_reno)

agg_covid = pd.read_csv("data/processed/nwss_aggregated.csv",
                         parse_dates=["date"]).set_index("date")
agg_flu = pd.read_csv("data/processed/nwss_flu_aggregated.csv",
                       parse_dates=["date"]).set_index("date")
nwss_nat_covid = agg_covid["national"].rolling("14D", min_periods=3).mean()
nwss_nat_flu   = agg_flu["national"].rolling("14D", min_periods=3).mean()

trends = pd.read_csv("data/raw/trends_nv_us.csv",
                     parse_dates=["date"]).set_index("date")
trends_us_covid = trends["US::covid symptoms"].astype(float)
trends_us_flu   = trends[["US::fever", "US::cough", "US::sore throat",
                           "US::body aches", "US::congestion"]].mean(axis=1)

wiki = pd.read_csv("data/raw/wikipedia_pageviews.csv",
                   parse_dates=["date"]).set_index("date")
wiki_covid = (wiki["Symptoms_of_COVID-19"] + wiki["Anosmia"] + wiki["Ageusia"])
wiki_covid = wiki_covid.rolling("14D", min_periods=3).mean()
wiki_flu = (wiki["Influenza"] + wiki["Common_cold"] + wiki["Cough"]
             + wiki["Fever"] + wiki["Sore_throat"])
wiki_flu = wiki_flu.rolling("14D", min_periods=3).mean()

ili = pd.read_csv("data/raw/ilinet.csv", parse_dates=["week_start"])
ili_us = ili[ili.region_code == "nat"].set_index("week_start").wili.astype(float)

covid_truth = pd.read_csv("data/raw/covid_ground_truth.csv",
                           parse_dates=["date"]).set_index("date")
hhs_covid = covid_truth["hhs_admits_US"].rolling("7D", min_periods=2).mean()

flu_truth = pd.read_csv("data/raw/hhs_flu_admissions.csv",
                         parse_dates=["date"]).set_index("date")
hhs_flu = flu_truth["hhs_flu_US"].rolling("7D", min_periods=2).mean()

# ---- helpers ---------------------------------------------------------------
def annotate_waves(ax, peaks, ymax_text=0.99, fontsize=7.5):
    ymin, ymax = ax.get_ylim()
    for name, d in peaks.items():
        d = pd.Timestamp(d)
        if ax.get_xlim()[0] <= mpl.dates.date2num(d) <= ax.get_xlim()[1]:
            ax.axvline(d, color="grey", lw=0.5, alpha=0.4, ls="--")
            ax.text(d, ymax_text, name, rotation=90,
                    transform=ax.get_xaxis_transform(),
                    va="top", ha="right", fontsize=fontsize, color="grey")

def weekly_smart(s, name=None):
    s = s.dropna()
    if len(s) == 0:
        return s.resample("W-SUN").mean()
    gap = s.index.to_series().diff().dt.days.dropna().median()
    if gap <= 2:
        s = s.rolling("14D", min_periods=3).mean()
    out = s.resample("W-SUN").mean() ; out.name = name or s.name ; return out

def z_full(s):
    s = s.dropna()
    return (s - s.mean()) / s.std(ddof=0) if s.std() else s * 0

# ===========================================================================
# Fig 1 — single-site MA28 control chart on NWSS Reno
# ===========================================================================
print("fig 1: single-site control chart...")
from src.detectors import moving_average_control_chart

s = nwss_reno_log.loc["2021-10-01":"2025-07-01"]
out = moving_average_control_chart(s, window="28D", k=2.5, gap_days=3)

fig, ax = plt.subplots(figsize=(12.6, 4.6))   # +15%
ax.plot(s.index, s.values, color=COL["nwss"], lw=0.8, alpha=0.7,
        label="NWSS Reno (log₁₀ conc, daily)")
ax.plot(out.index, out.baseline_mean, color="#ff8c00", lw=1.5, alpha=0.9,
        label="28-day rolling baseline (μ)")
ax.plot(out.index, out.threshold, color=COL["alert"], lw=1.0, alpha=0.7,
        label="threshold (μ + 2.5σ)")
alert_t = out.index[out.alert]
ax.scatter(alert_t, s.reindex(alert_t), color=COL["alert"], s=22,
           zorder=5, label="alert", edgecolor="white", linewidth=0.5)
ax.set_xlim(s.index.min(), s.index.max())
ax.set_ylabel("log₁₀(SARS-CoV-2 PCR copies)")
ax.set_xlabel("date")
annotate_waves(ax, WAVE_PEAKS_COVID)
ax.legend(loc="lower left", ncol=2, framealpha=0.95)
fig.savefig(OUT / "fig1_control_chart.png")
plt.close(fig)

# ===========================================================================
# Fig 2 — national multi-signal overlay (COVID), z-scored
# ===========================================================================
print("fig 2: national overlay...")
W = pd.concat([
    weekly_smart(nwss_nat_covid,  "NWSS"),
    weekly_smart(trends_us_covid, "Trends"),
    weekly_smart(wiki_covid,      "Wiki"),
    weekly_smart(hhs_covid,       "HHS"),
], axis=1)
W = W.loc[:"2024-04-30"].dropna(how="all")

z = pd.DataFrame({c: z_full(W[c]) for c in W.columns})

fig, ax = plt.subplots(figsize=(12.6, 5.2))   # +15%
ax.plot(z.index, z["HHS"],    color=COL["hhs"],   lw=2.2, alpha=0.95,
        label="HHS COVID admissions (US)")
ax.plot(z.index, z["NWSS"],   color=COL["nwss"],  lw=1.7, alpha=0.85,
        label="NWSS national wastewater (621 sites)")
ax.plot(z.index, z["Trends"], color=COL["trends"], lw=1.2, alpha=0.80,
        label="Google Trends 'covid symptoms' (US)")
ax.plot(z.index, z["Wiki"],   color=COL["wiki"],   lw=1.2, alpha=0.80,
        label="Wikipedia COVID articles")
ax.axhline(2, color=COL["alert"], lw=0.8, ls=":", alpha=0.6, label="z = 2")
ax.axhline(0, color="grey", lw=0.4)
ax.set_ylabel("z-score (over analysis window)")
ax.set_xlabel("week")
annotate_waves(ax, WAVE_PEAKS_COVID)
ax.legend(loc="upper right", ncol=2, framealpha=0.95)
fig.savefig(OUT / "fig2_national_overlay.png")
plt.close(fig)

# ===========================================================================
# Fig 3 — COVID vs flu side-by-side (HEADLINE)
# ===========================================================================
print("fig 3: COVID vs flu side-by-side...")
Wc = pd.concat([
    weekly_smart(hhs_covid,        "HHS"),
    weekly_smart(nwss_nat_covid,   "NWSS"),
    weekly_smart(trends_us_covid,  "Trends"),
    weekly_smart(wiki_covid,       "Wiki"),
], axis=1).loc[:"2024-04-30"]
Wf = pd.concat([
    weekly_smart(hhs_flu,        "HHS"),
    weekly_smart(nwss_nat_flu,   "NWSS"),
    weekly_smart(trends_us_flu,  "Trends"),
    weekly_smart(wiki_flu,       "Wiki"),
    weekly_smart(ili_us,         "ILI"),
], axis=1).loc["2021-09-01":"2025-09-30"]

zc = pd.DataFrame({c: z_full(Wc[c]) for c in Wc.columns})
zf = pd.DataFrame({c: z_full(Wf[c]) for c in Wf.columns})

# Approximate flu season peaks (reference dates for annotation)
FLU_PEAKS = {
    "2021-22 (early)": "2021-12-15",
    "2022-23 (post-pandemic rebound)": "2022-12-15",
    "2023-24": "2024-01-15",
    "2024-25": "2025-02-01",
}

fig, (ax_c, ax_f) = plt.subplots(2, 1, figsize=(12.6, 8.8))   # +5%

ax_c.plot(zc.index, zc["HHS"],    color=COL["hhs"],    lw=2.2, label="HHS COVID admissions (US)")
ax_c.plot(zc.index, zc["NWSS"],   color=COL["nwss"],   lw=1.7, label="NWSS national wastewater")
ax_c.plot(zc.index, zc["Trends"], color=COL["trends"], lw=1.2, label="Google Trends 'covid symptoms'")
ax_c.plot(zc.index, zc["Wiki"],   color=COL["wiki"],   lw=1.2, label="Wikipedia COVID articles")
ax_c.axhline(2, color=COL["alert"], lw=0.8, ls=":", alpha=0.6, label="z = 2")
ax_c.axhline(0, color="grey", lw=0.3)
ax_c.set_ylabel("z-score")
ax_c.set_title("(a) COVID-19", loc="left", fontsize=11.5, fontweight="bold")
annotate_waves(ax_c, WAVE_PEAKS_COVID)
ax_c.legend(loc="upper right", ncol=2, framealpha=0.95)

ax_f.plot(zf.index, zf["HHS"],    color=COL["hhs"],    lw=2.2, label="HHS flu admissions (US)")
ax_f.plot(zf.index, zf["NWSS"],   color=COL["nwss"],   lw=1.7, label="NWSS national wastewater (Flu A)")
ax_f.plot(zf.index, zf["Trends"], color=COL["trends"], lw=1.2, label="Google Trends (flu symptom cluster)")
ax_f.plot(zf.index, zf["Wiki"],   color=COL["wiki"],   lw=1.2, label="Wikipedia flu articles")
ax_f.plot(zf.index, zf["ILI"],    color=COL["ili"],    lw=1.7, label="CDC ILINet (national wILI %)")
ax_f.axhline(2, color=COL["alert"], lw=0.8, ls=":", alpha=0.6, label="z = 2")
ax_f.axhline(0, color="grey", lw=0.3)
ax_f.set_ylabel("z-score")
ax_f.set_xlabel("week")
ax_f.set_title("(b) Influenza", loc="left", fontsize=11.5, fontweight="bold")
annotate_waves(ax_f, FLU_PEAKS, fontsize=8)
ax_f.legend(loc="upper right", ncol=2, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT / "fig3_covid_vs_flu.png")
plt.close(fig)

# ===========================================================================
# Fig 4 — variance ratio bars
# ===========================================================================
print("fig 4: variance ratio bars...")
vr = pd.read_csv("notebooks/fig/06_variance_ratio_pivot.csv").set_index("signal")
sigs = ["NWSS national", "Google Trends", "Wikipedia", "HHS admissions"]
vr = vr.reindex(sigs)

fig, ax = plt.subplots(figsize=(9.8, 5.0))   # +15%
x = np.arange(len(sigs))
w = 0.36
bars_c = ax.bar(x - w/2, vr["COVID"], width=w, color=COL["wiki"], alpha=0.9,
                label="COVID-19 (BA.1 vs post-BA.1)", edgecolor="white")
bars_f = ax.bar(x + w/2, vr["Flu"],   width=w, color=COL["nwss"], alpha=0.9,
                label="Influenza (2022-23 vs subsequent seasons)", edgecolor="white")
ax.axhline(1, color="grey", lw=0.8, ls="--", alpha=0.6, label="ratio = 1 (no compression)")
ax.set_xticks(x) ; ax.set_xticklabels(sigs)
ax.set_ylabel("variance ratio\n(reference window / subsequent window)")
# Increase ylim headroom so value labels and legend don't overlap
ymax = max([h for h in list(vr["COVID"]) + list(vr["Flu"]) if pd.notna(h)])
ax.set_ylim(0, ymax * 1.30)
# Value labels on bars
for bs in [bars_c, bars_f]:
    for b in bs:
        h = b.get_height()
        if pd.notna(h):
            ax.text(b.get_x() + b.get_width()/2, h + ymax * 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9.5)
ax.legend(loc="upper right", framealpha=0.95)
fig.savefig(OUT / "fig4_variance_ratio.png")
plt.close(fig)

print(f"\nfigures written to {OUT}/")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1e3:.0f} KB)")
