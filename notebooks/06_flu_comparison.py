"""06_flu_comparison.py — comparative attention-decay analysis: COVID vs flu.

Hypothesis:
  COVID: attention signals (Trends, Wikipedia) showed massive variance during
         BA.1 era then collapsed. Wastewater stayed stable.
  Flu:   no novelty effect — endemic disease — attention signals should
         re-engage each annual season at similar amplitude. Wastewater also
         stable. Net: NO attention decay for flu.

If confirmed, the finding generalizes:
  "Attention-based surveillance fails for emerging-disease novelty cycles,
   not for established endemic diseases."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path("notebooks/fig") ; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- load all signals ------------------------------------------------------
covid_agg = pd.read_csv("data/processed/nwss_aggregated.csv",
                        parse_dates=["date"]).set_index("date")
flu_agg = pd.read_csv("data/processed/nwss_flu_aggregated.csv",
                      parse_dates=["date"]).set_index("date")

trends = pd.read_csv("data/raw/trends_nv_us.csv",
                     parse_dates=["date"]).set_index("date")
trends_us_covid = trends["US::covid symptoms"].astype(float)
trends_us_flu   = trends[["US::fever", "US::cough", "US::sore throat",
                          "US::body aches", "US::congestion"]].mean(axis=1)

wiki = pd.read_csv("data/raw/wikipedia_pageviews.csv",
                   parse_dates=["date"]).set_index("date")
wiki_covid = (wiki["Symptoms_of_COVID-19"] + wiki["Anosmia"] + wiki["Ageusia"])
wiki_flu   = (wiki["Influenza"] + wiki["Common_cold"] +
              wiki["Cough"] + wiki["Fever"] + wiki["Sore_throat"])

ili = pd.read_csv("data/raw/ilinet.csv", parse_dates=["week_start"])
ili_us = ili[ili.region_code == "nat"].set_index("week_start").wili.astype(float)

covid_truth = pd.read_csv("data/raw/covid_ground_truth.csv",
                          parse_dates=["date"]).set_index("date")
hhs_covid_us = covid_truth["hhs_admits_US"]

flu_truth = pd.read_csv("data/raw/hhs_flu_admissions.csv",
                        parse_dates=["date"]).set_index("date")
hhs_flu_us = flu_truth["hhs_flu_US"]

# ---- weekly resample -------------------------------------------------------
def weekly(s, name=None):
    """Smooth daily-cadence signals and resample to weekly. For already-weekly
    or sparser signals, just resample without rolling (no smoothing needed)."""
    s = s.dropna()
    if len(s) == 0:
        return s.resample("W-SUN").mean().rename(name or "empty")
    # Approximate cadence: median gap between observations
    gap_days = s.index.to_series().diff().dt.days.dropna().median()
    if gap_days <= 2:
        # Daily-ish input — smooth
        s = s.rolling("14D", min_periods=3).mean()
    out = s.resample("W-SUN").mean()
    out.name = name or s.name
    return out

W = pd.concat([
    weekly(covid_agg["national"],     "NWSS_covid"),
    weekly(flu_agg["national"],       "NWSS_flu"),
    weekly(trends_us_covid,           "Trends_covid"),
    weekly(trends_us_flu,             "Trends_flu"),
    weekly(wiki_covid,                "Wiki_covid"),
    weekly(wiki_flu,                  "Wiki_flu"),
    weekly(hhs_covid_us,              "HHS_covid"),
    weekly(hhs_flu_us,                "HHS_flu"),
    weekly(ili_us,                    "ILI_us"),
], axis=1)

# Trim to a sensible window for each disease analysis
COVID_END = pd.Timestamp("2024-04-30")  # HHS COVID admits trustworthy through here
FLU_END   = pd.Timestamp("2025-09-30")  # NWSS flu is current

# ---- variance ratio: pandemic-era vs post-pandemic-era ---------------------
print("=== variance ratio: 'reference event' vs subsequent period ===\n")

# COVID: BA.1 vs post
print("COVID: BA.1 (2021-12 to 2022-02) vs post-BA.1 (2022-03 to 2024-04)")
covid_ref_lo, covid_ref_hi = "2021-12-01", "2022-02-28"
covid_post_lo, covid_post_hi = "2022-03-01", "2024-04-30"

# Flu: 2022-23 season (post-pandemic rebound) vs subsequent seasons
print("Flu:   2022-23 season (2022-12 to 2023-02) vs 2023-24 + 2024-25 (2023-03 to 2025-04)")
flu_ref_lo, flu_ref_hi = "2022-12-01", "2023-02-28"
flu_post_lo, flu_post_hi = "2023-03-01", "2025-04-30"
print()

def var_ratio(s, ref_lo, ref_hi, post_lo, post_hi):
    s = s.dropna()
    ref = s.loc[ref_lo:ref_hi].var(ddof=1)
    post = s.loc[post_lo:post_hi].var(ddof=1)
    return ref / post if post > 0 else np.nan

rows = []
for sig, label in [
    ("NWSS_covid",   "NWSS national"),
    ("Trends_covid", "Google Trends"),
    ("Wiki_covid",   "Wikipedia"),
    ("HHS_covid",    "HHS admissions"),
]:
    r = var_ratio(W[sig], covid_ref_lo, covid_ref_hi, covid_post_lo, covid_post_hi)
    rows.append({"disease": "COVID", "signal": label, "var_ratio": r})

for sig, label in [
    ("NWSS_flu",   "NWSS national"),
    ("Trends_flu", "Google Trends"),
    ("Wiki_flu",   "Wikipedia"),
    ("HHS_flu",    "HHS admissions"),
    ("ILI_us",     "ILINet (CDC)"),
]:
    r = var_ratio(W[sig], flu_ref_lo, flu_ref_hi, flu_post_lo, flu_post_hi)
    rows.append({"disease": "Flu", "signal": label, "var_ratio": r})

vr = pd.DataFrame(rows)
print(vr.round(2).to_string(index=False))
vr.to_csv(FIG_DIR / "06_variance_ratio.csv", index=False)

# Pivot for nicer side-by-side comparison
print("\n--- side by side ---")
side = vr.pivot(index="signal", columns="disease", values="var_ratio").round(2)
print(side.to_string())
side.to_csv(FIG_DIR / "06_variance_ratio_pivot.csv")

# ---- visual overlay: COVID + flu side by side ------------------------------
def z_full(s):
    s = s.dropna()
    if s.std() == 0:
        return s * 0
    return (s - s.mean()) / s.std(ddof=0)

zc = pd.DataFrame({
    "HHS admissions (covid)":  z_full(W.loc[:COVID_END, "HHS_covid"]),
    "NWSS national":           z_full(W.loc[:COVID_END, "NWSS_covid"]),
    "Google Trends":           z_full(W.loc[:COVID_END, "Trends_covid"]),
    "Wikipedia":               z_full(W.loc[:COVID_END, "Wiki_covid"]),
})
zf = pd.DataFrame({
    "HHS admissions (flu)":    z_full(W.loc[:FLU_END, "HHS_flu"]),
    "NWSS national":           z_full(W.loc[:FLU_END, "NWSS_flu"]),
    "Google Trends":           z_full(W.loc[:FLU_END, "Trends_flu"]),
    "Wikipedia":               z_full(W.loc[:FLU_END, "Wiki_flu"]),
    "ILINet":                  z_full(W.loc[:FLU_END, "ILI_us"]),
})

fig, (ax_c, ax_f) = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

color_map = {"HHS admissions (covid)": "k", "HHS admissions (flu)": "k",
             "NWSS national": "C0", "Google Trends": "C2",
             "Wikipedia": "C3", "ILINet": "C5"}
lw_map = {"HHS admissions (covid)": 2.0, "HHS admissions (flu)": 2.0,
          "NWSS national": 1.4, "Google Trends": 1.0,
          "Wikipedia": 1.0, "ILINet": 1.4}

for col in zc.columns:
    ax_c.plot(zc.index, zc[col], color=color_map[col], lw=lw_map[col],
              alpha=0.85, label=col)
ax_c.axhline(2, color="red", lw=0.7, ls=":", alpha=0.6, label="z=2")
ax_c.set_title("COVID — attention signals show massive BA.1 spike + decay; "
               "wastewater stable across waves")
ax_c.set_ylabel("z-score")
ax_c.legend(loc="upper right", fontsize=8)

for col in zf.columns:
    ax_f.plot(zf.index, zf[col], color=color_map.get(col, "C7"),
              lw=lw_map.get(col, 1.0), alpha=0.85, label=col)
ax_f.axhline(2, color="red", lw=0.7, ls=":", alpha=0.6, label="z=2")
ax_f.set_title("Flu — attention signals show approximately equal-amplitude "
               "annual peaks; no decay (endemic disease)")
ax_f.set_ylabel("z-score")
ax_f.set_xlabel("week")
ax_f.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "06_covid_vs_flu_overlay.png", dpi=140)
print(f"\nsaved {FIG_DIR / '06_covid_vs_flu_overlay.png'}")

# ---- bar chart: variance ratio comparison ----------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
sigs = ["NWSS national", "Google Trends", "Wikipedia", "HHS admissions"]
covid_vals = [side.loc[s, "COVID"] if s in side.index else np.nan for s in sigs]
flu_vals   = [side.loc[s, "Flu"]   if s in side.index else np.nan for s in sigs]
x = np.arange(len(sigs))
w = 0.35
ax.bar(x - w/2, covid_vals, width=w, label="COVID (BA.1 vs post)", color="C3")
ax.bar(x + w/2, flu_vals,   width=w, label="Flu (22-23 vs subsequent)", color="C0")
ax.axhline(1, color="grey", lw=0.5, ls="--", alpha=0.6)
ax.set_xticks(x) ; ax.set_xticklabels(sigs, rotation=15, ha="right")
ax.set_ylabel("Variance ratio: reference / subsequent")
ax.set_title("Attention decay is COVID-specific, not signal-specific:\n"
             "Flu attention signals do not show variance compression "
             "across seasons")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "06_variance_ratio_bars.png", dpi=140)
print(f"saved {FIG_DIR / '06_variance_ratio_bars.png'}")
