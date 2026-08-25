"""01_baseline_detectors.py — run moving-average control chart + CUSUM on one
NWSS site and evaluate against known COVID waves.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors import moving_average_control_chart, cusum
from src.evaluation import evaluate, WAVE_PEAKS

FIG_DIR = Path("notebooks/fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITE = "NWSS_nv_554_Treatment plant_90_raw wastewater"

# ---- load & prep -----------------------------------------------------------
print("loading...")
conc = pd.read_csv("data/raw/nwss_concentration.csv", parse_dates=["date"])
conc = conc[(conc.key_plot_id == SITE) &
            (conc.normalization == "flow-population")]
s = conc.groupby("date", as_index=False).pcr_conc_lin.mean().sort_values("date")

# Drop the trailing-data cliff (Aug-Sep 2025 values drop toward 0 at dataset cut)
s = s[s.date <= "2025-07-01"]

# Daily grid, log10 transform, light interpolation
daily = (s.set_index("date").asfreq("D").pcr_conc_lin
          .clip(lower=1.0)           # floor to avoid log(0)
          .interpolate(method="time", limit=7))
log_y = np.log10(daily)
log_y.name = "log10_conc"

# ---- run detectors ---------------------------------------------------------
print("\n=== moving-average control chart ===")
summaries = {}
per_wave_tables = {}
for k in [2.0, 2.5, 3.0]:
    out = moving_average_control_chart(log_y, window="28D", k=k, gap_days=3)
    per_wave, summ = evaluate(out.alert)
    label = f"MA28 k={k}"
    summaries[label] = summ
    per_wave_tables[label] = per_wave
    print(f"{label}: {summ}")

print("\n=== CUSUM ===")
for h in [3.0, 5.0, 8.0]:
    out_cu = cusum(log_y, baseline_window="90D", k=0.5, h=h, gap_days=3)
    per_wave, summ = evaluate(out_cu.alert)
    label = f"CUSUM h={h}"
    summaries[label] = summ
    per_wave_tables[label] = per_wave
    print(f"{label}: {summ}")

# ---- summary table ---------------------------------------------------------
print("\n=== summary ===")
summary_df = pd.DataFrame(summaries).T
print(summary_df.to_string())

# ---- per-wave view for the best-looking MA detector ------------------------
print("\n=== per-wave detection (MA28 k=2.5) ===")
print(per_wave_tables["MA28 k=2.5"].to_string(index=False))

# ---- alert plot for a chosen detector -------------------------------------
out = moving_average_control_chart(log_y, window="28D", k=2.5, gap_days=3)

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(log_y.index, log_y.values, lw=0.7, alpha=0.6, label="log10 conc")
ax.plot(out.index, out.threshold, color="red", lw=0.9, alpha=0.7,
        label="threshold (μ+2.5σ)")
ax.plot(out.index, out.baseline_mean, color="orange", lw=1.0, alpha=0.7,
        label="28d baseline mean")
alert_t = out.index[out.alert]
alert_y = log_y.reindex(alert_t)
ax.scatter(alert_t, alert_y, color="red", s=6, zorder=5, label="alert")

for name, d in WAVE_PEAKS.items():
    d = pd.Timestamp(d)
    if log_y.index.min() <= d <= log_y.index.max():
        ax.axvline(d, color="grey", lw=0.6, alpha=0.4, ls="--")
        ax.text(d, 0.98, name, transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=7, color="grey")

ax.set_ylabel("log10 PCR conc")
ax.set_title(f"{SITE.split('_')[2].upper()} — MA28 k=2.5 control chart alerts")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_ma_control_chart.png", dpi=140)
print(f"\nsaved {FIG_DIR / '01_ma_control_chart.png'}")

# Save the summary table for later
summary_df.to_csv("notebooks/fig/01_summary.csv")
print(f"saved notebooks/fig/01_summary.csv")
