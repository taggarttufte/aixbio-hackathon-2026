"""Quick probe of both NWSS datasets to understand structure before analysis."""
import pandas as pd

print("=== CONCENTRATION ===")
conc = pd.read_csv("data/raw/nwss_concentration.csv", parse_dates=["date"])
print(f"rows={len(conc):,}  cols={list(conc.columns)}")
print(f"date range: {conc.date.min()} -> {conc.date.max()}")
print(f"unique key_plot_ids: {conc.key_plot_id.nunique():,}")
print(f"normalization counts:\n{conc.normalization.value_counts()}")
print()

print("=== METRIC ===")
m = pd.read_csv("data/raw/nwss_metric.csv", parse_dates=["date_start", "date_end", "first_sample_date"])
print(f"rows={len(m):,}")
print(f"cols={list(m.columns)}")
print(f"date_end range: {m.date_end.min()} -> {m.date_end.max()}")
print(f"states (wwtp_jurisdiction): {m.wwtp_jurisdiction.nunique()}")
print(f"unique key_plot_ids: {m.key_plot_id.nunique():,}")
print()

print("=== TOP STATES BY # OF UTILITIES ===")
sites_per_state = m.groupby("wwtp_jurisdiction").key_plot_id.nunique().sort_values(ascending=False)
print(sites_per_state.head(15))
print()

print("=== TOP SITES BY SAMPLE DENSITY (in concentration table) ===")
samples_per_site = conc.key_plot_id.value_counts().head(10)
print(samples_per_site)
print()

print("=== DATE COVERAGE OF A TOP SITE ===")
top = samples_per_site.index[0]
sub = conc[conc.key_plot_id == top].sort_values("date")
print(f"site: {top}")
print(f"  n_samples={len(sub)}  first={sub.date.min()}  last={sub.date.max()}")
print(f"  sample of values:\n{sub.head(3)}\n{sub.tail(3)}")
