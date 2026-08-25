"""Build population-weighted NWSS aggregates at national and state level.

Method (matches CDC's general approach for their "national wastewater level"):
  1. Load concentration data, filter to flow-population normalization.
  2. Join with metric dataset to get population_served + state per site.
  3. Drop sites with < MIN_SAMPLES samples in our window (noise filter).
  4. log10-transform each site's series; resample to daily with light interp.
  5. For each day, compute population-weighted mean across sites with data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/processed") ; OUT.mkdir(parents=True, exist_ok=True)
MIN_SAMPLES = 100        # drop sparsely-sampled sites
INTERP_LIMIT_DAYS = 14   # interpolate up to 2 weeks of gaps within a site


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load concentration + per-site metadata, filter to flow-population."""
    conc = pd.read_csv("data/raw/nwss_concentration.csv", parse_dates=["date"])
    conc = conc[conc.normalization == "flow-population"].copy()

    # First metric record per site = stable site metadata
    meta_cols = ["key_plot_id", "wwtp_jurisdiction", "population_served"]
    meta = pd.read_csv("data/raw/nwss_metric.csv", usecols=meta_cols)
    meta = meta.dropna(subset=["population_served"])
    # Drop dupes; population_served can drift in NWSS but pick most common per site
    meta = (meta.groupby("key_plot_id", as_index=False)
                .agg({"wwtp_jurisdiction": "first",
                      "population_served": "median"}))
    return conc, meta


def _per_site_daily(conc: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame indexed by date with one column per site, each column
    holding interpolated log10(concentration). Sites failing MIN_SAMPLES drop."""
    counts = conc.groupby("key_plot_id").size()
    keep = counts[counts >= MIN_SAMPLES].index
    sub = conc[conc.key_plot_id.isin(keep)].copy()
    print(f"  sites passing MIN_SAMPLES={MIN_SAMPLES}: {len(keep):,} of {counts.size:,}")

    # Pivot to wide: rows=date, cols=site, vals=concentration. Average dupes.
    wide = (sub.groupby(["date", "key_plot_id"]).pcr_conc_lin.mean()
                 .unstack("key_plot_id"))
    wide = wide.asfreq("D")  # ensure daily index
    # log10 transform; clip below 1 to avoid log of zero
    wide_log = np.log10(wide.clip(lower=1.0))
    # Per-site interpolation, bounded
    wide_log = wide_log.interpolate(method="time", axis=0,
                                     limit=INTERP_LIMIT_DAYS,
                                     limit_area="inside")
    return wide_log


def _pop_weighted_mean(wide: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """For each row, weighted mean across columns using `weights` (per site).
    Sites with NaN that day are dropped from the average for that day, and
    weights are renormalized over the present sites."""
    w = weights.reindex(wide.columns).fillna(0.0).values
    arr = wide.values
    mask = ~np.isnan(arr)
    weighted_sum = np.where(mask, arr, 0.0) @ w
    weight_sum = mask.astype(float) @ w
    out = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
    return pd.Series(out, index=wide.index)


def build_aggregates() -> pd.DataFrame:
    print("loading...")
    conc, meta = _load()
    print(f"  concentration rows: {len(conc):,}  sites: {conc.key_plot_id.nunique():,}")

    print("\npivoting to per-site daily log10...")
    wide_log = _per_site_daily(conc)
    print(f"  pivot shape: {wide_log.shape}")

    # Site-level metadata aligned to the pivot's columns
    pop = meta.set_index("key_plot_id").population_served.reindex(wide_log.columns)
    juris = meta.set_index("key_plot_id").wwtp_jurisdiction.reindex(wide_log.columns)

    print("\nbuilding aggregates...")
    out = pd.DataFrame(index=wide_log.index)

    # National
    out["national"] = _pop_weighted_mean(wide_log, pop)
    print(f"  national: {out['national'].notna().sum():,} non-null days")

    # Per-state aggregates we care about
    for state in ["Nevada", "California", "New York", "Texas"]:
        cols = juris[juris == state].index
        if len(cols) == 0:
            continue
        sub = wide_log[cols]
        sub_pop = pop[cols]
        out[state.lower().replace(" ", "_")] = _pop_weighted_mean(sub, sub_pop)
        n_sites = len(cols)
        n_days = out[state.lower().replace(" ", "_")].notna().sum()
        print(f"  {state}: {n_sites} sites, {n_days:,} non-null days")

    # Site count per day (national) — useful diagnostic
    out["national_n_sites"] = (~wide_log.isna()).sum(axis=1)

    out.index.name = "date"
    return out


if __name__ == "__main__":
    df = build_aggregates()
    out_path = OUT / "nwss_aggregated.csv"
    df.to_csv(out_path)
    print(f"\nsaved {out_path}  shape={df.shape}")
    print(f"date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"\ntail:\n{df.tail(5).to_string()}")
