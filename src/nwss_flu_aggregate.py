"""Build population-weighted NWSS Flu A aggregates at national and state level.

Schema (different from COVID dataset):
  sample_collect_date     — sample date
  state_territory         — state
  population_served       — for weighting
  pcr_target_flowpop_lin  — flow-population normalized concentration
  site                    — site identifier
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/processed") ; OUT.mkdir(parents=True, exist_ok=True)
MIN_SAMPLES = 50         # flu data is shorter (since 2022); lower bar than COVID
INTERP_LIMIT_DAYS = 14


def main():
    print("loading NWSS Flu A...")
    df = pd.read_csv("data/raw/nwss_flu_a.csv",
                     usecols=["site", "state_territory", "population_served",
                              "sample_collect_date", "pcr_target_flowpop_lin"],
                     parse_dates=["sample_collect_date"])
    print(f"  raw rows: {len(df):,}")

    # Filter rows with non-null flow-pop concentration
    df = df.dropna(subset=["pcr_target_flowpop_lin", "population_served",
                            "sample_collect_date"])
    df = df[df.pcr_target_flowpop_lin > 0]
    print(f"  with flow-pop conc > 0: {len(df):,}")
    print(f"  date range: {df.sample_collect_date.min()} -> "
          f"{df.sample_collect_date.max()}")
    print(f"  unique sites: {df.site.nunique():,}")

    # Filter sites with enough samples
    counts = df.groupby("site").size()
    keep_sites = counts[counts >= MIN_SAMPLES].index
    df = df[df.site.isin(keep_sites)].copy()
    print(f"  sites passing MIN_SAMPLES={MIN_SAMPLES}: {len(keep_sites):,}")

    # Per-site population (median in case of small drift) and state
    site_meta = (df.groupby("site")
                   .agg(state=("state_territory", "first"),
                        population=("population_served", "median"))
                   .reset_index())

    # Pivot to wide: rows=date, cols=site, values=conc (mean over dupes)
    wide = (df.groupby(["sample_collect_date", "site"]).pcr_target_flowpop_lin.mean()
              .unstack("site"))
    wide.index.name = "date"
    wide = wide.asfreq("D")
    wide_log = np.log10(wide.clip(lower=1.0))
    wide_log = wide_log.interpolate(method="time", axis=0,
                                     limit=INTERP_LIMIT_DAYS,
                                     limit_area="inside")
    print(f"  daily pivot shape: {wide_log.shape}")

    pop = site_meta.set_index("site").population.reindex(wide_log.columns)
    state = site_meta.set_index("site").state.reindex(wide_log.columns)

    def pwm(wide, weights):
        w = weights.reindex(wide.columns).fillna(0.0).values
        arr = wide.values
        mask = ~np.isnan(arr)
        ws = np.where(mask, arr, 0.0) @ w
        wm = mask.astype(float) @ w
        return pd.Series(np.where(wm > 0, ws / wm, np.nan), index=wide.index)

    out = pd.DataFrame(index=wide_log.index)
    out["national"] = pwm(wide_log, pop)
    print(f"  national: {out['national'].notna().sum():,} non-null days")

    for st_code, st_name in [("nv", "nevada"), ("ca", "california"),
                              ("ny", "new_york"), ("tx", "texas")]:
        cols = state[state == st_code].index
        if len(cols) == 0:
            print(f"  {st_name}: 0 sites (skipped)")
            continue
        out[st_name] = pwm(wide_log[cols], pop[cols])
        print(f"  {st_name}: {len(cols)} sites, "
              f"{out[st_name].notna().sum():,} non-null days")

    out["national_n_sites"] = (~wide_log.isna()).sum(axis=1)

    out_path = OUT / "nwss_flu_aggregated.csv"
    out.to_csv(out_path)
    print(f"\nsaved {out_path}  shape={out.shape}")
    print(f"date range: {out.index.min().date()} -> {out.index.max().date()}")
    print(f"\ntail:\n{out.tail(5).to_string()}")


if __name__ == "__main__":
    main()
