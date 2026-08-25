"""Anomaly detectors for NWSS wastewater time series.

Design notes:
- All detectors consume a pandas Series indexed by daily DatetimeIndex with
  log10-concentration values, return a DataFrame with columns:
    score        — detector output (higher = more anomalous)
    threshold    — per-timestep threshold
    alert        — bool, score > threshold
- Operate on log-concentration because variance scales with mean on linear.
- Baseline windows LAG the current day to avoid a firing wave contaminating
  its own baseline (if the baseline tracks the wave, no alert fires).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def moving_average_control_chart(
    y: pd.Series,
    window: str = "28D",
    k: float = 3.0,
    gap_days: int = 3,
) -> pd.DataFrame:
    """Shewhart-style control chart.

    Score = y_t, threshold = mean + k*std over a baseline window ending
    `gap_days` before t. Only upward alerts (one-sided, right tail).
    """
    y = y.asfreq("D")  # ensure daily grid with NaN for gaps
    # Lag the series by gap_days so rolling stats exclude very-recent points
    lagged = y.shift(gap_days)
    mu = lagged.rolling(window, min_periods=10).mean()
    sd = lagged.rolling(window, min_periods=10).std()
    threshold = mu + k * sd
    score = y
    alert = (score > threshold) & score.notna() & threshold.notna()
    return pd.DataFrame(
        {"score": score, "baseline_mean": mu, "baseline_std": sd,
         "threshold": threshold, "alert": alert.astype(bool)}
    )


def cusum(
    y: pd.Series,
    baseline_window: str = "90D",
    k: float = 0.5,  # slack in units of sigma
    h: float = 5.0,  # alert threshold in units of sigma
    gap_days: int = 3,
) -> pd.DataFrame:
    """One-sided upper CUSUM.

    S_t = max(0, S_{t-1} + (y_t - mu_t) / sigma_t - k)
    where (mu_t, sigma_t) are baseline stats over the window ending gap_days
    before t. Alert when S_t > h.
    """
    y = y.asfreq("D").astype(float)
    lagged = y.shift(gap_days)
    mu = lagged.rolling(baseline_window, min_periods=20).mean()
    sd = lagged.rolling(baseline_window, min_periods=20).std()

    z = (y - mu) / sd
    z = z.fillna(0.0)  # don't accumulate during NaN gaps
    s = np.zeros(len(z), dtype=float)
    for i in range(1, len(z)):
        s[i] = max(0.0, s[i - 1] + z.iloc[i] - k)
    score = pd.Series(s, index=y.index)
    return pd.DataFrame(
        {"score": score, "z": z, "baseline_mean": mu, "baseline_std": sd,
         "threshold": pd.Series(h, index=y.index),
         "alert": (score > h).astype(bool)}
    )
