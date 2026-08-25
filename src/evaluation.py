"""Evaluation protocol for wastewater anomaly detectors.

Ground-truth events are named COVID waves with approximate peak dates from CDC
reporting. For each event we define a *detection window* ending at the peak —
an alert anywhere in that window counts as a detection, and we record the
lead time (peak_date - first_alert_date; positive = caught ahead of peak).
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


# Peaks from CDC narrative reporting (approximate; hackathon-grade).
WAVE_PEAKS = {
    "Delta":    "2021-09-01",
    "BA.1":     "2022-01-15",
    "BA.2":     "2022-04-20",
    "BA.5":     "2022-07-20",
    "XBB/BQ":   "2023-01-05",
    "EG.5":     "2023-09-05",
    "JN.1":     "2024-01-05",
    "KP.3":     "2024-08-10",
}


@dataclass
class DetectionResult:
    wave: str
    peak: pd.Timestamp
    first_alert: pd.Timestamp | None
    lead_days: float | None  # peak - first_alert; None if no alert in window


def _alerts_to_episodes(alerts: pd.Series, gap_days: int = 7) -> pd.DatetimeIndex:
    """Collapse runs of consecutive alert days into a single episode (the
    onset date). Two alerts separated by <= `gap_days` of no-alert days are
    treated as the same episode.

    This matters for cumulative detectors (CUSUM): once the cumulative score
    crosses threshold it can stay above threshold for the rest of a wave,
    which would otherwise count as hundreds of "alerts" for one event.
    """
    alerts = alerts.astype(bool)
    alert_dates = alerts.index[alerts]
    if len(alert_dates) == 0:
        return alert_dates
    gap = pd.Timedelta(days=gap_days)
    deltas = alert_dates.to_series().diff()
    new_episode = (deltas.isna()) | (deltas > gap)
    return alert_dates[new_episode.values]


def evaluate(
    alerts: pd.Series,
    peaks: dict[str, str] = WAVE_PEAKS,
    window_days_before: int = 60,
    window_days_after: int = 0,
    episode_gap_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Return (per-wave detection results, summary dict).

    An alert counts for wave W if it lands in [peak_W - before, peak_W + after].
    Lead time = (peak_W - first_alert).days.
    False alarms = alert *episodes* outside ANY wave window (consecutive
    alert runs collapsed via `_alerts_to_episodes`).
    """
    alerts = alerts.astype(bool)
    alert_dates = _alerts_to_episodes(alerts, gap_days=episode_gap_days)

    windows = []
    rows: list[DetectionResult] = []
    for name, date in peaks.items():
        p = pd.Timestamp(date)
        lo = p - pd.Timedelta(days=window_days_before)
        hi = p + pd.Timedelta(days=window_days_after)
        if p < alerts.index.min() or p > alerts.index.max():
            continue
        windows.append((lo, hi))
        mask = (alert_dates >= lo) & (alert_dates <= hi)
        firing = alert_dates[mask]
        if len(firing):
            first = firing[0]
            rows.append(DetectionResult(
                wave=name, peak=p, first_alert=first,
                lead_days=(p - first).days
            ))
        else:
            rows.append(DetectionResult(wave=name, peak=p,
                                        first_alert=None, lead_days=None))

    per_wave = pd.DataFrame([r.__dict__ for r in rows])

    # False-alarm count: alerts not in ANY wave window
    in_any_window = pd.Series(False, index=alert_dates)
    for lo, hi in windows:
        in_any_window |= ((alert_dates >= lo) & (alert_dates <= hi))
    false_alarms = int((~in_any_window).sum())

    n_waves = per_wave.shape[0]
    n_detected = per_wave.first_alert.notna().sum()
    mean_lead = per_wave.lead_days.dropna().mean() if n_detected else float("nan")

    summary = {
        "n_waves": n_waves,
        "n_detected": int(n_detected),
        "detection_rate": n_detected / n_waves if n_waves else float("nan"),
        "mean_lead_days": mean_lead,
        "n_alert_episodes": len(alert_dates),
        "n_false_alarms": false_alarms,
    }
    return per_wave, summary
