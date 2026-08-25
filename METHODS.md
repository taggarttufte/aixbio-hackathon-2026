# Anomaly Detection Methods — Solo 48-Hour Tractable

Ranked from simplest (implement in an hour) to most involved. All are
realistic for a 48-hour solo submission. The "right" method is the one
whose failure modes you can discuss clearly in the report.

---

## Tier 1 — Classical, fast to implement, easy to explain

### 1. Moving-average control chart (Shewhart / EWMA)

- **What:** Flag when observation exceeds rolling mean by k standard deviations
- **Library:** `numpy` / `pandas.rolling` — no external dep needed
- **Strength:** Transparent, defensible baseline, matches what most public
  health systems actually deploy
- **Failure mode:** Slow to catch gradual exponential growth; struggles with
  seasonality

### 2. CUSUM (cumulative sum)

- **What:** Accumulate deviations from target; alert when running sum crosses
  threshold. Very fast to respond to small persistent shifts.
- **Library:** `scipy.stats` or hand-roll in ~20 lines
- **Strength:** The classic early-detection statistical method; strong on
  gradual onsets where a control chart lags
- **Failure mode:** Threshold tuning is sensitive; poorly calibrated CUSUMs
  fire constantly

### 3. Moving percentile (CIDARS-style)

- **What:** Alert when current window exceeds Nth percentile of same-week
  in prior years. China's CIDARS uses this with 7-day rolling + 95th
  percentile of prior 5 years.
- **Strength:** Naturally handles seasonality; currently deployed IRL so
  you can frame improvement relative to a real baseline
- **Failure mode:** Needs multi-year historical window, which some
  wastewater time series don't have

---

## Tier 2 — ML, moderate complexity

### 4. Isolation Forest

- **What:** Unsupervised ensemble of random trees that isolate anomalies in
  few splits; observations needing few splits to isolate = anomalies
- **Library:** `sklearn.ensemble.IsolationForest` or `pyod.models.iforest`
- **Strength:** Handles multivariate data well (multiple pathogens or
  multiple sites); minimal hyperparameter tuning
- **Failure mode:** Has no notion of time-ordering; can miss slow trends
- **Hybrid option:** Combine with rolling-window features (rate of change,
  rolling std) to inject temporal structure

### 5. Seasonal decomposition + residual anomaly detection

- **What:** Use STL / `statsmodels.tsa.seasonal_decompose` to split signal
  into trend + seasonal + residual; flag residual outliers
- **Library:** `statsmodels`
- **Strength:** Handles seasonality cleanly; very interpretable
- **Failure mode:** STL needs decent history; residuals aren't always
  Gaussian

### 6. Change-point detection

- **What:** Detects when a time series' distributional parameters change.
  Different problem from anomaly detection but related; useful for
  "regime change" alerts (new variant emergence).
- **Library:** `ruptures` (Python) — well-documented
- **Strength:** Good match for "did something fundamentally change in the
  signal?" questions
- **Failure mode:** Less about single-point anomalies; more about regimes

---

## Tier 3 — Deep learning, stretch but viable

### 7. LSTM autoencoder anomaly detector

- **What:** Train LSTM to reconstruct normal sequences; reconstruction error
  → anomaly score
- **Library:** PyTorch (Tagg has experience)
- **Strength:** Handles complex temporal patterns; publishable angle
- **Failure mode:** Overkill for univariate weekly data; data-hungry;
  hard to interpret. Consider skipping unless it adds something classical
  methods can't.
- **Realistic scope:** Only attempt if classical baselines are done
  by Saturday morning

### 8. Foundation-model as zero-shot classifier (Claude API)

- **What:** Pass sequence data or surveillance reports to Claude, ask it
  to flag anomalies with reasoning
- **Library:** Anthropic SDK
- **Strength:** Novel angle, lets you compare "LLM reasoning over
  surveillance data" vs. classical statistical detection
- **Failure mode:** Harder to evaluate rigorously; easy to over-claim;
  compute cost adds up
- **Framing:** "LLM-as-analyst vs. statistical-detector" comparison is
  a narrative judges haven't seen a hundred times

---

## Evaluation metrics (pick 2-3)

A submission that rigorously evaluates 2 methods beats one that casually
applies 5. Use:

1. **Lead time** (days/weeks between alert and documented wave onset —
   higher is better)
2. **False alarm rate** (alerts per year on non-outbreak periods —
   lower is better)
3. **Detection sensitivity at fixed FPR** (what fraction of waves caught
   if you cap false alarms at 1/year)
4. **AUROC** (if you can frame as binary classification — less natural
   for time series but familiar)

Honest limitations matter more than fancy numbers. Say explicitly: "ground
truth wave onsets are hand-labeled, not systematic"; "K weeks of lead time
is qualitative given the noise level at most individual sites."

---

## Reading list (priority-ordered)

1. **"Practical Approach for Evaluating Machine Learning Anomaly Detection
   Algorithms for Epidemic Early Warning Systems"** (2025, PubMed 40380686)
   — evaluation framework
2. **Zehnder et al. 2023** (GeoHealth, SARS-CoV-2 wastewater ML) —
   applied methods reference
3. **"AI-Assisted Real-Time Monitoring of Infectious Diseases in Urban
   Areas"** (MDPI 2025) — ARIMA / isolation forest / LSTM comparison
4. **Deep Learning for Time Series Anomaly Detection: A Survey**
   (arxiv 2211.05244) — for Tier 3 references
5. **METAGENE-1 paper** (arxiv 2501.02045) — cite as "foundation-model
   approach" you're contrasting against; probably not feasible to run
