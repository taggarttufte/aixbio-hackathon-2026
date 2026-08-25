# Pre-Event Prep Plan — AIxBio Hackathon Track 2

**Event window:** April 24-26, 2026 (Friday evening → Sunday).
**Today:** 2026-04-22. You have ~2 days.

The goal of prep is to **remove the first 12 hours of pain** from the 48-hour
sprint: dataset already downloaded, baseline already runnable, domain
vocabulary already familiar. Every hour invested in prep now is worth ~3
hours during the event.

---

## Priority ladder (if you only have a few hours)

### Must-do before Friday (~4 hours total)

1. **Join the Apart Discord.** Introduce yourself in the Track 2 channel.
   State you're solo/looking for a teammate + your rough angle (stats / ML
   on time-series surveillance data, not bio). Match happens before
   kickoff; most pairings happen within the first hour of Discord activity.
2. **Download the NWSS wastewater dataset** via `data.cdc.gov` (search
   "wastewater"). Store raw CSVs in `data/raw/`. This is your default
   primary dataset — you'll know on Friday whether you use it or swap.
3. **Run one baseline end-to-end on a toy slice.** Pick one city or one
   state, plot the SARS-CoV-2 signal, implement a moving-average control
   chart or CUSUM detector, see what alerts fire. Make sure the numbers
   match published COVID waves qualitatively.
4. **Read the "Practical Approach for Evaluating ML Anomaly Detection"
   paper** (PubMed 40380686, 2025). Focus on Section: "evaluation
   framework." This gives you vocabulary for framing your submission.

### Nice-to-have if time (~2 more hours)

5. Skim **Zehnder et al. 2023** (GeoHealth) — ML for infection hotspots from
   wastewater SARS-CoV-2. Cleanest methods reference for solo work.
6. Skim the **2026 review** of AI/ML in wastewater surveillance (ScienceDirect
   S0048969726000215) to map the landscape; read abstract + figures only.
7. Check **METAGENE-1** (arxiv 2501.02045) — probably too large to run
   locally (7B params), but useful to cite as "foundation-model approach"
   to contrast against your simpler baselines.
8. Browse the **EA Forum post** on the hackathon
   (forum.effectivealtruism.org/posts/J5Rd25x4XeEcofWfq) for tone + what
   organizers value.

### If you still have time (~2 more hours)

9. Skim the **CIDARS moving-percentile method** (China's outbreak alert
   system) — simplest real-world deployed baseline; lets you frame your
   method as "improvement over current practice."
10. Sketch your report outline now (intro / data / method / results /
    limitations / implications — 1 paragraph each) so Friday is "fill in"
    rather than "write from blank."

---

## Environment setup (do once, now)

```bash
cd aixbio-hackathon-2026
python -m venv venv
source venv/Scripts/activate  # bash on Windows
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install statsmodels ruptures pyod  # anomaly detection libraries
pip install jupyter
```

Libraries worth knowing:
- `scipy.stats` — basic statistical tests, control-chart math
- `statsmodels` — ARIMA, seasonal decomposition, CUSUM
- `ruptures` — change-point detection (complementary to anomaly detection)
- `pyod` — Python Outlier Detection, has isolation forest + 40 others
  pre-implemented
- `sklearn.ensemble.IsolationForest` — if you don't want pyod

## What NOT to waste prep time on

- **Reading the whole AI/ML wastewater review cover-to-cover.** Skim
  figures + section headers only.
- **Building a fancy data pipeline.** One pandas DataFrame with a clean
  date index is all you need to start.
- **Setting up experiment tracking (W&B, MLflow).** 48 hours, solo.
  Matplotlib + CSV results is fine.
- **Trying to run METAGENE-1 locally.** It won't fit cleanly. If you want
  a foundation-model angle, use Claude API for sequence classification as
  a light zero-shot baseline instead.
- **Pre-picking your project idea now.** You don't yet know (a) Friday's
  track briefing, (b) what teammates you might pair with, (c) what data
  is easiest to work with. Let Friday collapse the decision.

## What to bring to Friday kickoff

- Dataset loaded, plotted, and understood at a toy-slice level
- 2-3 project ideas scoped to 48 hours each (see PROJECT_IDEAS.md)
- Vocabulary: control chart, CUSUM, moving average, isolation forest,
  LSTM anomaly detection, lead time, false alarm rate, sensitivity,
  specificity. If you can explain each in one sentence, you're set.
- Notebook in `notebooks/00_explore_nwss.ipynb` that loads and plots the
  data, ready to fork for whichever project you pick Friday night
