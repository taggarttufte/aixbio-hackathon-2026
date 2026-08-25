# Project Ideas — Track 2, 48-hour Solo Submissions

Four project shapes, each scoped to a solo 48-hour sprint. Pick ONE at
kickoff. The ranking reflects fit to Tagg's strengths (math, stats,
Bayesian reasoning, ML methodology) over bio domain.

---

## A. **Wastewater detector benchmark** — SAFE DEFAULT

**One-sentence pitch:** "I implemented 4 anomaly-detection methods on CDC
NWSS data, evaluated each against documented COVID / mpox waves, and
report lead-time vs. false-alarm tradeoffs."

**Methods:** Moving-average control chart + CUSUM + Isolation Forest +
one seasonal approach (STL residual). Evaluate on 3-4 documented wave
events.

**Why it fits:**
- Entirely within math/stats comfort zone
- Clear, defensible, almost guaranteed to produce results
- Maps cleanly to Bayesian epistemology framing (what does "alert" mean
  probabilistically?)
- Plays well with "honest limitations" framing

**Weakness:**
- Least novel. Many past papers have benchmarked detectors.
- To differentiate: (a) include one pathogen that's been less studied
  (mpox or H5N1 if data exists), (b) include a rigorous evaluation
  protocol other papers skip, or (c) add the Bayesian calibration layer
  from Project C below.

**Scope guard:** If you're 12 hours in and still debugging data loading,
drop a method and focus.

---

## B. **Multi-signal fusion for early warning**

**One-sentence pitch:** "Combining wastewater and search-trend signals
reduces detection latency by X days vs. either signal alone, demonstrated
on 3 wave events."

**Methods:** Pull NWSS + Google Trends (via `pytrends`); build a simple
fusion detector (weighted ensemble, or a logistic-regression meta-detector
over method outputs); evaluate vs. single-signal baselines.

**Why it's interesting:**
- Multi-signal angle is newer and underexplored
- Has clear narrative arc: "signal X has Y limitations, signal Z fills
  the gap"
- Pairs well with Tagg's Bayesian framing (how do you combine
  evidence from independent sources?)

**Weakness:**
- More wrangling (Google Trends data is flaky and rate-limited)
- Claims need to be careful — it's easy to over-state fusion benefits
  when confounders drive correlation

**Scope guard:** If `pytrends` is broken by Saturday morning, fall back to
Project A without shame.

---

## C. **Bayesian calibration for outbreak alerts** — DISTINCTIVE PICK

**One-sentence pitch:** "Anomaly alerts without calibrated probabilities
are actionably meaningless. I applied Bayesian likelihood-ratio analysis
to NWSS detectors, producing probability-of-outbreak given alert values
that match observed wave rates."

**Methods:** Take one detector (e.g., CUSUM); instead of just firing
alerts, compute P(outbreak | alert level) using historical alert data +
Bayes' rule. Calibrate on held-out waves. Report reliability diagrams.

**Why it's distinctive:**
- Plays directly to Tagg's philosophy-of-science coursework (PHL 345,
  Bandyopadhyay — Bayesian confirmation vs. likelihood-ratio evidence)
- No past submission will have this angle — it's an epistemology
  contribution, not an ML contribution
- Ties naturally to a scalable-oversight research framing (what
  evidence justifies confidence?)
- 2-3 pages of this can be a better paper than 10 pages of B

**Weakness:**
- Harder to make "look it works" compelling without a calibration figure
  that actually calibrates cleanly
- Requires a second dataset for out-of-sample validation

**Scope guard:** Write the "ideal figure" on paper before coding. If the
data can't produce it, fall back to A.

---

## D. **Evaluation framework critique**

**One-sentence pitch:** "Current early-warning evaluation (lead time vs.
false alarm rate) creates perverse incentives. I propose a revised
framework with 3 concrete criteria and apply it to 2 published methods,
finding conclusions invert in 2 of 3 cases."

**Methods:** Read 3-5 early-warning papers; extract their evaluation
protocols; identify a specific failure mode (e.g., "high lead time with
implausible threshold tuning"); propose + apply an alternative protocol.

**Why it's interesting:**
- Meta / philosophical; fits Tagg's epistemology training
- No coding required for the bulk of the work — reading + argument
- The kind of submission that becomes a real working paper if Apart
  Lab Fellowship picks it up

**Weakness:**
- Hardest to produce cleanly in 48 hours
- Judges may prefer "built a thing" over "critiqued a thing"
- Relies on deep lit engagement Tagg doesn't have prep time for

**Scope guard:** Only pick this if you've identified the specific
evaluation flaw by Friday midnight. Otherwise it's Project A.

---

## Decision tree

```
Friday kickoff:
├── Data loads cleanly by hour 4?                       → yes: pick A or C
│                                                        → no: diagnose; consider pivoting dataset
├── Have a Bayesian calibration framing in mind?        → yes: C
│                                                        → no: A
├── Found a teammate strong in bio or data engineering? → consider B
└── Found a specific eval-framework flaw to argue?      → D
```

**Default recommendation:** A with a Bayesian-calibration angle from C
layered in. Best of both: safe execution plus distinctive framing. Total
scope: 4 detectors on NWSS, full evaluation, one-figure calibration
analysis at the end.
