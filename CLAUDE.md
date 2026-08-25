# CLAUDE.md — reviewer & agent router

A short orientation for anyone (human or AI tool) pointed at this repo. This was a
solo, time-boxed **Apart Research AIxBio Hackathon (Track 2, April 2026)** submission —
a research paper with reproducible supporting code, not a product build. Read the paper
and the README; this file just routes you there.

## Start here

- **[`paper/paper.pdf`](paper/paper.pdf)** (≈8–10 pp) — the deliverable. The full argument,
  tables, and figures. `paper/paper.md` is the same content in Markdown if you want to grep it.
- **[`README.md`](README.md)** — TL;DR with the headline result, the results table, exact
  data-download commands, and the methods summary. Everything load-bearing is ≤2 hops from here.
- **[`DATASETS.md`](DATASETS.md)** — the full data-source catalog; every dataset is public
  (two are HuggingFace-gated), with download instructions.

## The result in one line

A flu-controlled variance-ratio (and Bayesian likelihood-ratio) analysis showing that
attention-based surveillance signals (Google Trends, Wikipedia) lose calibration 5–23×
after the first COVID-19 wave while wastewater (NWSS) stays stable — so attention decay is
an **emerging-disease novelty-cycle** effect, not a property of the signal type. Includes an
honest negative result: LLM-conversation surveillance (WildChat-4.8M) is too sparse to be viable.

## Signal vs. low-signal, for a limited context budget

- **High signal:** the paper (esp. §5.3 attention-decay, §5.4 loud-vs-silent detector failure,
  §5.5 the LLM null), and `src/` (aggregators, detectors, evaluation protocol, the zero-shot
  classifier).
- **Reproduction:** all public data; `README.md` § Reproduction has the ordered commands. No CI.
- **Not here by design (`.gitignore`):** `data/raw` and `data/processed` (downloadable — see
  DATASETS.md), so referenced data paths are intentionally absent, not missing.
- **Pre-event scoping notes** (`PREP_PLAN.md`, `METHODS.md`, `PROJECT_IDEAS.md`) are historical
  planning docs, not results — skip them unless you care about how the project was scoped.

## Honest caveat to carry into any summary

A 48-hour hackathon submission: single author, US-centric signals, short reference windows
(13 weeks/era → wide implied CIs, stated in §5.4 caveats). Externally reviewed. The paper's
Limitations (§7) and dual-use note (§ Biosecurity relevance) state the bounds plainly; a faithful
summary should too.
