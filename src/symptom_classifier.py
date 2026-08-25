"""Two-stage classifier for "respiratory illness self-diagnosis" prompts.

Stage 1: keyword pre-filter on the first user message of a conversation.
  - Cheap, runs on CPU at >100k rows/sec.
  - Drops ~95% of conversations as obvious-no.

Stage 2: zero-shot NLI classifier on candidates that passed Stage 1.
  - MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (~360 MB).
  - Fits in <4 GB VRAM on a 3080 Ti.
  - Candidate labels distinguish:
      A. first-person illness self-diagnosis  ← what we want
      B. asking on behalf of someone else (still useful as weaker signal)
      C. general medical question (not illness-related)
      D. unrelated to health
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import re
import pandas as pd


# ---- Stage 1: keyword pre-filter -------------------------------------------

# Symptom and illness vocabulary. Mix of clinical and colloquial.
# Word boundaries on each token so "headache" doesn't match "ahead".
SYMPTOM_TOKENS = [
    # Respiratory / classic COVID-flu-RSV symptoms
    r"cough", r"cold", r"flu", r"fever", r"chills", r"sore throat", r"runny nose",
    r"stuffy nose", r"congestion", r"congested", r"phlegm", r"mucus", r"sneez",
    r"shortness of breath", r"trouble breathing", r"wheez", r"chest tight",
    # Loss-of-X
    r"loss of taste", r"loss of smell", r"can'?t taste", r"can'?t smell",
    # Systemic
    r"body aches", r"body pain", r"muscle aches", r"fatigue", r"exhausted",
    r"headache", r"head ache", r"head pain", r"migraine",
    # GI (covid does this in some)
    r"nausea", r"vomit", r"diarrhea", r"stomach bug",
    # Disease names
    r"covid", r"corona\s?virus", r"sars[- ]cov[- ]?2", r"omicron",
    r"influenza", r"\brsv\b", r"strep",
    # Self-state expressions
    r"i feel sick", r"i'?m sick", r"feel(?:ing)? unwell", r"feel(?:ing)? terrible",
    r"under the weather", r"coming down with", r"feel(?:ing)? feverish",
    r"tested positive", r"positive for", r"got covid", r"caught covid",
    # Care-seeking expressions
    r"should i (?:see|go to) (?:a |the )?doctor", r"go to (?:the )?er",
    r"er for", r"urgent care",
]
SYMPTOM_RE = re.compile(r"\b(?:" + r"|".join(SYMPTOM_TOKENS) + r")\b", re.IGNORECASE)


def keyword_match(text: str) -> tuple[bool, str | None]:
    """Return (matched, first_matched_token_or_phrase)."""
    if not isinstance(text, str) or not text:
        return False, None
    m = SYMPTOM_RE.search(text)
    if m:
        return True, m.group(0).lower()
    return False, None


def filter_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add `kw_match` and `kw_token` columns; return rows where kw_match=True."""
    matches = df[text_col].fillna("").map(keyword_match)
    df = df.assign(
        kw_match=[m[0] for m in matches],
        kw_token=[m[1] for m in matches],
    )
    return df[df.kw_match].copy()


# ---- Stage 2: zero-shot NLI classifier -------------------------------------

# Single binary hypothesis. NLI works better with declarative, conversational
# phrasings than clinical jargon. The target label is the one we aggregate.
# We deliberately collapse "self" and "close family member" into one bucket
# because for surveillance both are signals of household-level illness, and
# NLI confuses them anyway.
CANDIDATE_LABELS = [
    "the user or someone they live with is currently sick with respiratory or flu-like symptoms",
    "the user is asking a general informational question about a disease, not about a current illness",
    "the message is a fictional, hypothetical, or creative-writing scenario, not a real illness",
]
TARGET_LABEL = CANDIDATE_LABELS[0]
FICTION_LABEL = CANDIDATE_LABELS[2]


@dataclass
class ClassifierResult:
    label: str
    confidence: float
    label_scores: dict[str, float]


class ZeroShotSymptomClassifier:
    """Wraps HF `pipeline("zero-shot-classification")` with our label set."""

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: int | str = 0,  # 0 = first GPU; -1 for CPU
        batch_size: int = 16,
    ):
        from transformers import pipeline
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
        self.batch_size = batch_size

    def classify(self, texts: Iterable[str]) -> list[ClassifierResult]:
        """Run the classifier on a list of texts. Multi-label scoring (each
        label gets independent 0-1 probability). Truncates very long prompts."""
        texts = [(t or "")[:1500] for t in texts]
        out = self.pipe(
            texts,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=True,
            batch_size=self.batch_size,
        )
        if isinstance(out, dict):  # single-input case
            out = [out]
        results = []
        for o in out:
            scores = dict(zip(o["labels"], o["scores"]))
            best = o["labels"][0]
            results.append(ClassifierResult(
                label=best, confidence=o["scores"][0], label_scores=scores,
            ))
        return results


def aggregate_weekly(
    classified: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_score_col: str = "p_target",
    fiction_score_col: str = "p_fiction",
    target_threshold: float = 0.7,
    fiction_max: float = 0.5,
    denominator: pd.DataFrame | None = None,
    denominator_timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Per-week count of "household illness" prompts.

    A row counts as a positive if `p_target >= target_threshold` AND
    `p_fiction <= fiction_max`. The fiction filter rejects creative-writing
    prompts that score high on the target label.

    `classified` is the post-classifier DataFrame (only keyword-passes).
    `denominator` (optional) is the FULL conversation set (pre-filter) used
    to compute prompts-per-week so we can normalize by total volume; if not
    given, we use `classified` as both numerator-source and denominator
    (giving frac of keyword-matched prompts that are real illness).
    """
    df = classified.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True).dt.tz_localize(None)
    df["week"] = df[timestamp_col].dt.to_period("W").dt.start_time

    is_target = (df[target_score_col] >= target_threshold) & \
                (df[fiction_score_col] <= fiction_max)
    df["is_target"] = is_target.astype(int)

    num = df.groupby("week").agg(n_target=("is_target", "sum"),
                                 n_kw_pass=("is_target", "size")).reset_index()

    if denominator is not None:
        d = denominator.copy()
        d[denominator_timestamp_col] = (
            pd.to_datetime(d[denominator_timestamp_col], utc=True)
              .dt.tz_localize(None)
        )
        d["week"] = d[denominator_timestamp_col].dt.to_period("W").dt.start_time
        denom = d.groupby("week").size().reset_index(name="n_total")
        num = num.merge(denom, on="week", how="outer").fillna(0)
    else:
        num["n_total"] = num["n_kw_pass"]

    num["frac_target"] = num["n_target"] / num["n_total"].replace(0, pd.NA)
    return num.sort_values("week").reset_index(drop=True)
