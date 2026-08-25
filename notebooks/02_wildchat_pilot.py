"""02_wildchat_pilot.py — pilot run on a single WildChat-4.8M shard.

Goal: confirm that respiratory-illness self-diagnosis prompts are a non-trivial
fraction of WildChat (>=0.1% gives us a usable signal). Process one shard
(~37k rows), apply keyword filter + DeBERTa zero-shot, and report:
  - shard total rows
  - rows after US+English filter
  - rows after keyword pre-filter
  - rows classified as illness (P_target>=0.7, P_fiction<=0.5)
  - fraction at each stage
  - rough timestamp distribution
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pyarrow.parquet as pq

from src.symptom_classifier import (
    keyword_match, ZeroShotSymptomClassifier, TARGET_LABEL, FICTION_LABEL
)

SHARD = "data/raw/wildchat-4.8m/data/train-00000-of-00086.parquet"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print(f"loading {SHARD}...")
# Only the columns we need to keep memory reasonable
cols = ["conversation_hash", "timestamp", "conversation",
        "language", "country", "state"]
df = pq.read_table(SHARD, columns=cols).to_pandas()
print(f"  shard rows: {len(df):,}  ({time.time()-t0:.1f}s)")

# US + English filter
mask = (df.country == "United States") & (df.language == "English")
df = df[mask].reset_index(drop=True)
print(f"  after US+English: {len(df):,}")

# Extract first user message
def first_user_msg(conv):
    if conv is None or len(conv) == 0:
        return ""
    # WildChat stores list of dicts with 'content', 'role'. First turn is user.
    msg = conv[0]
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return str(msg)

df["first_msg"] = df.conversation.map(first_user_msg)
df = df[df.first_msg.str.len() > 0]
print(f"  with non-empty first message: {len(df):,}")

# Stage 1: keyword filter
matches = df.first_msg.map(keyword_match)
df["kw_match"] = [m[0] for m in matches]
df["kw_token"] = [m[1] for m in matches]
candidates = df[df.kw_match].reset_index(drop=True)
print(f"  after keyword pre-filter: {len(candidates):,}  "
      f"({len(candidates)/max(len(df),1):.2%} of US+English)")

if len(candidates) == 0:
    print("\nNo keyword matches in this shard — try a different shard or "
          "loosen keyword list.")
    sys.exit(0)

# Stage 2: DeBERTa zero-shot
print(f"\nclassifying {len(candidates):,} candidates with DeBERTa...")
clf = ZeroShotSymptomClassifier(device=0, batch_size=32)
t1 = time.time()
results = clf.classify(candidates.first_msg.tolist())
elapsed = time.time() - t1
print(f"  done in {elapsed:.1f}s ({len(candidates)/elapsed:.0f} prompts/sec)")

candidates["p_target"] = [r.label_scores.get(TARGET_LABEL, 0.0) for r in results]
candidates["p_fiction"] = [r.label_scores.get(FICTION_LABEL, 0.0) for r in results]
candidates["is_illness"] = ((candidates.p_target >= 0.7) &
                             (candidates.p_fiction <= 0.5))

n_ill = int(candidates.is_illness.sum())
print(f"\n=== RESULTS ===")
print(f"  total US+English rows in shard:       {len(df):,}")
print(f"  passed keyword pre-filter:            {len(candidates):,}")
print(f"  classified as illness (P_t>=0.7,      {n_ill:,}")
print(f"     P_f<=0.5):")
print(f"  illness rate of US+English shard:     {n_ill/max(len(df),1):.4%}")
print(f"  illness rate of keyword passes:       {n_ill/max(len(candidates),1):.2%}")

# Date span
candidates["timestamp"] = pd.to_datetime(candidates.timestamp, utc=True).dt.tz_localize(None)
print(f"\n  classified date span: {candidates.timestamp.min()} -> {candidates.timestamp.max()}")
ill = candidates[candidates.is_illness]
if len(ill):
    print(f"  illness-positive date span:           {ill.timestamp.min()} -> {ill.timestamp.max()}")

# Show a few examples
print(f"\n=== sample illness-positive prompts ===")
for _, row in ill.head(5).iterrows():
    print(f"  [P_t={row.p_target:.2f} P_f={row.p_fiction:.2f}]  {row.first_msg[:150]!r}")

# Save the classified candidates for downstream
candidates.drop(columns=["conversation"]).to_parquet(
    OUT_DIR / "wildchat_pilot_classified.parquet")
print(f"\nsaved {OUT_DIR / 'wildchat_pilot_classified.parquet'}")
print(f"\ntotal time: {time.time()-t0:.1f}s")
