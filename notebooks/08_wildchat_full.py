"""08_wildchat_full.py — exhaustive WildChat-4.8M illness-classifier pass.

Memory-conscious version. Per shard:
  - read only the columns we need (filtered by pyarrow at load time)
  - apply US+English filter, then extract first user message
  - drop the heavy `conversation` column immediately
  - keyword pre-filter (cheap)
  - DeBERTa zero-shot classification on candidates only
  - save per-shard parquet, gc.collect, move on

Peak memory should stay under ~1.5 GB CPU + ~1 GB GPU. Tunables:
  CLASSIFIER_BATCH = 16   (was 64; smaller = less peak GPU)
  Use --start-shard=N to resume from a specific shard.
"""
import sys, time, gc, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pyarrow.parquet as pq

from src.symptom_classifier import (
    keyword_match, ZeroShotSymptomClassifier, TARGET_LABEL, FICTION_LABEL
)

CLASSIFIER_BATCH = 16  # smaller batch -> lower peak GPU memory

SHARD_DIR = Path("data/raw/wildchat-4.8m/data")
OUT_DIR = Path("data/processed/wildchat_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--start-shard", type=int, default=0,
                help="Resume from this shard index (0-based)")
ap.add_argument("--max-shards", type=int, default=None,
                help="Stop after processing this many shards (for testing)")
args = ap.parse_args()

shards = sorted(SHARD_DIR.glob("train-*.parquet"))
print(f"found {len(shards)} shards in {SHARD_DIR}")
if args.start_shard > 0:
    print(f"resuming from shard {args.start_shard}")
    shards = shards[args.start_shard:]
if args.max_shards:
    shards = shards[: args.max_shards]
    print(f"will process at most {len(shards)} shards")

# Initialize classifier ONCE — model load is expensive
print(f"loading classifier (batch_size={CLASSIFIER_BATCH})...")
clf = ZeroShotSymptomClassifier(device=0, batch_size=CLASSIFIER_BATCH)

stats = []
t_start = time.time()

for i, shard in enumerate(shards):
    out_path = OUT_DIR / f"shard_{i:03d}.parquet"
    if out_path.exists():
        # Resume support — skip already-processed shards
        print(f"[{i+1}/{len(shards)}] {shard.name} — already processed, loading stats")
        existing = pd.read_parquet(out_path)
        stats.append({
            "shard": i, "rows_us_en": int(existing["_total_us_en"].iloc[0]),
            "rows_kw": len(existing), "rows_pos": int(existing.is_illness.sum()),
        })
        continue

    print(f"[{i+1}/{len(shards)}] {shard.name}", end=" ")
    t0 = time.time()
    cols = ["conversation_hash", "timestamp", "conversation",
            "language", "country", "state"]
    df = pq.read_table(shard, columns=cols).to_pandas()
    n_raw = len(df)

    df = df[(df.country == "United States") & (df.language == "English")]
    df = df.reset_index(drop=True)
    n_us_en = len(df)

    def first_msg(conv):
        if conv is None or len(conv) == 0:
            return ""
        m = conv[0]
        return (m.get("content", "") if isinstance(m, dict) else str(m)) or ""

    df["first_msg"] = df.conversation.map(first_msg)
    df = df[df.first_msg.str.len() > 0].reset_index(drop=True)

    matches = df.first_msg.map(keyword_match)
    df["kw_match"] = [m[0] for m in matches]
    df["kw_token"] = [m[1] for m in matches]
    cand = df[df.kw_match].reset_index(drop=True)
    n_kw = len(cand)

    if n_kw == 0:
        # Save an empty placeholder so resumption works
        empty = pd.DataFrame({"_total_us_en": [n_us_en]})
        empty.to_parquet(out_path)
        print(f"  raw={n_raw:,} us+en={n_us_en:,} kw=0  ({time.time()-t0:.1f}s)")
        stats.append({"shard": i, "rows_us_en": n_us_en, "rows_kw": 0, "rows_pos": 0})
        continue

    results = clf.classify(cand.first_msg.tolist())
    cand["p_target"]  = [r.label_scores.get(TARGET_LABEL, 0.0)  for r in results]
    cand["p_fiction"] = [r.label_scores.get(FICTION_LABEL, 0.0) for r in results]
    cand["is_illness"] = ((cand.p_target >= 0.7) & (cand.p_fiction <= 0.5))

    n_pos = int(cand.is_illness.sum())

    # Drop the heavy `conversation` field, save the rest
    cand_save = cand.drop(columns=["conversation"]).copy()
    cand_save["_total_us_en"] = n_us_en  # carry total denom for weekly agg later
    cand_save.to_parquet(out_path)
    elapsed = time.time() - t0
    print(f"  raw={n_raw:,} us+en={n_us_en:,} kw={n_kw} pos={n_pos}  ({elapsed:.1f}s)")
    stats.append({"shard": i, "rows_us_en": n_us_en, "rows_kw": n_kw, "rows_pos": n_pos})

    # Aggressive cleanup before next shard — keeps RAM headroom
    del df, cand, cand_save, results, matches
    gc.collect()

t_end = time.time()
print(f"\ntotal time: {(t_end - t_start)/60:.1f} min")

# ---- aggregate ------------------------------------------------------------
S = pd.DataFrame(stats)
print(f"\n=== corpus-wide stats ===")
print(f"  total US+English conversations:    {S.rows_us_en.sum():>12,}")
print(f"  passed keyword pre-filter:         {S.rows_kw.sum():>12,}  "
      f"({S.rows_kw.sum() / max(S.rows_us_en.sum(),1):.4%})")
print(f"  classified as illness:             {S.rows_pos.sum():>12,}  "
      f"({S.rows_pos.sum() / max(S.rows_us_en.sum(),1):.5%})")
print(f"  classified as illness (% of kw):   {S.rows_pos.sum() / max(S.rows_kw.sum(),1):.2%}")

S.to_csv(OUT_DIR / "shard_stats.csv", index=False)
print(f"\nsaved per-shard stats to {OUT_DIR}/shard_stats.csv")

# Concatenate all classified candidates for downstream use
all_cand = pd.concat(
    [pd.read_parquet(p) for p in sorted(OUT_DIR.glob("shard_*.parquet"))],
    ignore_index=True
)
all_cand = all_cand[all_cand.get("kw_match", False) == True].reset_index(drop=True) \
    if "kw_match" in all_cand.columns else all_cand
print(f"\ntotal classified candidates across corpus: {len(all_cand):,}")

if "is_illness" in all_cand.columns:
    pos = all_cand[all_cand.is_illness == True].reset_index(drop=True)
    pos.to_parquet(OUT_DIR / "_all_illness_positives.parquet")
    print(f"  saved {len(pos):,} illness-positive rows to "
          f"{OUT_DIR}/_all_illness_positives.parquet")

    # Random sample for hand inspection
    sample = pos.sample(min(50, len(pos)), random_state=42) if len(pos) else pos
    sample[["timestamp", "p_target", "p_fiction", "first_msg"]].to_csv(
        OUT_DIR / "_sample_for_inspection.csv", index=False)
    print(f"  saved random sample of {len(sample)} for hand-inspection to "
          f"{OUT_DIR}/_sample_for_inspection.csv")
