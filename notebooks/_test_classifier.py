"""Quick smoke test of symptom_classifier on synthetic prompts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.symptom_classifier import (
    keyword_match, ZeroShotSymptomClassifier, CANDIDATE_LABELS, TARGET_LABEL
)

EXAMPLES = [
    # Strong yes — first-person illness
    ("I've had a fever and sore throat for 3 days, and now my kids are sick too. Should I get tested?", "yes"),
    ("Tested positive for covid yesterday. How long until I can stop isolating?", "yes"),
    ("Lost my sense of taste this morning, and woke up with body aches. Could this be covid?", "yes"),
    # On behalf of someone else
    ("My elderly mother has had a bad cough for a week. What should I do?", "for someone"),
    # General medical, not own illness
    ("What's the difference between covid and flu in terms of symptoms?", "general"),
    # Unrelated (should fail keyword filter)
    ("Can you help me write a Python function to parse JSON?", "unrelated"),
    ("Explain the Riemann hypothesis in simple terms.", "unrelated"),
    # Tricky — keyword hits but not real self-diagnosis
    ("In my novel, a character gets covid in chapter 4. How would they describe their symptoms?", "creative writing"),
]

print("=== Stage 1: keyword filter ===")
for text, expected in EXAMPLES:
    matched, tok = keyword_match(text)
    print(f"  {'KW-PASS' if matched else 'KW-FILT'}  ({expected:<20}) tok={tok!r:<25}  {text[:70]}")

print("\n=== Stage 2: DeBERTa zero-shot on keyword passes ===")
candidates = [t for t, _ in EXAMPLES if keyword_match(t)[0]]
print(f"loading classifier (first run downloads ~360 MB)...")
clf = ZeroShotSymptomClassifier(device=0)

from src.symptom_classifier import FICTION_LABEL
results = clf.classify(candidates)
print(f"\nTarget label: {TARGET_LABEL!r}")
print(f"Fiction label: {FICTION_LABEL!r}\n")
print(f"  {'P(target)':<10} {'P(fiction)':<11} {'pass?':<6}  text")
for text, r in zip(candidates, results):
    pt = r.label_scores.get(TARGET_LABEL, 0.0)
    pf = r.label_scores.get(FICTION_LABEL, 0.0)
    passes = "YES" if (pt >= 0.7 and pf <= 0.5) else "no"
    print(f"  {pt:<10.2f} {pf:<11.2f} {passes:<6}  {text[:80]}")
