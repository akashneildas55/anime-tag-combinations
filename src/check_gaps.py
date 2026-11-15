#!/usr/bin/env python
"""check_gaps.py - a quick LOOK at the pair table (not the final test).
Run from the same folder as the other scripts:  python check_gaps.py
"filled" = the pair was never together in 2000-2014 and shares >= 3 anime in 2015+ (fixed in advance)."""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((r for r in (HERE, HERE.parent) if (r / "data" / "processed" / "tag_pairs.csv").exists()), HERE)
FILL_MIN = 3

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

with open(ROOT / "data" / "processed" / "tag_pairs.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if int(r["era1_count"]) == 0]
for r in rows:
    r["syn"] = float(r["sim_syn_tfidf"])
    r["desc"] = float(r["sim_desc_tfidf"])
    r["pop"] = int(r["freq1_a"]) * int(r["freq1_b"])
    r["filled"] = int(r["era2_count"]) >= FILL_MIN

base = sum(r["filled"] for r in rows) / len(rows)
print(f"unlinked pairs: {len(rows)}; filled later: {base:.1%} (base rate)\n")

print("25 unlinked pairs with the most similar SYNOPSES (2015+ count in brackets):")
for r in sorted(rows, key=lambda r: -r["syn"])[:25]:
    print(f"  {r['tag_a']} + {r['tag_b']}: sim {r['syn']:.2f} [{r['era2_count']}]")


def by_fifths(key, label):
    s = sorted(rows, key=lambda r: r[key])
    n = len(s)
    print(f"\nfill rate by fifths of {label} (lowest -> highest):")
    print("  " + "  ".join(f"{sum(r['filled'] for r in s[i * n // 5:(i + 1) * n // 5]) / len(s[i * n // 5:(i + 1) * n // 5]):.1%}"
                           for i in range(5)))


by_fifths("syn", "synopsis similarity")
by_fifths("pop", "tag popularity (freq A x freq B)")
zero = [r for r in rows if r["desc"] <= 1e-9]
nonzero = [r for r in rows if r["desc"] > 1e-9]
print(f"\ndescription similarity: fill rate {sum(r['filled'] for r in zero) / max(len(zero), 1):.1%} when 0, "
      f"{sum(r['filled'] for r in nonzero) / max(len(nonzero), 1):.1%} when above 0")

# does synopsis similarity still matter among pairs of similar popularity?
s = sorted(rows, key=lambda r: r["pop"])
n = len(s)
print("\nwithin each popularity fifth: fill rate of the less-similar half vs the more-similar half (synopsis):")
for i in range(5):
    grp = sorted(s[i * n // 5:(i + 1) * n // 5], key=lambda r: r["syn"])
    h = len(grp) // 2
    lo, hi = grp[:h], grp[h:]
    print(f"  popularity fifth {i + 1}: {sum(r['filled'] for r in lo) / len(lo):.1%} vs {sum(r['filled'] for r in hi) / len(hi):.1%}")
