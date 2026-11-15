#!/usr/bin/env python
"""
03_text_similarity.py  -  how alike do two tags "sound", and how often do they co-occur?

Reads : data/raw/anime.jsonl, data/raw/tags.json
Writes: data/processed/tag_pairs.csv        one row per candidate tag pair (the table script 4 uses)
        data/processed/text_summary.md      numbers for the paper

Three similarity scores per pair (cosine similarity, 0 = nothing in common, 1 = identical):
  sim_desc_tfidf : TF-IDF on the AniList tag descriptions
  sim_desc_emb   : sentence embeddings (all-MiniLM-L6-v2, 384-dim) on the same descriptions
  sim_syn_tfidf  : TF-IDF on the synopses of the 2000-2014 anime that carry each tag

Leakage guards
  * tag names are removed from every description (otherwise "A mentions B" inflates the score)
  * the synopsis score only uses era-1 anime, so the era-2 outcome can never leak into the text
  * co-occurrence counts (era1_count, era2_count) are stored next to the scores, never mixed into them

Usage:  python src/03_text_similarity.py [--min-anime 15] [--syn-max 60]
Needs : scikit-learn, scipy.  Optional: pip install sentence-transformers  (downloads a ~90 MB model once)
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

_HERE = Path(__file__).resolve().parent
# works whether the script sits in the project folder or in its src/ subfolder
ROOT = next((r for r in (_HERE, _HERE.parent) if (r / "data" / "raw" / "anime.jsonl").exists()), _HERE.parent)
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 42
RANK_MIN = 50
ERA_SPLIT = 2014


def clean_text(s):
    s = s or ""
    s = re.sub(r"~!.*?!~", " ", s, flags=re.S)           # AniList spoiler blocks
    s = re.sub(r"<[^>]+>", " ", s)                       # html
    s = re.sub(r"[_*~]{1,}", " ", s)                     # markdown marks
    s = re.sub(r"\((?:source|written by)[^)]*\)", " ", s, flags=re.I)
    s = re.sub(r"\[(?:source|written by)[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_name_re(names, mode):
    """multi = remove multi-word tag names only (default); all = every tag name; none = nothing.
    'all' also deletes ordinary words that happen to be tags (school, magic, war), so it is only a robustness check."""
    if mode == "none":
        return None
    if mode == "multi":
        names = [n for n in names if " " in n or "-" in n]
    names = sorted(names, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I) if names else None


def strip_tag_names(text, name_re):
    if name_re is None:
        return text
    return re.sub(r"\s+", " ", name_re.sub(" ", text)).strip()


def load_anime():
    seen, rows = set(), []
    with open(RAW / "anime.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["id"] in seen or not m.get("seasonYear"):
                continue
            seen.add(m["id"])
            m["tagset"] = sorted({t["name"] for t in m["tags"]
                                  if t["rank"] >= RANK_MIN and not t["isMediaSpoiler"]})
            rows.append(m)
    return rows


def cosine_matrix(X):
    """Rows of a (sparse or dense) matrix -> full cosine-similarity matrix. TF-IDF rows are already L2-normalised."""
    S = X @ X.T
    return S.toarray() if hasattr(S, "toarray") else np.asarray(S)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-anime", type=int, default=15)
    ap.add_argument("--syn-max", type=int, default=60,
                    help="use the synopses of at most this many (most popular) era-1 anime per tag")
    ap.add_argument("--strip", choices=["multi", "all", "none"], default="multi",
                    help="which tag names to delete from the tag descriptions")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with open(RAW / "tags.json", encoding="utf-8") as f:
        tags_meta = {t["name"]: t for t in json.load(f)}
    rows = load_anime()
    era1 = [m for m in rows if m["seasonYear"] <= ERA_SPLIT]
    era2 = [m for m in rows if m["seasonYear"] > ERA_SPLIT]

    freq1 = Counter(t for m in era1 for t in m["tagset"])
    freq2 = Counter(t for m in era2 for t in m["tagset"])
    cand = sorted(t for t, c in freq1.items() if c >= a.min_anime and t in tags_meta)
    name_re = build_name_re(list(tags_meta), a.strip)
    desc_of = {t: strip_tag_names(clean_text(tags_meta[t].get("description")), name_re) for t in cand}
    dropped = [t for t in cand if len(desc_of[t].split()) < 3]
    cand = [t for t in cand if t not in set(dropped)]
    n_empty = len(dropped)
    print(f"candidate tags: {len(cand)} (dropped {n_empty} with no usable description: {', '.join(dropped) or '-'})")

    cnt1, cnt2 = Counter(), Counter()
    for grp, cnt in ((era1, cnt1), (era2, cnt2)):
        for m in grp:
            for x, y in combinations(m["tagset"], 2):
                cnt[(x, y)] += 1

    # ---- text A: tag descriptions, tag names removed ----
    desc = [desc_of[t] for t in cand]
    words = sorted(len(d.split()) for d in desc)
    print(f"description length: median {words[len(words) // 2]} words (min {words[0]}, max {words[-1]})")

    tf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1, ngram_range=(1, 2))
    S_desc = cosine_matrix(tf.fit_transform(desc))

    # ---- text A2: sentence embeddings (optional) ----
    S_emb = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        E = model.encode(desc, normalize_embeddings=True, show_progress_bar=False)
        S_emb = cosine_matrix(np.asarray(E))
        print("embeddings: done (384-dim)")
    except Exception as e:
        print("embeddings SKIPPED (", type(e).__name__, "): pip install sentence-transformers  and rerun")

    # ---- text B: synopses of era-1 anime carrying each tag ----
    by_tag = defaultdict(list)
    for m in sorted(era1, key=lambda m: -(m.get("popularity") or 0)):
        syn = clean_text(m.get("description"))
        for t in m["tagset"]:
            if t in freq1 and len(by_tag[t]) < a.syn_max and syn:
                by_tag[t].append(syn)
    syn_docs = [" ".join(by_tag[t]) for t in cand]   # synopses are plain prose: nothing stripped
    tf2 = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=3, max_df=0.5)
    S_syn = cosine_matrix(tf2.fit_transform(syn_docs))

    # ---- pair table ----
    idx = {t: i for i, t in enumerate(cand)}
    PROC.mkdir(parents=True, exist_ok=True)
    cols = ["tag_a", "tag_b", "sim_desc_tfidf", "sim_desc_emb", "sim_syn_tfidf",
            "era1_count", "era2_count", "freq1_a", "freq1_b", "freq2_a", "freq2_b", "cat_a", "cat_b"]
    table = []
    with open(PROC / "tag_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for x, y in combinations(cand, 2):
            i, j = idx[x], idx[y]
            row = [x, y, f"{S_desc[i, j]:.4f}", "" if S_emb is None else f"{S_emb[i, j]:.4f}",
                   f"{S_syn[i, j]:.4f}", cnt1.get((x, y), 0), cnt2.get((x, y), 0), freq1[x], freq1[y],
                   freq2[x], freq2[y], tags_meta[x]["category"], tags_meta[y]["category"]]
            w.writerow(row)
            table.append(row)
    n = len(table)
    print(f"pairs written: {n}")

    # ---- checks ----
    col = lambda k: np.array([float(r[cols.index(k)]) for r in table])
    c1 = col("era1_count")
    lines = ["# Text similarity summary (generated by 03_text_similarity.py)", "",
             f"- candidate tags: {len(cand)}; pairs: {n}; tags dropped for empty description: {n_empty}",
             f"- tag-name stripping mode: {a.strip}; description length median {words[len(words) // 2]} words",
             f"- synopsis text: era-1 anime only, at most {a.syn_max} most popular anime per tag"]
    sims = {"desc TF-IDF": col("sim_desc_tfidf"), "synopsis TF-IDF": col("sim_syn_tfidf")}
    if S_emb is not None:
        sims["desc embeddings"] = col("sim_desc_emb")
    names = list(sims)
    lines.append("- agreement between text methods (Spearman over all pairs):")
    for p, q in combinations(names, 2):
        r, _ = spearmanr(sims[p], sims[q])
        lines.append(f"    - {p} vs {q}: rho = {r:.3f}")
    lines.append("- does text similarity relate to co-occurrence in era 1? (Spearman with era-1 pair count)")
    for k, v in sims.items():
        r, _ = spearmanr(v, c1)
        lines.append(f"    - {k}: rho = {r:.3f}")
    gap = c1 == 0
    for k, v in sims.items():
        note = " (inflated by construction: linked tags share the same synopses)" if k.startswith("synopsis") else ""
        lines.append(f"- {k}: mean similarity of unlinked pairs {v[gap].mean():.3f} vs linked pairs "
                     f"{v[~gap].mean():.3f}{note}")
    for k, v in sims.items():
        lines.append(f"- {k}: {np.mean(v[gap] <= 1e-9):.1%} of unlinked pairs have similarity exactly 0 "
                     f"(many zeros = this score cannot rank those pairs)")

    key = "desc embeddings" if S_emb is not None else "desc TF-IDF"
    sim_key = sims[key]
    order = np.argsort(-np.where(gap, sim_key, -1))[:25]
    lines.append(f"- 25 most text-similar pairs never together in era 1 ({key}); era-2 count in brackets:")
    for o in order:
        r = table[o]
        lines.append(f"    - {r[0]} + {r[1]}: sim {float(r[cols.index('sim_desc_emb' if S_emb is not None else 'sim_desc_tfidf')]):.2f} [{r[6]}]")
    (PROC / "text_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nsaved data/processed/tag_pairs.csv and text_summary.md")


if __name__ == "__main__":
    main()