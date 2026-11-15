#!/usr/bin/env python
"""
05_checks.py  -  (A) score the hand labels, (B) rerun the main test under other settings.

A. Hand labels
   labels/hand_label_sample.csv (your 1/0 answers) is joined with data/processed/hand_label_key.csv.
   Reports how many pairs you judged plausible in each group (top network / top text / neither)
   and whether plausible pairs were combined more often in 2015+.

B. Robustness: the whole pipeline (network, backbone, synopsis text, test, nulls) is rebuilt from
   the raw data for each setting, so nothing is reused from earlier runs:
     baseline            era split 2014, all tags with rank >= 50   (must reproduce script 4)
     top-6 tags          only the 6 highest-ranked tags per anime   (same tagging density in both eras)
     split 2011 / 2017   earlier / later cut between "before" and "after"
     rank >= 70          only tags the community is very sure about
   Same fixed definitions as script 4: filled = >= 3 shared anime later, top = top 10%,
   20 popularity bins, 1,000 text permutations, 200 rewirings, seed 42.

Writes: data/processed/checks_summary.md
Usage : python 05_checks.py            (takes a few minutes)
        python 05_checks.py --labels-only
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import fisher_exact, rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = next((r for r in (HERE, HERE.parent) if (r / "data" / "raw" / "anime.jsonl").exists()), HERE)
RAW, PROC, LAB = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "labels"
SEED, FILL_MIN, TOP_FRAC, POP_BINS, TAG_BINS = 42, 3, 0.10, 20, 10
N_PERM_TEXT, N_PERM_NET = 1000, 200
MIN_ANIME, SYN_MAX, ALPHA = 15, 60, 0.05


# ---------- shared statistics (identical to script 4) ----------
def auc(score, y):
    r = rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def qbins(x, k):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
    return np.digitize(x, edges, right=True)


def top_mask(v, frac=TOP_FRAC):
    thr = np.quantile(v, 1 - frac)
    return v > 0 if thr <= 0 else v >= thr


def excess(top, y, strata):
    ns = int(strata.max()) + 1
    non = ~top
    n_non = np.bincount(strata[non], minlength=ns)
    y_non = np.bincount(strata[non], weights=y[non].astype(float), minlength=ns)
    g = top & (n_non[strata] > 0)
    if g.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rate_non = y_non / np.maximum(n_non, 1)
    obs, exp = float(y[g].mean()), float(rate_non[strata[g]].mean())
    return obs, exp, obs - exp


def pval(null, obs):
    null = np.asarray([x for x in null if not np.isnan(x)])
    return (int((null >= obs).sum()) + 1) / (len(null) + 1)


def jaccard(A, ia, ib):
    C = A @ A
    deg = A.sum(1)
    c = C[ia, ib]
    u = deg[ia] + deg[ib] - c
    return np.where(u > 0, c / np.maximum(u, 1), 0.0)


def clean_text(s):
    s = s or ""
    s = re.sub(r"~!.*?!~", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[_*~]{1,}", " ", s)
    s = re.sub(r"\((?:source|written by)[^)]*\)", " ", s, flags=re.I)
    s = re.sub(r"\[(?:source|written by)[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------- pipeline for one setting ----------
def load_anime(rank_min, top_k):
    seen, rows = set(), []
    with open(RAW / "anime.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["id"] in seen or not m.get("seasonYear"):
                continue
            seen.add(m["id"])
            good = sorted((t for t in m["tags"] if t["rank"] >= rank_min and not t["isMediaSpoiler"]),
                          key=lambda t: -t["rank"])
            if top_k:
                good = good[:top_k]
            m["tagset"] = sorted({t["name"] for t in good})
            rows.append(m)
    return rows


def run_setting(name, tags_meta, split=2014, rank_min=50, top_k=0):
    rows = load_anime(rank_min, top_k)
    era1 = [m for m in rows if m["seasonYear"] <= split]
    era2 = [m for m in rows if m["seasonYear"] > split]
    freq1 = Counter(t for m in era1 for t in m["tagset"])
    cand = sorted(t for t, c in freq1.items() if c >= MIN_ANIME and t in tags_meta
                  and len(clean_text(tags_meta[t].get("description")).split()) >= 3)
    idx = {t: i for i, t in enumerate(cand)}
    n = len(cand)
    C1, C2 = np.zeros((n, n)), np.zeros((n, n))
    for grp, C in ((era1, C1), (era2, C2)):
        for m in grp:
            ids = [idx[t] for t in m["tagset"] if t in idx]
            for i, j in combinations(ids, 2):
                C[i, j] += 1
                C[j, i] += 1

    # backbone of era 1 (disparity filter)
    deg = (C1 > 0).sum(1)
    strength = C1.sum(1)
    B = nx.Graph()
    B.add_nodes_from(range(n))
    for i, j in zip(*np.nonzero(np.triu(C1, 1))):
        w = C1[i, j]
        for a in (i, j):
            if deg[a] > 1 and (1 - w / strength[a]) ** (deg[a] - 1) < ALPHA:
                B.add_edge(int(i), int(j))
                break

    # synopsis text, era 1 only
    by_tag = defaultdict(list)
    for m in sorted(era1, key=lambda m: -(m.get("popularity") or 0)):
        syn = clean_text(m.get("description"))
        for t in m["tagset"]:
            if t in idx and len(by_tag[t]) < SYN_MAX and syn:
                by_tag[t].append(syn)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=3, max_df=0.5).fit_transform(
        [" ".join(by_tag[t]) for t in cand])
    S = (X @ X.T).toarray()

    iu, ju = np.triu_indices(n, 1)
    unl = C1[iu, ju] == 0
    ia, ib = iu[unl], ju[unl]
    y = C2[ia, ib] >= FILL_MIN
    f1 = np.array([freq1[t] for t in cand], dtype=float)
    pop = np.log(f1[ia] * f1[ib])
    net = jaccard(nx.to_numpy_array(B, nodelist=range(n)), ia, ib)
    txt = S[ia, ib]
    strata = qbins(pop, POP_BINS)

    rng = np.random.default_rng(SEED)
    tag_bin = qbins(rankdata(f1), TAG_BINS)
    groups = [np.where(tag_bin == b)[0] for b in np.unique(tag_bin)]
    o_txt = excess(top_mask(txt), y, strata)
    null_t = []
    for _ in range(N_PERM_TEXT):
        pi = np.arange(n)
        for g in groups:
            pi[g] = rng.permutation(g)
        null_t.append(excess(top_mask(S[pi[ia], pi[ib]]), y, strata)[2])

    o_net = excess(top_mask(net), y, strata)
    null_n = []
    m_e = B.number_of_edges()
    for r in range(N_PERM_NET):
        H = B.copy()
        try:
            nx.double_edge_swap(H, nswap=10 * m_e, max_tries=200 * m_e, seed=SEED + r)
        except nx.NetworkXException:
            continue
        null_n.append(excess(top_mask(jaccard(nx.to_numpy_array(H, nodelist=range(n)), ia, ib)), y, strata)[2])

    tpa = lambda g: sum(len(m["tagset"]) for m in g) / max(len(g), 1)
    res = dict(name=name, tags=n, unlinked=len(y), base=y.mean(), tpa1=tpa(era1), tpa2=tpa(era2),
               auc_pop=auc(pop, y), auc_net=auc(net, y), auc_txt=auc(txt, y),
               net=o_net, p_net=pval(null_n, o_net[2]), txt=o_txt, p_txt=pval(null_t, o_txt[2]))
    print(f"  {name}: network {o_net[2]:+.1%} (p={res['p_net']:.3f}), text {o_txt[2]:+.1%} (p={res['p_txt']:.3f})",
          flush=True)
    return res


# ---------- hand labels ----------
def read_csv_any(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        f.seek(0)
        return list(csv.DictReader(f, delimiter=";" if head.count(";") > head.count(",") else ","))


def score_labels(L):
    sample, key = LAB / "hand_label_sample.csv", PROC / "hand_label_key.csv"
    L += ["## A. Hand labels"]
    if not sample.exists() or not key.exists():
        L += [f"- not found ({sample}); run script 4 and label the sample first.", ""]
        return
    lab = {r["pair_id"].strip(): (r.get("plausible_1_or_0") or "").strip() for r in read_csv_any(sample)}
    rows = [(k["group"], lab.get(k["pair_id"], ""), int(k["filled"])) for k in read_csv_any(key)]
    done = [(g, int(v), f) for g, v, f in rows if v in ("0", "1")]
    L.append(f"- labelled pairs: {len(done)} of {len(rows)}; judged plausible: "
             f"{sum(v for _, v, _ in done)} ({np.mean([v for _, v, _ in done]):.0%})" if done else "- no labels filled in yet")
    if len(done) < 10:
        L.append("")
        return
    tab = {}
    for g in ("top_network", "top_text", "neither"):
        vs = [v for gg, v, _ in done if gg == g]
        tab[g] = (sum(vs), len(vs))
        if vs:
            L.append(f"- {g}: {sum(vs)} of {len(vs)} judged plausible ({np.mean(vs):.0%})")
    for g in ("top_network", "top_text"):
        (a, na), (b, nb) = tab[g], tab["neither"]
        if na and nb:
            _, p = fisher_exact([[a, na - a], [b, nb - b]])
            L.append(f"- {g} vs neither: Fisher exact p = {p:.3f} (two-sided; about 20 pairs per group, so only large differences can show)")
    pl = [f for _, v, f in done if v == 1]
    npl = [f for _, v, f in done if v == 0]
    if pl and npl:
        _, p = fisher_exact([[sum(pl), len(pl) - sum(pl)], [sum(npl), len(npl) - sum(npl)]])
        L.append(f"- combined in 2015+: {sum(pl)} of {len(pl)} plausible pairs ({np.mean(pl):.0%}) vs "
                 f"{sum(npl)} of {len(npl)} implausible pairs ({np.mean(npl):.0%}); Fisher exact p = {p:.3f}")
    L.append("- one labeller, labels made blind to group and outcome.")
    L.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-only", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    L = ["# Checks summary (generated by 05_checks.py)", ""]
    score_labels(L)
    print("\n".join(L))

    if not a.labels_only:
        with open(RAW / "tags.json", encoding="utf-8") as f:
            tags_meta = {t["name"]: t for t in json.load(f)}
        print("## B. Robustness (rebuilding everything per setting, a few minutes)...")
        settings = [("baseline (split 2014, rank>=50)", {}), ("top-6 tags per anime", dict(top_k=6)),
                    ("split 2011", dict(split=2011)), ("split 2017", dict(split=2017)),
                    ("rank >= 70", dict(rank_min=70))]
        res = [run_setting(nm, tags_meta, **kw) for nm, kw in settings]
        B = ["## B. Robustness of the main test",
             f"Fixed: filled = >= {FILL_MIN} shared anime after the split, top {TOP_FRAC:.0%}, {POP_BINS} popularity bins, "
             f"{N_PERM_TEXT} text permutations, {N_PERM_NET} rewirings, seed {SEED}. "
             "Cells show top-pair fill rate vs same-popularity pairs, the excess, and the permutation p.", "",
             "| setting | tags | unlinked pairs | base rate | tags/anime before, after | AUC pop / net / text | network | text (synopsis) |",
             "|---|---|---|---|---|---|---|---|"]
        for r in res:
            B.append(f"| {r['name']} | {r['tags']} | {r['unlinked']} | {r['base']:.1%} | {r['tpa1']:.1f}, {r['tpa2']:.1f} | "
                     f"{r['auc_pop']:.3f} / {r['auc_net']:.3f} / {r['auc_txt']:.3f} | "
                     f"{r['net'][0]:.1%} vs {r['net'][1]:.1%} ({r['net'][2]:+.1%}, p={r['p_net']:.3f}) | "
                     f"{r['txt'][0]:.1%} vs {r['txt'][1]:.1%} ({r['txt'][2]:+.1%}, p={r['p_txt']:.3f}) |")
        B.append("")
        B.append("- the baseline row should match script 4 (network and text excess); small p differences are possible.")
        print("\n".join(B))
        L += B
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "checks_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\nsaved data/processed/checks_summary.md")


if __name__ == "__main__":
    main()
