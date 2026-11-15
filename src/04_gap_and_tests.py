#!/usr/bin/env python
"""
04_gap_and_tests.py  -  what predicts which tag pairs anime combines next?

Question
  Among tag pairs never combined in 2000-2014, which ones get combined in 2015+ ?
  Three predictors compete:
    popularity : log(freq A x freq B) in 2000-2014                      (the obvious baseline)
    network    : Jaccard overlap of the two tags' neighbours in the 2000-2014 backbone
    text       : similarity of the synopses of the 2000-2014 anime carrying each tag
                 (description TF-IDF / embeddings are reported next to it as a contrast)

Fixed before looking at any test result
  filled     = pair shares >= 3 anime in 2015+
  "top" pairs = top 10% of unlinked pairs on a predictor
  statistic  = excess fill rate: fill rate of the top pairs minus the fill rate of
               non-top pairs with the same popularity (20 popularity bins)

Nulls
  text    : permute which tag owns which text, only among tags of similar popularity (1,000 times)
  network : degree-preserving rewiring of the backbone, Jaccard recomputed (200 times)
  p = (count + 1) / (N + 1), one-sided, seed fixed.

Reads : data/processed/tag_pairs.csv, tag_edges_era1.csv, data/raw/tags.json
Writes: data/processed/gap_summary.md, figures/fill_by_decile.png, figures/null_tests.png,
        labels/hand_label_sample.csv (blind, for you to fill in), data/processed/hand_label_key.csv
"""
import csv
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = next((r for r in (HERE, HERE.parent) if (r / "data" / "processed" / "tag_pairs.csv").exists()), HERE)
PROC, RAW, FIG, LAB = ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT / "figures", ROOT / "labels"
SEED = 42
FILL_MIN = 3
TOP_FRAC = 0.10
POP_BINS = 20
OTHER_BINS = 4
TAG_BINS = 10
N_PERM_TEXT = 1000
N_PERM_NET = 200


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
    """fill rate of top pairs minus fill rate of non-top pairs in the same strata."""
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
    return (int((null >= obs).sum()) + 1) / (len(null) + 1), len(null)


def jaccard(A, ia, ib):
    C = A @ A
    deg = A.sum(1)
    c = C[ia, ib]
    u = deg[ia] + deg[ib] - c
    return np.where(u > 0, c / np.maximum(u, 1), 0.0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = np.random.default_rng(SEED)

    with open(PROC / "tag_pairs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tags = sorted({r["tag_a"] for r in rows} | {r["tag_b"] for r in rows})
    idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    freq1 = np.zeros(n)
    for r in rows:
        freq1[idx[r["tag_a"]]] = int(r["freq1_a"])
        freq1[idx[r["tag_b"]]] = int(r["freq1_b"])

    text_cols = {"synopsis TF-IDF": "sim_syn_tfidf", "description TF-IDF": "sim_desc_tfidf",
                 "description embeddings": "sim_desc_emb"}
    text_cols = {k: c for k, c in text_cols.items() if all(r[c] != "" for r in rows)}
    S = {k: np.zeros((n, n)) for k in text_cols}
    for r in rows:
        i, j = idx[r["tag_a"]], idx[r["tag_b"]]
        for k, c in text_cols.items():
            S[k][i, j] = S[k][j, i] = float(r[c])

    U = [r for r in rows if int(r["era1_count"]) == 0]
    ia = np.array([idx[r["tag_a"]] for r in U])
    ib = np.array([idx[r["tag_b"]] for r in U])
    era2 = np.array([int(r["era2_count"]) for r in U])
    y = era2 >= FILL_MIN
    same_cat = np.array([r["cat_a"] == r["cat_b"] for r in U])
    pop = np.log(freq1[ia] * freq1[ib])

    edge_file = PROC / "tag_edges_era1.csv"
    if not edge_file.exists():
        sys.exit("tag_edges_era1.csv not found - run 02_build_networks.py first.")
    B = nx.Graph()
    B.add_nodes_from(range(n))
    with open(edge_file, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["in_backbone"] == "1" and r["tag_a"] in idx and r["tag_b"] in idx:
                B.add_edge(idx[r["tag_a"]], idx[r["tag_b"]])
    A = nx.to_numpy_array(B, nodelist=range(n))
    net = jaccard(A, ia, ib)

    pop_s = qbins(pop, POP_BINS)
    net_b = qbins(net, OTHER_BINS)
    main_text = "synopsis TF-IDF"
    txt = {k: S[k][ia, ib] for k in S}
    txt_b = qbins(txt[main_text], OTHER_BINS)
    strata_pop = pop_s
    strata_pop_net = pop_s * (OTHER_BINS + 1) + net_b
    strata_pop_txt = pop_s * (OTHER_BINS + 1) + txt_b

    L = ["# Gap test summary (generated by 04_gap_and_tests.py)", "",
         f"Settings: filled = >= {FILL_MIN} shared anime in 2015+; top = top {TOP_FRAC:.0%}; "
         f"{POP_BINS} popularity bins; seed {SEED}", "",
         f"- tags: {n}; pairs never combined in 2000-2014: {len(U)}; filled later: {int(y.sum())} "
         f"({y.mean():.1%} base rate)",
         f"- backbone used for the network predictor: {B.number_of_edges()} edges", "",
         "## 1. Each predictor alone (AUC: 0.5 = useless, 1 = perfect)",
         f"- popularity: {auc(pop, y):.3f}",
         f"- network (neighbour overlap): {auc(net, y):.3f}"]
    for k, v in txt.items():
        L.append(f"- text, {k}: {auc(v, y):.3f}")

    # ---------- text: tag permutation within popularity bins ----------
    tag_bin = qbins(rankdata(freq1), TAG_BINS)
    groups = [np.where(tag_bin == b)[0] for b in np.unique(tag_bin)]
    L += ["", "## 2. Does text add anything beyond popularity?  (and beyond popularity + network)",
          f"Null: tag texts permuted among tags of similar popularity, {N_PERM_TEXT} times."]
    nulls_for_fig = {}
    for k, Sk in S.items():
        v = txt[k]
        top = top_mask(v)
        o1 = excess(top, y, strata_pop)
        o2 = excess(top, y, strata_pop_net)
        a_obs = auc(v, y)
        n1, n2, na = [], [], []
        for _ in range(N_PERM_TEXT):
            pi = np.arange(n)
            for g in groups:
                pi[g] = rng.permutation(g)
            vp = Sk[pi[ia], pi[ib]]
            tp = top_mask(vp)
            n1.append(excess(tp, y, strata_pop)[2])
            n2.append(excess(tp, y, strata_pop_net)[2])
            na.append(auc(vp, y))
        p1, m1 = pval(n1, o1[2])
        p2, _ = pval(n2, o2[2])
        pa, _ = pval(na, a_obs)
        L += [f"- {k}:",
              f"    - top pairs fill at {o1[0]:.1%} vs {o1[1]:.1%} for same-popularity pairs: excess "
              f"{o1[2]:+.1%} (null mean {np.nanmean(n1):+.1%}, p = {p1:.4f}, {m1} permutations)",
              f"    - same, also matched on network overlap: {o2[0]:.1%} vs {o2[1]:.1%}: excess {o2[2]:+.1%} "
              f"(null mean {np.nanmean(n2):+.1%}, p = {p2:.4f})",
              f"    - AUC {a_obs:.3f} vs null mean {np.mean(na):.3f} (p = {pa:.4f})"]
        nulls_for_fig[f"text: {k}"] = (n1, o1[2])

    # ---------- network: degree-preserving rewiring ----------
    top_net = top_mask(net)
    o1 = excess(top_net, y, strata_pop)
    o2 = excess(top_net, y, strata_pop_txt)
    a_obs = auc(net, y)
    n1, n2, na = [], [], []
    m = B.number_of_edges()
    for r in range(N_PERM_NET):
        H = B.copy()
        try:
            nx.double_edge_swap(H, nswap=10 * m, max_tries=200 * m, seed=SEED + r)
        except nx.NetworkXException:
            continue
        vp = jaccard(nx.to_numpy_array(H, nodelist=range(n)), ia, ib)
        tp = top_mask(vp)
        n1.append(excess(tp, y, strata_pop)[2])
        n2.append(excess(tp, y, strata_pop_txt)[2])
        na.append(auc(vp, y))
    p1, m1 = pval(n1, o1[2])
    p2, _ = pval(n2, o2[2])
    pa, _ = pval(na, a_obs)
    L += ["", "## 3. Does network position add anything beyond popularity?  (and beyond popularity + text)",
          f"Null: degree-preserving rewiring of the backbone, {m1} times.",
          f"- top pairs fill at {o1[0]:.1%} vs {o1[1]:.1%} for same-popularity pairs: excess {o1[2]:+.1%} "
          f"(null mean {np.nanmean(n1):+.1%}, p = {p1:.4f})",
          f"- same, also matched on synopsis similarity: {o2[0]:.1%} vs {o2[1]:.1%}: excess {o2[2]:+.1%} "
          f"(null mean {np.nanmean(n2):+.1%}, p = {p2:.4f})",
          f"- AUC {a_obs:.3f} vs null mean {np.mean(na):.3f} (p = {pa:.4f})"]
    nulls_for_fig["network"] = (n1, o1[2])

    # ---------- robustness (effect sizes only, no new p-values) ----------
    L += ["", "## 4. Robustness: excess fill rate beyond popularity under other settings",
          "| setting | network | text (synopsis) |", "|---|---|---|"]
    v_txt = txt[main_text]
    for fm in (2, 3, 5):
        for fr in (0.05, 0.10, 0.20):
            yy = era2 >= fm
            L.append(f"| filled >= {fm}, top {fr:.0%} | {excess(top_mask(net, fr), yy, strata_pop)[2]:+.1%} | "
                     f"{excess(top_mask(v_txt, fr), yy, strata_pop)[2]:+.1%} |")
    keep = ~same_cat
    L.append(f"| different-category pairs only (n = {int(keep.sum())}) | "
             f"{excess(top_mask(net[keep]), y[keep], strata_pop[keep])[2]:+.1%} | "
             f"{excess(top_mask(v_txt[keep]), y[keep], strata_pop[keep])[2]:+.1%} |")
    if same_cat.sum() > 50:
        L.append(f"- same-category pairs fill at {y[same_cat].mean():.1%}, different-category pairs at "
                 f"{y[~same_cat].mean():.1%} (same category often means alternatives, e.g. two sports)")

    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "gap_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # ---------- figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, v in (("popularity", pop), ("network overlap", net), ("synopsis similarity", v_txt)):
            d = qbins(rankdata(v, method="ordinal"), 10)
            ax.plot(range(1, 11), [y[d == k].mean() * 100 for k in range(10)], marker="o", label=name)
        ax.axhline(y.mean() * 100, color="grey", ls="--", lw=0.8, label="base rate")
        ax.set_xlabel("decile of predictor (1 = lowest)")
        ax.set_ylabel("pairs combined in 2015+ (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "fill_by_decile.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(nulls_for_fig), figsize=(4 * len(nulls_for_fig), 3.2))
        for ax, (name, (null, obs)) in zip(np.atleast_1d(axes), nulls_for_fig.items()):
            ax.hist(np.array(null) * 100, bins=30, color="lightgrey")
            ax.axvline(obs * 100, color="crimson")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("excess fill rate beyond popularity (pp)")
        fig.tight_layout()
        fig.savefig(FIG / "null_tests.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print("figures skipped:", e)

    # ---------- blind hand-label sample ----------
    LAB.mkdir(parents=True, exist_ok=True)
    sample_file = LAB / "hand_label_sample.csv"
    if sample_file.exists():
        print("\nhand_label_sample.csv already exists - not overwritten.")
    else:
        desc = {}
        if (RAW / "tags.json").exists():
            with open(RAW / "tags.json", encoding="utf-8") as f:
                desc = {t["name"]: (t.get("description") or "") for t in json.load(f)}
        rnd = random.Random(SEED)
        top_t, top_n = top_mask(v_txt), top_net
        pools = {"top_text": np.where(top_t & ~top_n)[0], "top_network": np.where(top_n & ~top_t)[0],
                 "neither": np.where(~top_t & ~top_n)[0]}
        picks = [(g, int(i)) for g, p in pools.items() for i in rnd.sample(list(p), min(20, len(p)))]
        rnd.shuffle(picks)
        with open(sample_file, "w", newline="", encoding="utf-8") as f1, \
                open(PROC / "hand_label_key.csv", "w", newline="", encoding="utf-8") as f2:
            w1, w2 = csv.writer(f1), csv.writer(f2)
            w1.writerow(["pair_id", "tag_a", "description_a", "tag_b", "description_b", "plausible_1_or_0"])
            w2.writerow(["pair_id", "group", "filled", "era2_count"])
            for pid, (g, i) in enumerate(picks, 1):
                a, b = U[i]["tag_a"], U[i]["tag_b"]
                w1.writerow([pid, a, desc.get(a, ""), b, desc.get(b, ""), ""])
                w2.writerow([pid, g, int(y[i]), int(era2[i])])
        print(f"\nwrote {sample_file} ({len(picks)} pairs to label) - do NOT open hand_label_key.csv before labelling.")
    print("saved data/processed/gap_summary.md and figures/")


if __name__ == "__main__":
    main()
