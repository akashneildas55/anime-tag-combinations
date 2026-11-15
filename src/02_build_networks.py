#!/usr/bin/env python
"""
02_build_networks.py  -  build the tag co-occurrence network for each era.

Reads : data/raw/anime.jsonl, data/raw/tags.json
Writes: data/processed/tag_edges_era1.csv, tag_edges_era2.csv   (every pair with its weight + backbone flag)
        data/processed/tag_communities_era1.csv, tag_communities_era2.csv
        data/processed/network_summary.md                       (all numbers for the paper)
        figures/network_era1.png, network_era2.png

What the numbers mean
  weight       = number of anime (in that era) that carry both tags
  backbone     = edge survives the disparity filter (course week 6); a plain global threshold
                 is computed as the naive baseline for comparison
  Louvain      = communities on the backbone
  null for Q   = 100 degree-preserving rewirings of the backbone, Louvain again, z-score + p=(count+1)/(N+1)
  NMI          = Louvain communities vs AniList's own tag categories, against 1,000 shuffles of the labels

Usage (project folder):  python src/02_build_networks.py [--alpha 0.05] [--min-anime 15]
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC, FIG = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "figures"
SEED = 42
RANK_MIN = 50
ERA_SPLIT = 2014
N_NULL_Q = 100       # rewirings for the modularity null
N_NULL_NMI = 1000    # label shuffles for the NMI null
N_STAB = 20          # Louvain runs with different seeds (stability check)


# ---------- data ----------
def load_anime(top_k=0):
    seen, rows = set(), []
    with open(RAW / "anime.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["id"] in seen or not m.get("seasonYear"):
                continue
            seen.add(m["id"])
            good = sorted((t for t in m["tags"] if t["rank"] >= RANK_MIN and not t["isMediaSpoiler"]),
                          key=lambda t: -t["rank"])
            if top_k:
                good = good[:top_k]
            m["tagset"] = sorted({t["name"] for t in good})
            rows.append(m)
    return rows


def pair_weights(group):
    w = Counter()
    for m in group:
        for a, b in combinations(m["tagset"], 2):
            w[(a, b)] += 1
    return w


# ---------- backbone ----------
def disparity_backbone(G, alpha):
    """Serrano et al. disparity filter. Keep an edge if it is significant for at least one endpoint."""
    keep = set()
    strength = dict(G.degree(weight="weight"))
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        for node in (u, v):
            k = G.degree(node)
            if k <= 1:
                continue
            p = w / strength[node]
            if (1 - p) ** (k - 1) < alpha:
                keep.add((u, v))
                break
    B = nx.Graph()
    B.add_nodes_from(G.nodes())
    B.add_edges_from((u, v, G[u][v]) for u, v in keep)
    return B


# ---------- NMI (arithmetic-mean normalisation, same as scikit-learn's default) ----------
def nmi(a, b):
    n = len(a)
    ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
    h = lambda c: -sum(v / n * math.log(v / n) for v in c.values())
    ha, hb = h(ca), h(cb)
    mi = sum(v / n * math.log((v / n) / ((ca[x] / n) * (cb[y] / n))) for (x, y), v in cab.items())
    if ha == 0 and hb == 0:
        return 1.0
    return mi / ((ha + hb) / 2) if (ha + hb) > 0 else 0.0


def louvain(G, weighted=True):
    return nx.community.louvain_communities(G, weight="weight" if weighted else None, seed=SEED)


def era_analysis(name, group, cats, min_anime, alpha, summary):
    tag_freq = Counter(t for m in group for t in m["tagset"])
    nodes = sorted(t for t, c in tag_freq.items() if c >= min_anime)
    node_set = set(nodes)
    W = pair_weights(group)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a, b), w in W.items():
        if a in node_set and b in node_set:
            G.add_edge(a, b, weight=w)

    B = disparity_backbone(G, alpha)
    B.remove_nodes_from(list(nx.isolates(B)))
    if B.number_of_nodes() < 10 or B.number_of_edges() < 10:
        msg = (f"## {name}\n- backbone too small ({B.number_of_nodes()} nodes, {B.number_of_edges()} edges) "
               f"at alpha={alpha}; try a larger --alpha (e.g. 0.1) or smaller --min-anime.\n")
        summary.append(msg)
        print(msg)
        return

    # naive baseline: global threshold chosen to keep the same number of edges as the backbone
    ws = sorted((d["weight"] for _, _, d in G.edges(data=True)), reverse=True)
    thr = ws[min(len(B.edges()), len(ws)) - 1] if len(B.edges()) else 0
    naive = nx.Graph([(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= thr])
    b_edges = {frozenset(e) for e in B.edges()}
    n_edges = {frozenset(e) for e in naive.edges()}
    jac = len(b_edges & n_edges) / max(len(b_edges | n_edges), 1)
    tags_per_anime = sum(len(m["tagset"]) for m in group) / max(len(group), 1)

    # stability: does Louvain give the same answer with other random seeds?
    order = sorted(B.nodes())
    parts = []
    for s_ in range(N_STAB):
        cs = nx.community.louvain_communities(B, weight="weight", seed=SEED + s_)
        lab = {t: i for i, c in enumerate(cs) for t in c}
        parts.append([lab[t] for t in order])
    stab = [nmi(parts[i], parts[j]) for i, j in combinations(range(N_STAB), 2)]
    n_comms = [len(set(p)) for p in parts]

    comps = sorted(nx.connected_components(B), key=len, reverse=True)
    comms = louvain(B)
    Q = nx.community.modularity(B, comms, weight="weight")
    comms_unw = louvain(B, weighted=False)
    Qu = nx.community.modularity(B, comms_unw, weight=None)

    # null: degree-preserving rewiring of the (unweighted) backbone
    rng_seed = SEED
    null_q = []
    m_edges = B.number_of_edges()
    for r in range(N_NULL_Q):
        H = B.copy()
        try:
            nx.double_edge_swap(H, nswap=10 * m_edges, max_tries=200 * m_edges, seed=rng_seed + r)
        except nx.NetworkXException:
            continue
        null_q.append(nx.community.modularity(H, louvain(H, weighted=False), weight=None))
    n_ok = len(null_q)
    mu = sum(null_q) / n_ok if n_ok else float("nan")
    sd = (sum((x - mu) ** 2 for x in null_q) / (n_ok - 1)) ** 0.5 if n_ok > 1 else float("nan")
    z = (Qu - mu) / sd if n_ok > 1 and sd > 0 else float("nan")
    p_q = (sum(1 for x in null_q if x >= Qu) + 1) / (n_ok + 1)

    # communities vs AniList categories
    label = {}
    for ci, c in enumerate(comms):
        for t in c:
            label[t] = ci
    tags_in = sorted(label)
    full_cat = [cats.get(t, "NA") for t in tags_in]
    top_cat = [c.split("-")[0] for c in full_cat]
    comm_lab = [label[t] for t in tags_in]
    out_nmi = {}
    random.seed(SEED)
    for nm, labs in (("full category", full_cat), ("top-level category", top_cat)):
        obs = nmi(comm_lab, labs)
        shuf = []
        for _ in range(N_NULL_NMI):
            s = labs[:]
            random.shuffle(s)
            shuf.append(nmi(comm_lab, s))
        p = (sum(1 for x in shuf if x >= obs) + 1) / (N_NULL_NMI + 1)
        out_nmi[nm] = (obs, sum(shuf) / len(shuf), p)

    # bridge tags
    btw = nx.betweenness_centrality(B, weight=None)
    bridges = sorted(btw.items(), key=lambda x: -x[1])[:10]

    # write files
    PROC.mkdir(parents=True, exist_ok=True)
    with open(PROC / f"tag_edges_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag_a", "tag_b", "weight", "in_backbone"])
        for u, v, d in G.edges(data=True):
            w.writerow([u, v, d["weight"], int(B.has_edge(u, v))])
    with open(PROC / f"tag_communities_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "community", "category", "n_anime", "betweenness"])
        for t in tags_in:
            w.writerow([t, label[t], cats.get(t, "NA"), tag_freq[t], f"{btw[t]:.4f}"])

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(parents=True, exist_ok=True)
        pos = nx.spring_layout(B, seed=SEED, k=0.35)
        cmap = plt.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(11, 8))
        nx.draw_networkx_edges(B, pos, alpha=0.15, width=0.5, ax=ax)
        nx.draw_networkx_nodes(B, pos, node_size=[20 + 3 * tag_freq[t] ** 0.7 for t in B.nodes()],
                               node_color=[cmap(label[t] % 20) for t in B.nodes()], ax=ax)
        nx.draw_networkx_labels(B, pos, labels={t: t for t, _ in bridges}, font_size=8, ax=ax)
        ax.set_title(f"Tag network, {name} (backbone, Louvain communities; labels = top bridge tags)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG / f"network_{name}.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # figure is a bonus, never block on it
        print("figure skipped:", e)

    lines = [
        f"## {name}",
        f"- anime: {len(group)}; tags kept (>= {min_anime} anime): {len(nodes)}; "
        f"tags per anime: {tags_per_anime:.1f} (if the eras differ a lot, older anime are tagged less thoroughly)",
        f"- tag pairs that co-occur at least once: {G.number_of_edges()}",
        f"- disparity backbone (alpha={alpha}): {B.number_of_edges()} edges on {B.number_of_nodes()} nodes "
        f"({len(comps)} connected components, largest {len(comps[0]) if comps else 0})",
        f"- naive global-threshold baseline (weight >= {thr}): {naive.number_of_edges()} edges but only "
        f"{naive.number_of_nodes()} tags keep a link (backbone: {B.number_of_nodes()}); "
        f"edge overlap with backbone (Jaccard) = {jac:.2f}",
        f"- Louvain: {len(comms)} communities, weighted modularity Q = {Q:.3f}",
        f"- Louvain stability over {N_STAB} seeds: mean pairwise NMI = {sum(stab) / len(stab):.3f}, "
        f"communities found: {min(n_comms)}-{max(n_comms)}",
        f"- unweighted Q = {Qu:.3f} vs {n_ok} degree-preserving rewirings: mean {mu:.3f}, sd {sd:.3f}, "
        f"z = {z:.1f}, p = {p_q:.4f}",
    ]
    for nm, (obs, mean_null, p) in out_nmi.items():
        lines.append(f"- NMI(communities, AniList {nm}) = {obs:.3f} vs shuffled labels mean {mean_null:.3f} "
                     f"(p = {p:.4f}, {N_NULL_NMI} shuffles)")
    lines.append("- top bridge tags (betweenness): " + ", ".join(f"{t} ({b:.3f})" for t, b in bridges))
    lines.append("- communities (top 6 tags by frequency each):")
    for ci, c in enumerate(sorted(comms, key=len, reverse=True)):
        top = sorted(c, key=lambda t: -tag_freq[t])[:6]
        lines.append(f"    - #{ci} ({len(c)} tags): " + ", ".join(top))
    summary.extend(lines + [""])
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-anime", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=0,
                    help="keep only the k highest-ranked tags per anime (equalises tagging density between eras)")
    a = ap.parse_args()

    try:  # Windows consoles can choke on non-ASCII tag names
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    with open(RAW / "tags.json", encoding="utf-8") as f:
        cats = {t["name"]: t["category"] for t in json.load(f)}
    rows = load_anime(a.top_k)
    era1 = [m for m in rows if m["seasonYear"] <= ERA_SPLIT]
    era2 = [m for m in rows if m["seasonYear"] > ERA_SPLIT]
    summary = ["# Network summary (generated by 02_build_networks.py)", "",
               f"Settings: rank >= {RANK_MIN}, top-k {a.top_k or 'all'}, era split {ERA_SPLIT}, seed {SEED}, alpha {a.alpha}", ""]
    era_analysis("era1", era1, cats, a.min_anime, a.alpha, summary)
    era_analysis("era2", era2, cats, a.min_anime, a.alpha, summary)
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "network_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\nsaved data/processed/network_summary.md")


if __name__ == "__main__":
    main()