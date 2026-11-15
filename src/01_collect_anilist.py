#!/usr/bin/env python
"""
01_collect_anilist.py  -  download anime + tags from the AniList GraphQL API.

What it does
  1. tags    : downloads the full tag list (name, description, category)  -> data/raw/tags.json
  2. anime   : for every year in a range, downloads the N most popular anime
               (title, year, popularity, synopsis, tags with rank)         -> data/raw/anime.jsonl
  3. report  : prints "data counts" so you know if the project question is answerable
  Every run appends to data/raw/collection_log.md (date, settings, counts).

Usage (from the project folder):
  python src/01_collect_anilist.py                 # does tags + anime + report
  python src/01_collect_anilist.py --report-only   # only re-print the counts
  python src/01_collect_anilist.py --start 2000 --end 2025 --per-year 300

It is resumable: years already finished are skipped (data/raw/done_years.txt).
Delete that file and anime.jsonl to start from scratch.
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ENDPOINT = "https://graphql.anilist.co"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PAUSE = 2.2          # seconds between requests (gentle: ~27/min, below the 30/min worst case)
PER_PAGE = 50        # AniList maximum
RANK_MIN = 50        # a tag only counts on an anime if its rank (0-100) is at least this
ERA_SPLIT = 2014     # era 1 = year <= 2014, era 2 = year >= 2015 (decide the final cut after looking)

TAGS_QUERY = """
query { MediaTagCollection { id name description category isAdult } }
"""

ANIME_QUERY = """
query ($page: Int, $perPage: Int, $year: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage currentPage }
    media(type: ANIME, seasonYear: $year, countryOfOrigin: JP, isAdult: false,
          format_in: [TV, MOVIE, OVA, ONA], sort: POPULARITY_DESC) {
      id
      title { romaji english }
      seasonYear
      format
      popularity
      averageScore
      genres
      description(asHtml: false)
      tags { name rank isMediaSpoiler }
    }
  }
}
"""


def post(query, variables=None):
    """One polite GraphQL request with retry on rate limit."""
    import requests
    for attempt in range(6):
        r = requests.post(ENDPOINT, json={"query": query, "variables": variables or {}},
                          headers={"Accept": "application/json"}, timeout=60)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "60")) + 2
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            print(f"  server error {r.status_code}, retry in 10s", flush=True)
            time.sleep(10)
            continue
        r.raise_for_status()
        remaining = r.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) <= 2:
            time.sleep(30)
        data = r.json()
        if "errors" in data:
            raise RuntimeError(data["errors"])
        return data["data"]
    raise RuntimeError("too many retries")


def log(line):
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(RAW / "collection_log.md", "a", encoding="utf-8") as f:
        f.write(f"- {stamp}: {line}\n")


def collect_tags():
    data = post(TAGS_QUERY)["MediaTagCollection"]
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "tags.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    cats = Counter(t["category"] for t in data)
    print(f"tags: {len(data)} downloaded, {len(cats)} categories")
    log(f"Downloaded tag collection from {ENDPOINT}: {len(data)} tags, {len(cats)} categories.")
    return data


def collect_anime(start, end, per_year):
    RAW.mkdir(parents=True, exist_ok=True)
    done_file = RAW / "done_years.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    pages_needed = -(-per_year // PER_PAGE)
    total_new = 0
    for year in range(start, end + 1):
        if str(year) in done:
            print(f"{year}: already done, skipping")
            continue
        got = []
        for page in range(1, pages_needed + 1):
            d = post(ANIME_QUERY, {"page": page, "perPage": PER_PAGE, "year": year})["Page"]
            got.extend(d["media"])
            time.sleep(PAUSE)
            if not d["pageInfo"]["hasNextPage"]:
                break
        got = got[:per_year]
        with open(RAW / "anime.jsonl", "a", encoding="utf-8") as f:
            for m in got:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(done_file, "a") as f:
            f.write(f"{year}\n")
        total_new += len(got)
        print(f"{year}: {len(got)} anime", flush=True)
    log(f"Downloaded anime {start}-{end}, top {per_year} per year by popularity "
        f"(JP origin, non-adult, formats TV/MOVIE/OVA/ONA), pause {PAUSE}s between requests, "
        f"{total_new} new records this run.")


def load_anime():
    path = RAW / "anime.jsonl"
    if not path.exists():
        sys.exit("No anime.jsonl yet - run without --report-only first.")
    seen, rows = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["id"] in seen or not m.get("seasonYear"):
                continue
            seen.add(m["id"])
            m["tagset"] = sorted({t["name"] for t in m["tags"]
                                  if t["rank"] >= RANK_MIN and not t["isMediaSpoiler"]})
            rows.append(m)
    return rows


def report():
    rows = load_anime()
    by_year = Counter(m["seasonYear"] for m in rows)
    print("\n=== DATA COUNTS ===")
    print(f"anime: {len(rows)}   years: {min(by_year)}-{max(by_year)}")
    print("per year:", ", ".join(f"{y}:{n}" for y, n in sorted(by_year.items())))
    avg = sum(len(m["tagset"]) for m in rows) / len(rows)
    print(f"tags per anime (rank >= {RANK_MIN}, no spoilers): {avg:.1f} on average")

    freq = Counter(t for m in rows for t in m["tagset"])
    print(f"distinct tags used: {len(freq)}; with >= 20 anime: {sum(1 for c in freq.values() if c >= 20)}")

    e1 = [m for m in rows if m["seasonYear"] <= ERA_SPLIT]
    e2 = [m for m in rows if m["seasonYear"] > ERA_SPLIT]
    print(f"era 1 (<= {ERA_SPLIT}): {len(e1)} anime;  era 2 (> {ERA_SPLIT}): {len(e2)} anime")

    def pair_counts(group):
        c = Counter()
        for m in group:
            for a, b in combinations(m["tagset"], 2):
                c[(a, b)] += 1
        return c

    p1, p2 = pair_counts(e1), pair_counts(e2)
    f1 = Counter(t for m in e1 for t in m["tagset"])
    cand = sorted(t for t, c in f1.items() if c >= 15)
    print(f"tags with >= 15 anime in era 1 (candidates): {len(cand)}")

    unlinked = [(a, b) for a, b in combinations(cand, 2) if p1.get((a, b), 0) == 0]
    total_pairs = len(cand) * (len(cand) - 1) // 2
    print(f"candidate pairs: {total_pairs}; never together in era 1: {len(unlinked)}")
    print("of those, got linked in era 2 (n anime sharing both):")
    best = 0
    for k in (2, 3, 5):
        n = sum(1 for pr in unlinked if p2.get(pr, 0) >= k)
        print(f"   >= {k} anime: {n} pairs ({n / max(len(unlinked), 1):.1%})")
        if k == 3:
            best = n

    verdict = ("OK - enough filled pairs to test the question" if best >= 30 else
               "THIN - try a later era split, more anime per year, or a lower rank threshold" if best >= 15 else
               "TOO FEW - change the era split / settings before going on")
    print(f"\nverdict (pairs filled with >= 3 anime = {best}): {verdict}")
    log(f"Report: {len(rows)} anime, {len(freq)} tags used, {len(cand)} candidate tags, "
        f"{len(unlinked)} unlinked pairs in era 1, {best} filled in era 2 (>=3 anime). Verdict: {verdict}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--per-year", type=int, default=300)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        collect_tags()
        collect_anime(a.start, a.end, a.per_year)
    report()


if __name__ == "__main__":
    main()
