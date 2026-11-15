# anime-tag-combinations

Which anime tag pairs get combined next? Using the AniList API, this project builds
tag co-occurrence networks for 2000-2014 and 2015+, then tests whether tag pairs that
were never combined in the earlier era are more likely to be combined later when they
are similar in network structure or in the text of their synopses, after controlling
for tag popularity.

## Run order

Run from the project root. Requires `requests`, `numpy`, `scipy`, `scikit-learn`, `networkx` and
`matplotlib`; `sentence-transformers` is optional (used in step 03).

1. `python src/01_collect_anilist.py` - download tags and the most popular anime per year from AniList (resumable).
2. `python src/02_build_networks.py` - build backbone tag networks and communities for each era.
3. `python src/03_text_similarity.py` - compute text similarity scores per tag pair.
4. `python src/04_gap_and_tests.py` - test which predictors explain pairs filled in 2015+; writes the hand-label sample.
5. `python src/05_checks.py` - score the hand labels and rerun the main test under other settings.

`src/check_gaps.py` is an optional quick look at the pair table (run after step 03).

Note: `data/raw/anime.jsonl` and `data/raw/done_years.txt` are git-ignored; regenerate them with step 01.
