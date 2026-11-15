# What predicts which anime tags get combined next?

*A personal data project on anime tags: network analysis plus text analysis.
Why I looked at this. I wanted to see whether tags could give a streaming platform such as Crunchyroll a hint about where to commission or license next, meaning which untried combinations might be worth betting on. What I found is that tags alone are a weak guide. Popularity explains most of which combinations appear later. Shared neighbours in the tag network and similar story text each add a small signal, but not enough to point at a specific bet with confidence. I did not test whether the combinations that appear actually earn more viewers or money, so that question is still open.

## In plain words

[AniList](https://anilist.co) is a website where fans track and rate anime. Every anime there carries **tags**, short labels such as *School*, *Vampire*, *Cooking* or *Time Manipulation*. Many tags sit together all the time. Others have never appeared on the same anime, at least until some year.

This project asks a simple question: **when two tags have never been combined, can we predict which pairs will be combined later?** For example, would the first vampire detective anime have been easy to see coming?

To find out, I downloaded about 6,700 anime from 2000 to 2025 and split them into two periods, **2000-2014** and **2015-2025**. I took every pair of tags that was never combined in the first period and checked which pairs were combined in the second. Then I tested three ideas for what predicts it:

1. **Popularity.** Tags that appear on many anime meet more other tags, so they are more likely to be combined by chance.
2. **Shared neighbours.** Two tags that already appear alongside the same other tags may be easy to combine. This is the *network* idea: tags are points, and two are linked when they share an anime.
3. **Similar stories.** Two tags whose anime have similar plot summaries may be easy to combine. This is the *text* idea.

**What I found, briefly:** popularity matters most. Shared neighbours and similar stories each add a little on top, but the extra effect is small, and neither one clearly beats the other. The result is a prediction, not an explanation of why studios make these combinations.

**Who might care:** anyone interested in how popular culture mixes ideas, and, in principle, anyone choosing what to make or recommend. This project does not test whether combining a new pair of tags makes an anime more successful.

The rest of this page gives the numbers, how to reproduce them, and the limits of the analysis.

## Main findings

Among **9,200** tag pairs that no anime combined in 2000-2014, **1,214 (13.2%)** were combined in 2015-2025 (3 or more anime carrying both tags).

| Predictor | Result | Test |
|---|---|---|
| Tag popularity | Best single predictor, AUC 0.69 | Baseline |
| Shared neighbours in the tag network | Top 10% of pairs filled 18.2% of the time, against 8.9% for equally popular pairs | 200 degree-preserving rewirings, p <= 0.005 |
| Synopsis similarity | 26.4% against 22.2% | 1,000 permutations of tag texts among equally popular tags, p = 0.016 |
| Tag description similarity | 17.6% against 12.0% | Same permutation test, p = 0.003 |

- Popularity is the main driver in every setting tried.
- Beyond popularity, network overlap and synopsis similarity each add a **small but consistent** signal. Across five settings the network adds 1.7 to 9.3 percentage points and the synopsis text adds 4.1 to 7.4. Neither is reliably the bigger one.
- The tag network is one dense, loosely grouped structure (modularity 0.258 against 0.151 for rewired networks). Its groups follow audience and cast more than genre.

What did not work is reported too: tag descriptions are too short to be a reliable text source (median 10 words), and a 60-pair hand-labelled check could not separate the groups (see Limitations).

## Method in five steps

| Step | Script | What it does | Writes |
|---|---|---|---|
| 1. Collect | `src/01_collect_anilist.py` | Downloads the 300 most popular Japanese, non-adult anime per year (2000-2025) with tags, synopses and popularity, plus the full tag list. Waits 2.2 s between requests. Resumable. Prints data counts at the end. | `data/raw/`, `collection_log.md` |
| 2. Network | `src/02_build_networks.py` | Builds the tag co-occurrence network for 2000-2014 and 2015-2025. Keeps significant links with the disparity filter. Louvain communities. Nulls: 100 degree-preserving rewirings, 1,000 label shuffles for NMI against AniList's categories, 20 Louvain seeds for stability. | `data/processed/`, `figures/network_era*.png` |
| 3. Text | `src/03_text_similarity.py` | Cosine similarity between every pair of tags from synopses of 2000-2014 anime (TF-IDF) and from tag descriptions (TF-IDF, and sentence embeddings if installed). Tag names are removed from the text. | `data/processed/tag_pairs.csv` |
| 4. Test | `src/04_gap_and_tests.py` | The main test. Compares popularity, network overlap and text as predictors of which unlinked pairs get combined. Writes a blind sample of 60 pairs for hand labelling. | `data/processed/gap_summary.md`, `figures/`, `labels/` |
| 5. Checks | `src/05_checks.py` | Scores the hand labels. Rebuilds the whole analysis for four other settings (top-6 tags per anime, splits at 2011 and 2017, tag rank 70 or higher). | `data/processed/checks_summary.md` |

`src/check_gaps.py` is an optional helper that prints a quick look at the most text-similar unlinked pairs and fill rates by fifths.

### Definitions fixed before looking at test results

- **Filled:** a pair that no anime combined in 2000-2014 and that 3 or more anime combine in 2015-2025.
- **Top pairs:** the top 10% of unlinked pairs on a predictor, compared with pairs of the same popularity (20 popularity bins).
- A tag counts on an anime only if its community rank is 50 or more and it is not a spoiler tag.
- Random seed 42 everywhere. p-values are (count + 1) / (N + 1), one-sided. The minimum possible p-value is therefore 0.005 for the network test and 0.001 for the text tests.
- The robustness runs report effect sizes and permutation p-values only for the stated settings. No threshold was tuned after seeing results.

## Running it

Python 3.10 or newer.

```bash
pip install -r requirements.txt
# optional, for the second text method (large download):
pip install sentence-transformers
```

Run from the repository root, in order:

```bash
python src/01_collect_anilist.py     # about 8 minutes, needs internet
python src/02_build_networks.py      # about a minute
python src/03_text_similarity.py
python src/04_gap_and_tests.py
# now fill in labels/hand_label_sample.csv (1 or 0 per pair) before step 5,
# and do not open data/processed/hand_label_key.csv until you have finished
python src/05_checks.py              # a few minutes
```

The raw download (`data/raw/anime.jsonl`) is not committed. Script 1 recreates it, though AniList data changes over time, so numbers will differ slightly. The collection used in the results was made on 15 November 2026 (see `data/raw/collection_log.md`).

## Repository layout

```
src/                       the five scripts and the helper
data/raw/                  tags.json and the collection log (anime.jsonl is not committed)
data/processed/            tag network tables, pair table, and the summary .md files with every reported number
figures/                   network pictures and test figures
labels/                    the 60-pair hand-labelling sample
```

## Limitations

- **One source and one selection.** AniList only, and only the 300 most popular anime per year (all of them before about 2012, when fewer than 300 qualify). Tags are assigned by today's community, so older anime may be tagged less thoroughly: 6.7 tags per anime in 2000-2014 against 9.4 in 2015-2025. The top-6-tags robustness run narrows this difference (4.2 against 5.2) without removing it.
- **Prediction, not cause.** The analysis shows what predicts new combinations, not why they happen.
- **Many tests.** About ten p-values were computed. The baseline synopsis result (p = 0.016) is the weakest.
- **Hand labels were inconclusive.** One labeller, blind to group and outcome, 60 pairs (20 per group), 83% judged plausible. The differences between groups are within chance (Fisher exact p = 0.70 and 0.41).
- **Sequels.** Seasons of one franchise share character names and can inflate the synopsis similarity. The data has no franchise links to remove them.
- **Not done:** sentence embeddings as a second text method (optional step above), manga as an earlier signal, a combined model of all three predictors, and whether filling a gap pays off in popularity or score.

## Glossary

- **Tag:** a short label on an anime, such as *School* or *Vampire*.
- **Co-occur:** two tags are on the same anime.
- **Gap:** a pair of tags that no anime combined in 2000-2014. A gap is *filled* when 3 or more anime combine them in 2015-2025.
- **Tag network:** a map where each tag is a point, and two points are linked if the tags share anime.
- **Null test (shuffle test):** scramble the thing being tested many times and see how often chance alone matches the real result. A small p-value means chance rarely does.
- **AUC:** how well a score ranks the pairs that were later combined above those that were not. 0.5 is a coin flip and 1 is perfect.
- **Louvain, modularity:** a method that finds groups in a network, and a score for how clearly separated they are.
- **NMI:** how much two ways of grouping the same items agree. 0 means no agreement, 1 means identical.
- **TF-IDF, cosine similarity:** turning a text into word weights and measuring how much two texts overlap, from 0 to 1.

## Related work

- Concept graphs and link prediction for new combinations of ideas: Krenn et al., SemNet; Luo et al., [Scientific Discovery as Link Prediction in Influence and Citation Graphs](https://aclanthology.org/W18-1701/) (TextGraphs 2018); [Predicting New Research Directions in Materials Science using LLMs and Concept Graphs](https://arxiv.org/abs/2506.16824).
- Evolving tag networks in culture: [Evolutionary Dynamics of Cultural Memes and Application to Massive Movie Data](https://arxiv.org/abs/1903.02197); a [music study built on user tags](https://pmc.ncbi.nlm.nih.gov/articles/PMC5541564).
- Anime and text: [Analysis of Bipartite Networks in Anime Series](https://arxiv.org/abs/2411.03333).

## Data source and credit

Data from the [AniList GraphQL API](https://anilist.gitbook.io/anilist-apiv2-docs/), collected gently with a pause between requests. Please check AniList's terms before redistributing any raw data.

## Built with Claude

I built this with Claude (Anthropic) as a coding and analysis partner. Most of the work used Claude Sonnet 5.5, plus Claude Fable 5.2 for two code-review passes. The main chat took about 90 requests and roughly 25 million tokens, almost all of it re-reading earlier context; Claude wrote about 138,000 tokens of new text. These figures are approximate and leave out my separate Claude Code session on my own computer.

Claude drafted and tested the scripts on mock data. I ran them on the real data, made the decisions about what to keep, and did the hand labelling in `labels/` myself. Interpretations of the results were worked out together with Claude and checked against the printed output.

## Author

Akash Neil Das, MSc Human-Centered Artificial Intelligence, DTU.
