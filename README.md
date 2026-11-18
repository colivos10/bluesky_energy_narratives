# Energy Transition Narratives on Bluesky: Data

This repository contains the data for the paper:

> Varas-Fuentes, E., Olivos, C., & Moncla, L. *Energy Transition Narratives on Bluesky: A Multidimensional Computational Analysis*. Submitted to *Energy Research & Social Science*.

The study looks at how people discuss the energy transition on Bluesky. It uses relevance classification, topic modeling (BERTopic), sentiment analysis, stance detection, Named Entity Recognition (NER), and temporal analysis.

## Files

| File | Description | Rows | Columns |
|---|---|---|---|
| `scrap100pm.xz` | Raw corpus collected from Bluesky | 93,301 | 9 |
| `df_final_100pm.xz` | Filtered and labeled corpus used in the analysis | 50,107 | 16 |

Both files are CSVs compressed with xz. pandas reads them directly:

```python
import pandas as pd

raw = pd.read_csv("scrap100pm.xz")
df = pd.read_csv("df_final_100pm.xz")
```

To decompress from the terminal: `xz -dk scrap100pm.xz`

## Data collection

- **Source:** Bluesky, accessed through the AT Protocol (Authenticated Transfer Protocol) libraries.
- **Period:** January 2024 to December 2025 (24 months).
- **Queries:** 58 search queries covering technology, socioeconomic issues, environmental and social justice, and skeptical or oppositional narratives. The full list is in Appendix A of the paper.
- **Sampling:** stratified by month, with up to 100 posts per query and month ("100pm" in the file names). This keeps recent months from dominating the corpus, since the platform returns results newest first.

## `scrap100pm`: raw corpus

Every post retrieved by the queries, before any filtering. A post can appear more than once if it matched more than one query: there are 5,671 duplicated `uri` values.

| Column | Type | Description |
|---|---|---|
| `query` | string | Search query that retrieved the post |
| `uri` | string | Unique AT Protocol identifier of the post (`at://...`) |
| `text` | string | Original post text |
| `author` | string | Author handle |
| `created_at` | datetime (UTC, ISO 8601) | Publication timestamp |
| `like_count` | int | Number of likes when collected |
| `repost_count` | int | Number of reposts when collected |
| `reply_count` | int | Number of replies when collected |
| `month_ref` | string (`YYYY-MM`) | Collection month stratum |

## `df_final_100pm`: analytical corpus

The posts kept after relevance filtering and deduplication (one row per `uri`), plus the model outputs. Every post in this file also appears in `scrap100pm`, so the two files can be joined on `uri`.

**Relevance filter:** a fine-tuned `bert-base-multilingual-cased` classifier, refined with active learning (validation macro-F1 = 0.8168). Only posts predicted as relevant with probability ≥ 0.90 were kept.

The first 9 columns are the same as in `scrap100pm`. The additional columns are:

| Column | Type | Description |
|---|---|---|
| `prediccion_modelo` | int | Relevance prediction (1 = relevant). Every row is 1 |
| `confianza_modelo` | float | Predicted probability of relevance (≥ 0.90) |
| `normalized_text` | string | Preprocessed text: Unicode NFKC (Normalization Form Compatibility Composition) normalization, URLs and mentions removed, hashtags turned into plain words |
| `topic_id` | int | BERTopic topic (0 to 48). `-1` marks outlier posts (0.5% of rows), which are left out of topic-level results |
| `sentiment` | string | `negative`, `neutral`, or `positive` |
| `stance` | string | Position toward the energy transition (see below) |
| `countries` | string (list) | Countries mentioned in the post, found by NER (spaCy) and standardized with `country-converter`. `[]` means none were found |

**Stance labels and how they appear in the paper:**

| Value in the data | Label in the paper | Share |
|---|---|---|
| `pro-renewable energy and climate activism` | Support | 42.7% |
| `neutral news or technical information` | Neutral | 34.4% |
| `opposition to green energy policies and climate skepticism` | Opposition | 22.9% |

**Sentiment distribution:** negative 49.3%, neutral 29.3%, positive 21.4%.

To turn `countries` into a Python list:

```python
import ast
df["countries"] = df["countries"].apply(ast.literal_eval)
```

## Macro-topics

The 49 topics are grouped into six macro-topics (agglomerative hierarchical clustering, cosine distance, threshold 0.78):

| Macro-topic | Topic IDs |
|---|---|
| 1. Institutional Governance and Political Narratives | 4, 10, 12, 13, 14, 18, 22, 23, 25, 28, 32, 33, 36, 39, 46 |
| 2. Energy Infrastructure and Power Grid Resilience | 0, 1, 5, 6, 9, 11, 15, 19, 20, 24, 26, 27, 29, 38 |
| 3. Electric Vehicles (EVs) and Sector Economics in the UK | 8, 16, 17, 31, 37, 40, 41, 42 |
| 4. Technological Innovation and Distributed Solutions | 2, 35, 43, 45 |
| 5. Climate Justice and Anti-Mining Activism | 7, 21, 30, 44, 48 |
| 6. Offshore Wind Development and Local Tensions | 3, 34, 47 |

```python
macro = {
    1: [4, 10, 12, 13, 14, 18, 22, 23, 25, 28, 32, 33, 36, 39, 46],
    2: [0, 1, 5, 6, 9, 11, 15, 19, 20, 24, 26, 27, 29, 38],
    3: [8, 16, 17, 31, 37, 40, 41, 42],
    4: [2, 35, 43, 45],
    5: [7, 21, 30, 44, 48],
    6: [3, 34, 47],
}
topic_to_macro = {t: m for m, ts in macro.items() for t in ts}
df["macro_topic"] = df["topic_id"].map(topic_to_macro)  # NaN for topic -1
```

## Notes

- Engagement counts (likes, reposts, replies) are from the moment of collection and may have changed since.
- Bluesky users are not representative of the general population. These data describe discourse on the platform, not public opinion in general.
- All labels come from automated models and may be wrong, especially for short, ironic, or ambiguous posts.

## Citation

If you use these data, please cite the paper (full reference to be added after publication).

## Contact

Carlos Olivos, Departamento de Ingeniería Industrial, Universidad Católica del Norte: carlos.olivos@ucn.cl
