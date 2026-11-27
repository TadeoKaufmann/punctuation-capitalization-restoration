# Data

No corpus is distributed with this repository. This page describes how the training file is built.

## What training needs

One CSV, `datasets/instancias_final/instancias_final.csv`, with one instance per row:

| Column | Meaning |
|---|---|
| `Indice` | unique integer id |
| `Fragmento` | the text with its original punctuation and capitals |
| `documento` | film, briefing or article it comes from; used to split without leakage |
| `fuente` | source name (`conferencias`, `subtitulos`, `wikinoticias`, `paenza`, `sintetico`) |
| `clase` | `parte`, `1_oracion`, `2_oraciones` or `3_o_mas` |
| `bloque` | id shared by pieces of the same sentence; the split unit for `paenza` and `sintetico` |

You can train on your own text by producing this file. The split uses 5 and 7 stratified group folds, so it needs a
reasonable number of distinct documents, each class present several times.

Labels are derived from `Fragmento`: each whitespace-separated word is lowercased and stripped of punctuation to form
the input, and the stripped marks and original casing become the three labels.

## Building the original sources

| Source | Script | Input you must provide |
|---|---|---|
| Press briefings | `src/scrapear_conferencias.py` | downloads from casarosada.gob.ar |
| Film subtitles | `datasets/subtitulos/scripts/*.py`, then `src/subtitulos_a_fragmentos.py` | an OpenSubtitles API key (see `.env.example`); files listed in [film_corpus.md](film_corpus.md) |
| Wikinews | `src/descargar_wikinoticias.py`, `src/armar_wikinoticias.py`, `src/seleccionar_wikinoticias.py` | downloads from es.wikinews.org |
| Popular maths books | none | `datasets/fragmentos.csv` (columns `Indice`, `Fragmento`) |
| Synthetic speech | none | `datasets/dataset_sintetico/dataset_sintetico.jsonl` |
| All together | `src/armar_instancias_final.py` | the files above, plus Europarl v7 Spanish at `datasets/no-argentinos/europarl_es_v7/es.txt.gz` as a dictionary to filter PDF noise |

Web sources change over time, so a rebuild will not be identical to the original dataset.

## Subtitle quality checks

Many Spanish subtitle files neutralize Rioplatense dialogue (`tú` instead of `vos`) or drop the opening `¿`, which
would corrupt the labels. `analizar_subtitulos.py` measures voseo per thousand words and the ratio of `¿` to `?`.
Files with voseo below 3 per thousand or a ratio below 0.9 were rejected, as were machine translations.

## Fingerprints

Every run records a short hash of the labels and embeddings (`7bdb6554` for the published results) and of the split
(`0b10b250`). The first one hashes the bytes of `etiquetas_palabras.csv`, so it depends on line endings: it matches
with CRLF (Windows default), while the same data written with LF gives a different hash.

## Licences

The sources have different terms. Check them before redistributing anything derived from them, including trained
weights. Books and film subtitles are copyrighted.
